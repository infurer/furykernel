-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE AllowAmbiguousTypes #-}

-- | The interpreter: its arithmetic, its lanes, its errors.
module Test.FuryKernel.InterpTest (tests) where

import Data.Bits ((.&.))
import Data.Int (Int32, Int8)
import Data.List (nub)
import Data.Map.Strict qualified as Map
import Data.Word (Word16, Word32, Word8)
import FuryKernel.Build
import FuryKernel.Constraint (TgArrayOk, ThreadsOk)
import FuryKernel.IR (DimRange (..), Name)
import FuryKernel.Interp
import FuryKernel.Numerics (narrowBF16, narrowF16Double, widenBF16, widenF16)
import FuryKernel.Target (Target)
import FuryKernel.Type (BF16 (..), F16 (..))
import GHC.Float (castFloatToWord32, castWord32ToFloat, double2Float, float2Double)
import Hedgehog (Property, forAll, property, (===))
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Test.FuryKernel.Kernels (scale, scalePacked)
import Test.FuryKernel.Run (runOn)
import Test.FuryKernel.Targets (T16, T32, T64, T8)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (assertBool, testCase, (@?=))
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
  testGroup
    "Interp"
    [ testProperty "an element-wise kernel computes its spec exactly" scaleSpec
    , testProperty "packs read and write what single elements do" packsSpec
    , testCase "lanes branch and reconverge" $
        runOn @T32 lanes [] [("y", replicate 64 (VU32 0))] 0
          @?= Right (Map.fromList [("y", [VU32 (if l `mod` 32 < 3 then 1 else 2) | l <- [0 .. 63 :: Int]])])
    , testCase "a strided loop visits every element once" $
        runOn @T32 count [("n", VU32 100)] [("y", replicate 100 (VU32 0))] 0
          @?= Right (Map.fromList [("y", replicate 100 (VU32 1))])
    , testCase "SIMD-group sums give the same total on every SIMD width" $ do
        let want = Right (Map.fromList [("y", [VU32 (sum [0 .. 63])])])
        runOn @T8 total [] [("y", [VU32 0])] 0 @?= want
        runOn @T16 total [] [("y", [VU32 0])] 0 @?= want
        runOn @T32 total [] [("y", [VU32 0])] 0 @?= want
        runOn @T64 total [] [("y", [VU32 0])] 0 @?= want
    , testCase "SIMD-group sums add pairwise, neighbours first: 2^24 and 31 ones, in f32, and 2048 and 31 ones in f16" $ do
        -- In lane order every 1 rounds away; pairwise only the first does.
        let out = runOn pairwise [] [("y", replicate 32 (VF32 0)), ("h", replicate 32 (VF16 (F16 0)))] 0
        fmap (Map.lookup "y") out @?= Right (Just (replicate 32 (VF32 16777246)))
        fmap (Map.lookup "h") out @?= Right (Just (replicate 32 (VF16 (F16 0x680F))))
    , testCase "SIMD-group sums add neighbours first, not in lane order nor xor 16 first: 2^24 with ones at lanes 16 and 17" $ do
        -- Neighbours first, the two ones meet before 2^24 does: 2^24 + 2. In
        -- lane order, or with lane 16 first, each 1 meets 2^24 alone and
        -- rounds away.
        let out = runOn neighbours [] [("y", replicate 32 (VF32 0)), ("h", replicate 32 (VF16 (F16 0)))] 0
        fmap (Map.lookup "y") out @?= Right (Just (replicate 32 (VF32 16777218)))
        fmap (Map.lookup "h") out @?= Right (Just (replicate 32 (VF16 (F16 0x6801))))
    , testCase "fma rounds once: a product's rounding error, a sum that overflows when rounded twice, ties settled by low bits, subnormals" $ do
        let (ys, gs) = fmasOf (map fst fma32) (map fst fma16)
        ys @?= map snd fma32
        gs @?= map snd fma16
    , testCase "fma's special cases: NaNs, ∞ × 0 whatever the addend, ∞ - ∞, and signed zeros by IEEE 754's sum" $ do
        let (ys, gs) = fmasOf (map fst fmaSpecial32) (map fst fmaSpecial16)
        ys @?= map snd fmaSpecial32
        gs @?= map snd fmaSpecial16
    , testCase "Rsqrt, Exp, Log and Tanh: binary64's, rounded once to the operand's type, binary16 not through binary32" $ do
        let spot = [0xC19FDC5D, 0xC11967A1, 0x3E278700, 0x411F089A, 0x419FEA4A]
            xs = spot <> [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x00000001] <> [fromIntegral k * 1048573 | k <- [0 .. 4095 :: Int]]
            hs = [0 .. 0xFFFF]
            ops = [\v -> 1 / sqrt v, exp, log, tanh] :: [Double -> Double]
            (ys, gs) = transcendentalsOf xs hs
            want32 = [castFloatToWord32 (double2Float (op (float2Double (castWord32ToFloat x)))) | x <- xs, op <- ops]
            want16 = [fromIntegral h' | h <- hs, op <- ops, let F16 h' = narrowF16Double (op (float2Double (widenF16 (F16 h))))]
        take 5 [(i, a, b) | (i, a, b) <- zip3 [0 :: Int ..] ys want32, a /= b] @?= []
        take 5 [(i, a, b) | (i, a, b) <- zip3 [0 :: Int ..] gs want16, a /= b] @?= []
        -- Five of the 1243 points of [-20, 20] (a 200001-point sweep) where
        -- macOS's binary32 exp misrounds, against exp computed exactly.
        [ys !! (4 * i + 1) | i <- [0 .. 4]] @?= [0x311020AD, 0x388FC7EB, 0x3F96C052, 0x46A1FEE6, 0x4DE4E7BE]
        -- The binary16 exps and log that rounding through binary32 would
        -- round twice: 0x3C08, 0x3C18 and 0xC53C.
        [gs !! (4 * 0x1F79 + 1), gs !! (4 * 0x25CF + 1), gs !! (4 * 0x1D78 + 2)] @?= [0x3C07, 0x3C17, 0xC53B]
    , testCase "a race shows as results that depend on the seed" $ do
        let outcomes = nub [runOn @T32 racy [] [("y", [VU32 0])] seed | seed <- [1 .. 16]]
        assertBool "more than one outcome" (length outcomes > 1)
    , testCase "errors instead of undefined behaviour" $ do
        runOn @T32 count [("n", VU32 100)] [("y", replicate 99 (VU32 0))] 0
          @?= Left (ShortBuffer "y" 100 99)
        runOn @T32 uninitialised [] [("y", [VU32 0])] 0
          @?= Left (UninitializedRead "t" 1)
        runOn @T32 divide [("d", VU32 0)] [("y", [VU32 0])] 0
          @?= Left DivideByZero
        runOn @T32 pastEnd [] [("x", replicate 4 (VF32 0)), ("y", [VF32 0])] 0
          @?= Left (OutOfBounds "x" 4 4)
    , testCase "a loop whose variable stands still or wraps never ends" $ do
        let stepping start step = runOn @T32 fromBy [("start", VU32 start), ("step", VU32 step)] [("y", [VU32 0])] 0
        stepping 0 0 @?= Left (NonTermination "i")
        stepping 4294967290 7 @?= Left (NonTermination "i")
        stepping 4294967290 5 @?= Right (Map.fromList [("y", [VU32 1])])
    , testCase "the declared contract: buffers to their extents, dimensions in their range" $ do
        -- The elements past a buffer's extent come back as given.
        runOn @T32 count [("n", VU32 3)] [("y", replicate 5 (VU32 0))] 0
          @?= Right (Map.fromList [("y", map VU32 [1, 1, 1, 0, 0])])
        runOn @T32 (pastExtent 6 0) [] [("x", map VF32 [0 .. 7]), ("y", replicate 4 (VF32 0))] 0
          @?= Left (OutOfBounds "x" 6 4)
        runOn @T32 (pastExtent 2 3) [] [("x", map VF32 [0 .. 7]), ("y", replicate 4 (VF32 0))] 0
          @?= Left (OutOfBounds "y" 3 1)
        let dimmed n = runOn @T32 eights [("n", VU32 n)] [("y", replicate 64 (VU32 0))] 0
        fmap (fmap (take 2) . Map.elems) (dimmed 16) @?= Right [map VU32 [0, 1]]
        dimmed 12 @?= Left (DimOutOfRange "n" 12 (DimRange 8 64 8))
        dimmed 72 @?= Left (DimOutOfRange "n" 72 (DimRange 8 64 8))
        dimmed 0 @?= Left (DimOutOfRange "n" 0 (DimRange 8 64 8))
    , testCase "minimum and maximum: minimumNumber and maximumNumber, up to what the IR leaves to a target" $ do
        let snan = 0x7F800001
            qnan = 0x7FC00001
            f32Pairs = [(0x3F800000, snan), (snan, 0x3F800000), (0x00000000, 0x80000000), (0x80000000, 0x00000000), (qnan, 0xFF800001), (0x3F800000, 0xBF800000)]
            f16Pairs = [(0x3C00, 0x7C01), (0x7C01, 0x3C00), (0x0000, 0x8000), (0x8000, 0x0000), (0x7E01, 0xFC01), (0x3C00, 0xBC00)]
            out = floatsOf f32Pairs f16Pairs
        -- Of -0 and +0 either may come out, and two NaNs give a NaN: those by class.
        map (take 2) (chunks 4 (map leeway32 (out "out"))) @?= [[0x3F800000, 0x3F800000], [0x3F800000, 0x3F800000], [0x00000000, 0x00000000], [0x00000000, 0x00000000], [0x7FC00000, 0x7FC00000], [0xBF800000, 0x3F800000]]
        map (take 2) (chunks 4 (map leeway16 (out "out_h"))) @?= [[0x3C00, 0x3C00], [0x3C00, 0x3C00], [0x0000, 0x0000], [0x0000, 0x0000], [0x7E00, 0x7E00], [0xBC00, 0x3C00]]
    , testCase "negation and magnitude: the sign flipped and cleared, NaN payloads kept" $ do
        let out = floatsOf [(0x7F800001, 0), (0xFFC00001, 0), (0x80000000, 0)] [(0x7D00, 0), (0xFD01, 0), (0x8000, 0)]
        map (drop 2) (chunks 4 (out "out")) @?= [[0xFF800001, 0x7F800001], [0x7FC00001, 0x7FC00001], [0x00000000, 0x00000000]]
        map (drop 2) (chunks 4 (out "out_h")) @?= [[0xFD00, 0x7D00], [0x7D01, 0x7D01], [0x0000, 0x0000]]
    , testCase "i32 arithmetic wraps; a zero divisor and the quotient that overflows are errors" $ do
        let ints a b = runOn @T32 int32s [("a", VI32 a), ("b", VI32 b)] [("y", replicate 5 (VI32 0))] 0
        ints minBound (-1) @?= Right (Map.fromList [("y", map VI32 [maxBound, minBound + 1, minBound, minBound, minBound])])
        ints maxBound 2 @?= Right (Map.fromList [("y", map VI32 [minBound + 1, maxBound - 2, -2, minBound + 1, maxBound])])
        let quotient a b = runOn @T32 int32Quotient [("a", VI32 a), ("b", VI32 b)] [("y", replicate 2 (VI32 0))] 0
        quotient (-7) 2 @?= Right (Map.fromList [("y", map VI32 [-3, -1])])
        quotient 7 0 @?= Left DivideByZero
        quotient minBound (-1) @?= Left IntegerOverflow
    , testCase "an integer narrows clamped to the narrower range, or wrapped around" $ do
        let out = runOn narrowings [] ([("a", map VI32 [300, -300, -129, 127]), ("b", map VU32 [300, 255, 256, 65535])] <> [(n, replicate 4 (VI8 0)) | n <- ["clamped_a", "wrapped_a"]] <> [(n, replicate 4 (VU8 0)) | n <- ["clamped_b", "wrapped_b"]]) 0
        fmap (\m -> map (`Map.lookup` m) ["clamped_a", "wrapped_a", "clamped_b", "wrapped_b"]) out
          @?= Right [Just (map VI8 [127, -128, -128, 127]), Just (map VI8 [44, -44, 127, 127]), Just (map VU8 [255, 255, 255, 255]), Just (map VU8 [44, 255, 0, 255])]
    ]

-- | Bit for bit what the kernel's arithmetic says, NaNs included.
scaleSpec :: Property
scaleSpec = property $ do
  n <- forAll (Gen.int (Range.linear 1 300))
  s <- forAll (Gen.float (Range.linearFracFrom 0 (-4) 4))
  xs <- forAll (Gen.list (Range.singleton n) (BF16 <$> Gen.word16 Range.constantBounded))
  let want = [VBF16 (narrowBF16 (widenBF16 x * s)) | x <- xs]
      got =
        runOn @T32
          scale
          [("n", VU32 (fromIntegral n)), ("s", VF32 s)]
          [("x", map VBF16 xs), ("y", replicate n (VBF16 (BF16 0)))]
          0
  fmap (Map.lookup "y") got === Right (Just want)

-- | The packed kernel gives what the scalar one does, bit for bit.
packsSpec :: Property
packsSpec = property $ do
  packs <- forAll (Gen.int (Range.linear 1 40))
  s <- forAll (Gen.float (Range.linearFracFrom 0 (-4) 4))
  xs <- forAll (Gen.list (Range.singleton (8 * packs)) (BF16 <$> Gen.word16 Range.constantBounded))
  let ys = replicate (8 * packs) (VBF16 (BF16 0))
      scalar = runOn @T32 scale [("n", VU32 (fromIntegral (8 * packs))), ("s", VF32 s)] [("x", map VBF16 xs), ("y", ys)] 0
      packed = runOn @T32 scalePacked [("packs", VU32 (fromIntegral packs)), ("s", VF32 s)] [("x", map VBF16 xs), ("y", ys)] 0
  fmap (Map.lookup "y") packed === fmap (Map.lookup "y") scalar

-- | y = 1 in the first three lanes of each SIMD group, 2 in the others.
lanes :: forall t. (Target t, ThreadsOk t 64) => Kernel t
lanes = kernel "lanes" $ do
  y <- buffer @WriteOnly @16 @Word32 "y" 64
  threads @64
  groups 1
  pure . pure $ do
    whenLane (simdLane .< 3) (store y threadIndex 1)
    whenLane (simdLane .>= 3) (store y threadIndex 2)

-- | Adds one to every element of y.
count :: forall t. (Target t, ThreadsOk t 64) => Kernel t
count = kernel "count" $ do
  n <- dim "n" (DimRange 1 1024 1)
  y <- buffer @ReadWrite @16 @Word32 "y" n
  threads @64
  groups 1
  pure . pure $ strided "i" n $ \i -> store y i (load y i + 1)

{- | The sum of the thread indices, through SIMD-group sums and a
threadgroup array of partials.
-}
total :: forall t. (Target t, ThreadsOk t 64, TgArrayOk t 8 Word32) => Kernel t
total = kernel "total" $ do
  y <- buffer @WriteOnly @16 @Word32 "y" 1
  threads @64
  groups 1
  pure $ do
    partial <- tgArray @8 @Word32 "partial"
    pure $ do
      s <- simdSum "s" threadIndex
      whenLane (simdLane .== 0) (tgStore partial simdGroupIndex s)
      barrier
      groupCount <- simdGroupCount
      whenLane (threadIndex .== 0) $ do
        acc <- var "acc" 0
        forU "g" 0 groupCount $ \g -> acc .= get acc + tgLoad partial g
        store y 0 (get acc)

-- | Every thread stores to one address.
racy :: forall t. (Target t, ThreadsOk t 64) => Kernel t
racy = kernel "racy" $ do
  y <- buffer @WriteOnly @16 @Word32 "y" 1
  threads @64
  groups 1
  pure . pure $ store y 0 threadIndex

-- | Reads an element of threadgroup memory nobody wrote.
uninitialised :: forall t. (Target t, ThreadsOk t 32, TgArrayOk t 2 Word32) => Kernel t
uninitialised = kernel "uninitialised" $ do
  y <- buffer @WriteOnly @16 @Word32 "y" 1
  threads @32
  groups 1
  pure $ do
    t <- tgArray @2 @Word32 "t"
    pure $ do
      whenLane (threadIndex .== 0) (tgStore t 0 7)
      barrier
      whenLane (threadIndex .== 0) (store y 0 (tgLoad t 1))

-- | Divides by a parameter.
divide :: forall t. (Target t, ThreadsOk t 32) => Kernel t
divide = kernel "divide" $ do
  d <- param @Word32 "d"
  y <- buffer @WriteOnly @16 @Word32 "y" 1
  threads @32
  groups 1
  pure . pure $ whenLane (threadIndex .== 0) (store y 0 (1 `quotE` d))

-- | Lane 0 counts its loop's iterations, from @start@ by @step@ to u32's greatest.
fromBy :: forall t. (Target t, ThreadsOk t 32) => Kernel t
fromBy = kernel "from_by" $ do
  start <- param @Word32 "start"
  step <- param @Word32 "step"
  y <- buffer @WriteOnly @16 @Word32 "y" 1
  threads @32
  groups 1
  pure . pure $ whenLane (threadIndex .== 0) $ do
    n <- var "n" 0
    forStep "i" start 4294967295 step $ \_ -> n .= get n + 1
    store y 0 (get n)

-- | Reads the second pack of four from a buffer of four.
pastEnd :: Kernel T32
pastEnd = kernel "past_end" $ do
  x <- buffer @ReadOnly @16 @Float "x" 4
  y <- buffer @WriteOnly @16 @Float "y" 1
  threads @32
  groups 1
  pure . pure $ do
    v <- loadPack @4 "v" x 1
    whenLane (threadIndex .== 0) (store y 0 (sum (unpack v)))

-- | Stores x[i], of a buffer of four, to y[j], of a buffer of one.
pastExtent :: Word32 -> Word32 -> Kernel T32
pastExtent i j = kernel "past_extent" $ do
  x <- buffer @ReadOnly @16 @Float "x" 4
  y <- buffer @WriteOnly @16 @Float "y" 1
  threads @32
  groups 1
  pure . pure $ whenLane (threadIndex .== 0) (store y (fromIntegral j) (load x (fromIntegral i)))

-- | y[i] = i over a dimension of 8 to 64, a multiple of 8.
eights :: Kernel T32
eights = kernel "eights" $ do
  n <- dim "n" (DimRange 8 64 8)
  y <- buffer @WriteOnly @16 @Word32 "y" 64
  threads @32
  groups 1
  pure . pure $ strided "i" n $ \i -> store y i i

-- | The minimum, maximum, negation and magnitude of each pair, in f32 and in f16.
floats :: Kernel T32
floats = kernel "floats" $ do
  n <- dim "n" (DimRange 1 64 1)
  x <- buffer @ReadOnly @16 @Float "x" n
  y <- buffer @ReadOnly @16 @Float "y" n
  h <- buffer @ReadOnly @16 @F16 "h" n
  g <- buffer @ReadOnly @16 @F16 "g" n
  out <- buffer @WriteOnly @16 @Float "out" (n * 4)
  outH <- buffer @WriteOnly @16 @F16 "out_h" (n * 4)
  threads @32
  groups 1
  pure . pure $ strided "i" n $ \i -> do
    a <- let_ "a" (load x i)
    b <- let_ "b" (load y i)
    store out (i * 4) (minE a b)
    store out (i * 4 + 1) (maxE a b)
    store out (i * 4 + 2) (negate a)
    store out (i * 4 + 3) (abs a)
    c <- let_ "c" (load h i)
    d <- let_ "d" (load g i)
    store outH (i * 4) (minE c d)
    store outH (i * 4 + 1) (maxE c d)
    store outH (i * 4 + 2) (negate c)
    store outH (i * 4 + 3) (abs c)

-- | The outputs of 'floats' on pairs of f32 and of f16 bits, as bits.
floatsOf :: [(Word32, Word32)] -> [(Word16, Word16)] -> Name -> [Word32]
floatsOf fs hs name = case runOn floats [("n", VU32 (fromIntegral (length fs)))] inputs 0 of
  Left e -> error (show e)
  Right out -> map bits (Map.findWithDefault [] name out)
 where
  inputs =
    [ ("x", [VF32 (castWord32ToFloat a) | (a, _) <- fs])
    , ("y", [VF32 (castWord32ToFloat b) | (_, b) <- fs])
    , ("h", [VF16 (F16 a) | (a, _) <- hs])
    , ("g", [VF16 (F16 b) | (_, b) <- hs])
    , ("out", replicate (4 * length fs) (VF32 0))
    , ("out_h", replicate (4 * length hs) (VF16 (F16 0)))
    ]
  bits = \case
    VF32 v -> castFloatToWord32 v
    VF16 (F16 v) -> fromIntegral v
    v -> error (show v)

-- | What the IR leaves to a target, as one class: any NaN as 0x7FC00000 (0x7E00), -0 as +0.
leeway32, leeway16 :: Word32 -> Word32
leeway32 w
  | w .&. 0x7F800000 == 0x7F800000 && w .&. 0x007FFFFF /= 0 = 0x7FC00000
  | w == 0x80000000 = 0
  | otherwise = w
leeway16 w
  | w .&. 0x7C00 == 0x7C00 && w .&. 0x03FF /= 0 = 0x7E00
  | w == 0x8000 = 0
  | otherwise = w

chunks :: Int -> [a] -> [[a]]
chunks n = takeWhile (not . null) . map (take n) . iterate (drop n)

-- | A SIMD-group sum where the order shows: lane 0 holds 2^24 (2048 in f16), the others 1.
pairwise :: Kernel T32
pairwise = kernel "pairwise" $ do
  y <- buffer @WriteOnly @16 @Float "y" 32
  h <- buffer @WriteOnly @16 @F16 "h" 32
  threads @32
  groups 1
  pure . pure $ do
    s <- simdSum "s" (select (simdLane .== 0) 16777216 1)
    store y threadIndex s
    t <- simdSum "t" (select (simdLane .== 0) 2048 (1 :: Exp F16))
    store h threadIndex t

-- | A SIMD-group sum that only neighbours first gets right: 2^24 at lane 0 (2048 in f16), ones at lanes 16 and 17.
neighbours :: Kernel T32
neighbours = kernel "neighbours" $ do
  y <- buffer @WriteOnly @16 @Float "y" 32
  h <- buffer @WriteOnly @16 @F16 "h" 32
  threads @32
  groups 1
  pure . pure $ do
    let ones = simdLane .== 16 .|| simdLane .== 17
    s <- simdSum "s" (select (simdLane .== 0) 16777216 (select ones 1 0))
    store y threadIndex s
    t <- simdSum "t" (select (simdLane .== 0) 2048 (select ones 1 (0 :: Exp F16)))
    store h threadIndex t

-- | fma of each triple, in f32 and in f16.
fmas :: Kernel T32
fmas = kernel "fmas" $ do
  n <- dim "n" (DimRange 1 64 1)
  m <- dim "m" (DimRange 1 64 1)
  a <- buffer @ReadOnly @16 @Float "a" n
  b <- buffer @ReadOnly @16 @Float "b" n
  c <- buffer @ReadOnly @16 @Float "c" n
  ah <- buffer @ReadOnly @16 @F16 "ah" m
  bh <- buffer @ReadOnly @16 @F16 "bh" m
  ch <- buffer @ReadOnly @16 @F16 "ch" m
  y <- buffer @WriteOnly @16 @Float "y" n
  g <- buffer @WriteOnly @16 @F16 "g" m
  threads @32
  groups 1
  pure . pure $ do
    strided "i" n $ \i -> store y i (fmaE (load a i) (load b i) (load c i))
    strided "j" m $ \j -> store g j (fmaE (load ah j) (load bh j) (load ch j))

-- | The outputs of 'fmas' on triples of f32 and of f16 bits, as bits.
fmasOf :: [(Word32, Word32, Word32)] -> [(Word16, Word16, Word16)] -> ([Word32], [Word16])
fmasOf fs hs = case runOn fmas [("n", VU32 (fromIntegral (length fs))), ("m", VU32 (fromIntegral (length hs)))] inputs 0 of
  Left e -> error (show e)
  Right out -> ([castFloatToWord32 v | VF32 v <- Map.findWithDefault [] "y" out], [v | VF16 (F16 v) <- Map.findWithDefault [] "g" out])
 where
  bits = VF32 . castWord32ToFloat
  inputs =
    [ ("a", [bits x | (x, _, _) <- fs])
    , ("b", [bits x | (_, x, _) <- fs])
    , ("c", [bits x | (_, _, x) <- fs])
    , ("ah", [VF16 (F16 x) | (x, _, _) <- hs])
    , ("bh", [VF16 (F16 x) | (_, x, _) <- hs])
    , ("ch", [VF16 (F16 x) | (_, _, x) <- hs])
    , ("y", replicate (length fs) (VF32 0))
    , ("g", replicate (length hs) (VF16 (F16 0)))
    ]

{- | Triples whose fused result differs from @a * b + c@ rounded twice, or
that round at a subnormal, and the fused result.
-}
fma32 :: [((Word32, Word32, Word32), Word32)]
fma32 =
  [ -- (1 + 2^-23)^2 - (1 + 2^-22): the product's rounding error, 2^-46.
    ((0x3F800001, 0x3F800001, 0xBF800002), 0x28800000)
  , -- (1 + 2^-12)^2 - (1 + 2^-11): 2^-24, where rounding twice gives 0.
    ((0x3F800800, 0x3F800800, 0xBF801000), 0x33800000)
  , -- The largest f32 times 2, minus itself: no overflow on the way.
    ((0x7F7FFFFF, 0x40000000, 0xFF7FFFFF), 0x7F7FFFFF)
  , -- (1 + 2^-23)(1 - 2^-24) is just below the midpoint 1 + 2^-24; with 2^-46
    -- added, just above it: 1 + 2^-23, where rounding twice gives 1.
    ((0x3F800001, 0x3F7FFFFF, 0x28800000), 0x3F800001)
  , -- 2^-75 * 1.5 * 2^-75: 0.75 of the least subnormal, up to it.
    ((0x1A000000, 0x1A400000, 0x00000000), 0x00000001)
  , -- -2^-75 * 2^-76: a quarter of the least subnormal, to -0.
    ((0x9A000000, 0x19800000, 0x00000000), 0x80000000)
  , -- 2^-75 * 2^-75: half the least subnormal, a tie, to +0.
    ((0x1A000000, 0x1A000000, 0x00000000), 0x00000000)
  ]

fma16 :: [((Word16, Word16, Word16), Word16)]
fma16 =
  [ -- 29 * 2^-10 * 565 * 2^-15 = 2^-11 + 2^-25, plus 1: just above the
    -- midpoint 1 + 2^-11, so 1 + 2^-10; through f32, or from binary64's
    -- upper bits alone, the tie, to 1.
    ((0x2740, 0x246A, 0x3C00), 0x3C01)
  , -- 255 * 257 = 65535: infinity; minus 16, 65519: the largest finite.
    ((0x5BF8, 0x5C04, 0x0000), 0x7C00)
  , ((0x5BF8, 0x5C04, 0xCC00), 0x7BFF)
  , -- (1 + 2^-10)^2 - (1 + 2^-9) = 2^-20, a subnormal, where rounding twice gives 0.
    ((0x3C01, 0x3C01, 0xBC02), 0x0010)
  , -- 2^-12 * 2^-12 = 2^-24, the least subnormal; 2^-12 * 2^-13, half
    -- of it, to 0; 2^-12 * 1.5 * 2^-13, 0.75 of it, up; -2^-12 * 2^-14, to -0.
    ((0x0C00, 0x0C00, 0x0000), 0x0001)
  , ((0x0C00, 0x0800, 0x0000), 0x0000)
  , ((0x0C00, 0x0A00, 0x0000), 0x0001)
  , ((0x8C00, 0x0400, 0x0000), 0x8000)
  ]

fmaSpecial32 :: [((Word32, Word32, Word32), Word32)]
fmaSpecial32 =
  [ ((0x7F800000, 0x00000000, 0x3F800000), 0x7FC00000)
  , ((0x00000000, 0xFF800000, 0x7FC00000), 0x7FC00000)
  , ((0x7F800000, 0x3F800000, 0xFF800000), 0x7FC00000)
  , ((0x7F800000, 0x3F800000, 0x7F800000), 0x7F800000)
  , ((0x7F800000, 0xBF800000, 0x40A00000), 0xFF800000)
  , ((0x40000000, 0x40400000, 0x7F800000), 0x7F800000)
  , ((0x7FC00001, 0x00000000, 0x7F800000), 0x7FC00000)
  , -- +0 * 1 + -0 = +0; -0 * 1 + -0 = -0; -1 * +0 + -0 = -0; 1 * 1 - 1 = +0;
    -- -1 * 1 + 1 = +0; -0 * -0 + -0 = +0.
    ((0x00000000, 0x3F800000, 0x80000000), 0x00000000)
  , ((0x80000000, 0x3F800000, 0x80000000), 0x80000000)
  , ((0xBF800000, 0x00000000, 0x80000000), 0x80000000)
  , ((0x3F800000, 0x3F800000, 0xBF800000), 0x00000000)
  , ((0xBF800000, 0x3F800000, 0x3F800000), 0x00000000)
  , ((0x80000000, 0x80000000, 0x80000000), 0x00000000)
  ]

fmaSpecial16 :: [((Word16, Word16, Word16), Word16)]
fmaSpecial16 =
  [ ((0x7C00, 0x0000, 0x3C00), 0x7E00)
  , ((0x7C00, 0x3C00, 0xFC00), 0x7E00)
  , ((0xFC00, 0x3C00, 0xFC00), 0xFC00)
  , ((0x7E01, 0x3C00, 0x3C00), 0x7E00)
  , ((0x8000, 0x3C00, 0x8000), 0x8000)
  , ((0x3C00, 0x3C00, 0xBC00), 0x0000)
  ]

-- | Rsqrt, Exp, Log and Tanh of each element, in f32 and in f16.
transcendentals :: Kernel T32
transcendentals = kernel "transcendentals" $ do
  n <- dim "n" (DimRange 1 65536 1)
  m <- dim "m" (DimRange 1 65536 1)
  x <- buffer @ReadOnly @16 @Float "x" n
  h <- buffer @ReadOnly @16 @F16 "h" m
  y <- buffer @WriteOnly @16 @Float "y" (n * 4)
  g <- buffer @WriteOnly @16 @F16 "g" (m * 4)
  threads @256
  groups 1
  pure . pure $ do
    strided "i" n $ \i -> do
      a <- let_ "a" (load x i)
      store y (i * 4) (rsqrtE a)
      store y (i * 4 + 1) (expE a)
      store y (i * 4 + 2) (logE a)
      store y (i * 4 + 3) (tanhE a)
    strided "j" m $ \j -> do
      b <- let_ "b" (load h j)
      store g (j * 4) (rsqrtE b)
      store g (j * 4 + 1) (expE b)
      store g (j * 4 + 2) (logE b)
      store g (j * 4 + 3) (tanhE b)

-- | The outputs of 'transcendentals' on f32 and f16 bits, as bits.
transcendentalsOf :: [Word32] -> [Word16] -> ([Word32], [Word16])
transcendentalsOf xs hs = case runOn transcendentals [("n", VU32 (fromIntegral (length xs))), ("m", VU32 (fromIntegral (length hs)))] inputs 0 of
  Left e -> error (show e)
  Right out -> ([castFloatToWord32 v | VF32 v <- Map.findWithDefault [] "y" out], [v | VF16 (F16 v) <- Map.findWithDefault [] "g" out])
 where
  inputs =
    [ ("x", map (VF32 . castWord32ToFloat) xs)
    , ("h", map (VF16 . F16) hs)
    , ("y", replicate (4 * length xs) (VF32 0))
    , ("g", replicate (4 * length hs) (VF16 (F16 0)))
    ]

-- | Each i32 of a and u32 of b narrowed both ways.
narrowings :: Kernel T32
narrowings = kernel "narrowings" $ do
  a <- buffer @ReadOnly @16 @Int32 "a" 4
  b <- buffer @ReadOnly @16 @Word32 "b" 4
  ca <- buffer @WriteOnly @16 @Int8 "clamped_a" 4
  wa <- buffer @WriteOnly @16 @Int8 "wrapped_a" 4
  cb <- buffer @WriteOnly @16 @Word8 "clamped_b" 4
  wb <- buffer @WriteOnly @16 @Word8 "wrapped_b" 4
  threads @32
  groups 1
  pure . pure $ whenLane (threadIndex .< 4) $ do
    store ca threadIndex (narrowClamped @Int8 (load a threadIndex))
    store wa threadIndex (narrowWrapped @Int8 (load a threadIndex))
    store cb threadIndex (narrowClamped @Word8 (load b threadIndex))
    store wb threadIndex (narrowWrapped @Word8 (load b threadIndex))

-- | a + b, a - b, a * b, -a and |a| of two i32 parameters.
int32s :: Kernel T32
int32s = kernel "int32s" $ do
  a <- param @Int32 "a"
  b <- param @Int32 "b"
  y <- buffer @WriteOnly @16 @Int32 "y" 5
  threads @32
  groups 1
  pure . pure $ whenLane (threadIndex .== 0) $ do
    store y 0 (a + b)
    store y 1 (a - b)
    store y 2 (a * b)
    store y 3 (negate a)
    store y 4 (abs a)

-- | a / b and a % b of two i32 parameters.
int32Quotient :: Kernel T32
int32Quotient = kernel "int32_quotient" $ do
  a <- param @Int32 "a"
  b <- param @Int32 "b"
  y <- buffer @WriteOnly @16 @Int32 "y" 2
  threads @32
  groups 1
  pure . pure $ whenLane (threadIndex .== 0) $ do
    store y 0 (a `quotE` b)
    store y 1 (a `remE` b)
