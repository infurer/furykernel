-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE AllowAmbiguousTypes #-}

{- | Threadgroup reductions, through an RMSNorm kernel: within one bf16 ulp
of its f64 specification on every SIMD width, and bitwise the same for a
row whatever else is in the batch. Split-K's combine, in its order.
-}
module Test.FuryKernel.ReduceTest (tests) where

import Data.Bits (testBit, (.&.))
import Data.Map.Strict qualified as Map
import Data.Proxy (Proxy (..))
import Data.Word (Word16, Word32)
import FuryKernel.Build
import FuryKernel.Check (check)
import FuryKernel.Constraint (TgArrayOk, ThreadsOk, withTgArray)
import FuryKernel.Finding (Finding (..), Rule (..))
import FuryKernel.IR (Name)
import FuryKernel.Interp (Value (..))
import FuryKernel.Numerics (narrowBF16, widenBF16)
import FuryKernel.Reduce (combineSplits, tgReduceMax, tgReduceSum)
import FuryKernel.Target (Target)
import FuryKernel.Type (BF16 (..))
import Hedgehog (Gen, Property, annotateShow, assert, forAll, property, withTests, (===))
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Numeric.Natural (Natural)
import Test.FuryKernel.Kernels (rmsnorm)
import Test.FuryKernel.Run (runOn)
import Test.FuryKernel.Targets (T32, T64, T8)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (testCase, (@?=))
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
  testGroup
    "Reduce"
    [ testProperty "rmsnorm is within one bf16 ulp of its spec, SIMD width 8" (withinUlp @T8)
    , testProperty "rmsnorm is within one bf16 ulp of its spec, SIMD width 32" (withinUlp @T32)
    , testProperty "rmsnorm is within one bf16 ulp of its spec, SIMD width 64" (withinUlp @T64)
    , testProperty "a row's result does not depend on the batch" batchInvariant
    , testCase "the maximum, over every thread" $
        fmap (Map.lookup "y") (runOn (maxOfThreads @T32) [] [("y", [VF32 0])] 0) @?= Right (Just [VF32 255])
    , testCase "the sum starts from -0, the identity of addition: a sum of -0s is -0, of +0s +0" $
        [fmap (fmap (map negativeZero) . Map.lookup "y") (runOn (sumOfThreads @T32 v) [] [("y", [VF32 1])] 0) | v <- [-0.0, 0]]
          @?= [Right (Just [True]), Right (Just [False])]
    , testCase "a reduction's result is uniform: a branch on it holds a barrier, the sum's and the maximum's, positive or not" $ do
        let stored k = fmap (Map.lookup "y") (runOn k [] [("y", [VF32 0])] 0)
        stored (branchOn @T32 False 1) @?= Right (Just [VF32 256])
        stored (branchOn @T32 False (-1)) @?= Right (Just [VF32 0])
        stored (branchOn @T32 True 3) @?= Right (Just [VF32 3])
        stored (branchOn @T32 True (-3)) @?= Right (Just [VF32 0])
    , testCase "the scratch needs a slot per SIMD group" $
        either (map (.findingRule)) (const []) (check (smallScratch @T8)) @?= [ThreadgroupMemory]
    , testCase "split-K adds the partials in split order" $
        -- ((1e8 + 1) - 1e8) + 1 is 1 in f32; the other orders give 0 or 2.
        fmap (Map.lookup "y") (runOn (splitSums @T32 8) [] [("p", map VF32 [1e8, 1, -1e8, 1, 3, 4, 5, 6]), ("y", [VF32 0, VF32 0])] 0)
          @?= Right (Just [VF32 1, VF32 18])
    , testCase "split-K's partials need a slot per split and output" $
        either (map (.findingRule)) (const []) (check (splitSums @T32 4)) @?= [ThreadgroupMemory]
    , testCase "split-K's combine needs at least one split" $
        either (map (.findingMessage)) (const []) (check (combineOf @T32 0))
          @?= ["The partials of total come in no splits: a combine needs at least one."]
    ]

hiddens :: Gen Word32
hiddens = Gen.element [64, 256, 1000]

-- | bf16 values of moderate size, as activations and weights are.
values :: Float -> Gen BF16
values r = narrowBF16 <$> Gen.float (Range.linearFracFrom 0 (-r) r)

eps :: Float
eps = 1e-6

-- | The specification, in f64, rounded once at the end.
spec :: Int -> [BF16] -> [BF16] -> [BF16]
spec h xs ws = concatMap row (chunks xs)
 where
  chunks [] = []
  chunks zs = let (a, b) = splitAt h zs in a : chunks b
  row r =
    let ds = map (realToFrac . widenBF16) r :: [Double]
        meanSq = sum (map (^ (2 :: Int)) ds) / fromIntegral h
        s = 1 / sqrt (meanSq + realToFrac eps)
     in [narrowBF16 (realToFrac (d * s * realToFrac (widenBF16 w))) | (d, w) <- zip ds ws]

-- | Distance in representable steps, across zero too.
ulps :: BF16 -> BF16 -> Int
ulps (BF16 a) (BF16 b) = abs (ordered a - ordered b)
 where
  ordered :: Word16 -> Int
  ordered w = if testBit w 15 then negate (fromIntegral (w .&. 0x7FFF)) else fromIntegral w

withinUlp :: forall t. (Target t, ThreadsOk t 256, TgArrayOk t 32 Float) => Property
withinUlp = withTests 60 $ property $ do
  h <- forAll hiddens
  rows <- forAll (Gen.int (Range.linear 1 3))
  xs <- forAll (Gen.list (Range.singleton (rows * fromIntegral h)) (values 8))
  ws <- forAll (Gen.list (Range.singleton (fromIntegral h)) (values 2))
  let got = outputs (runOn (rmsnorm @t h) (launchParams rows) (launchBuffers xs ws) 7)
      want = spec (fromIntegral h) xs ws
      worst = maximum (zipWith ulps got want)
  annotateShow worst
  length got === length want
  assert (worst <= 1)

batchInvariant :: Property
batchInvariant = withTests 30 $ property $ do
  h <- forAll hiddens
  let n = fromIntegral h
  row <- forAll (Gen.list (Range.singleton n) (values 8))
  others <- forAll (Gen.list (Range.singleton (2 * n)) (values 8))
  ws <- forAll (Gen.list (Range.singleton n) (values 2))
  let alone = outputs (runOn (rmsnorm @T32 h) (launchParams 1) (launchBuffers row ws) 1)
      batch = take n others <> row <> drop n others
      inBatch = outputs (runOn (rmsnorm @T32 h) (launchParams 3) (launchBuffers batch ws) 2)
  take n (drop n inBatch) === alone

launchParams :: Int -> [(Name, Value)]
launchParams rows = [("rows", VU32 (fromIntegral rows)), ("eps", VF32 eps)]

launchBuffers :: [BF16] -> [BF16] -> [(Name, [Value])]
launchBuffers xs ws =
  [ ("x", map VBF16 xs)
  , ("w", map VBF16 ws)
  , ("y", map (const (VBF16 (BF16 0x7FC0))) xs)
  ]

outputs :: Either e (Map.Map Name [Value]) -> [BF16]
outputs = \case
  Right m -> [b | VBF16 b <- Map.findWithDefault [] "y" m]
  Left _ -> []

-- | The largest thread index, as a float.
maxOfThreads :: forall t. (Target t, ThreadsOk t 256, TgArrayOk t 32 Float) => Kernel t
maxOfThreads = kernel "max_of_threads" $ do
  y <- buffer @WriteOnly @16 @Float "y" 1
  threads @256
  groups 1
  pure $ do
    scratch <- tgArray @32 @Float "scratch"
    pure $ do
      m <- tgReduceMax "largest" scratch (toF32 threadIndex)
      whenLane (threadIndex .== 0) (store y 0 m)

-- | The sum of a value every thread holds.
sumOfThreads :: forall t. (Target t, ThreadsOk t 256, TgArrayOk t 32 Float) => Float -> Kernel t
sumOfThreads v = kernel "sum_of_threads" $ do
  y <- buffer @WriteOnly @16 @Float "y" 1
  threads @256
  groups 1
  pure $ do
    scratch <- tgArray @32 @Float "scratch"
    pure $ do
      s <- tgReduceSum "total" scratch (f32 v)
      whenLane (threadIndex .== 0) (store y 0 s)

negativeZero :: Value -> Bool
negativeZero = \case
  VF32 x -> isNegativeZero x
  _ -> False

{- | Every thread's @v@ reduced, by the sum or by the maximum, and a branch
on the result declared uniform, holding a barrier: thread 0 stores the
result where it is positive.
-}
branchOn :: forall t. (Target t, ThreadsOk t 256, TgArrayOk t 32 Float) => Bool -> Float -> Kernel t
branchOn largest v = kernel "branch_on" $ do
  y <- buffer @WriteOnly @16 @Float "y" 1
  threads @256
  groups 1
  pure $ do
    scratch <- tgArray @32 @Float "scratch"
    pure $ do
      r <- (if largest then tgReduceMax else tgReduceSum) "total" scratch (f32 v)
      whenU (r .> 0) $ do
        barrier
        whenLane (threadIndex .== 0) (store y 0 r)

-- | A reduction whose scratch is too small for 32 SIMD groups.
smallScratch :: forall t. (Target t, ThreadsOk t 256, TgArrayOk t 4 Float) => Kernel t
smallScratch = kernel "small_scratch" $ do
  y <- buffer @WriteOnly @16 @Float "y" 1
  threads @256
  groups 1
  pure $ do
    scratch <- tgArray @4 @Float "scratch"
    pure $ do
      s <- tgReduceSum "total" scratch 1
      whenLane (threadIndex .== 0) (store y 0 s)

-- | Split-K's combine of @splits@ splits of two outputs, from eight slots.
combineOf :: forall t. (Target t, ThreadsOk t 32, TgArrayOk t 8 Float) => Natural -> Kernel t
combineOf splits = kernel "combine_of" $ do
  y <- buffer @WriteOnly @16 @Float "y" 2
  threads @32
  groups 1
  pure $ do
    partials <- tgArray @8 @Float "partials"
    pure $ do
      whenLane (threadIndex .< 8) (tgStore partials threadIndex 0)
      combineSplits "total" partials splits 2 (store y)

-- | Two outputs of four splits each, from @p@, with @slots@ partials.
splitSums :: forall t. (Target t, ThreadsOk t 32) => Natural -> Kernel t
splitSums slots = either (error . show) id $ withTgArray @t @Float slots $ \(_ :: Proxy n) ->
  kernel "split_sums" $ do
    p <- buffer @ReadOnly @16 @Float "p" 8
    y <- buffer @WriteOnly @16 @Float "y" 2
    threads @32
    groups 1
    pure $ do
      partials <- tgArray @n @Float "partials"
      pure $ do
        whenLane (threadIndex .< 8) (tgStore partials threadIndex (load p threadIndex))
        combineSplits "total" partials 4 2 (store y)
