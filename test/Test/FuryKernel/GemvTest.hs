-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE AllowAmbiguousTypes #-}

{- | The GEMV family against its f64 specification: bit for bit where f32
sums are exact, within the bound of f32 accumulation where they aren't.
A row's bits depend on the pack width and the split only: not on @m@,
on the row's place in the batch, on the rows per SIMD group or on the
threads.
-}
module Test.FuryKernel.GemvTest (tests) where

import Data.Either (fromLeft)
import Data.Map.Strict qualified as Map
import FuryKernel.Build (Kernel)
import FuryKernel.Finding (Finding (..), Rule (..))
import FuryKernel.Interp (Value (..))
import FuryKernel.Numerics (narrowBF16, widenBF16)
import FuryKernel.Target (Target, TargetCaps (..), targetCaps)
import FuryKernel.Type (BF16 (..))
import Hedgehog (Gen, Property, annotateShow, assert, failure, forAll, property, withTests, (===))
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Numeric.Natural (Natural)
import Test.FuryKernel.Gemv (GemvShape (..), GemvVariant (..), gemv)
import Test.FuryKernel.Run (runOn)
import Test.FuryKernel.Targets (T32, T8)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (testCase, (@?=))
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
  testGroup
    "Gemv"
    [ testProperty "exact sums: every variant gives its spec bit for bit, SIMD width 32" (exact @T32)
    , testProperty "exact sums: every variant gives its spec bit for bit, SIMD width 8" (exact @T8)
    , testProperty "any sums: within the bound of f32 accumulation" rounding
    , testProperty "a row's bits depend neither on m nor on its place in the batch" batchInvariant
    , testProperty "variants that differ in rows and threads give the same bits" scheduleInvariant
    , testCase "illegal variants say why" $ do
        failures (GemvVariant 1 64 8 4)
          @?= [Finding Threads "The split, 4, does not divide the 2 SIMD groups of a threadgroup."]
        map (.findingRule) (failures (GemvVariant 1 32 16 1)) @?= [Packs, Packs]
    ]
 where
  failures = fromLeft [] . gemv @T32 "gemv" (GemvShape 8 64 1)

-- | Variants with up to four SIMD groups, legal on any target here.
variants :: forall t. (Target t) => [GemvVariant]
variants =
  [ GemvVariant r (simds * width) p s
  | r <- [1, 2, 4]
  , simds <- [1, 2, 4]
  , p <- [4, 8]
  , s <- [1, 2, 4]
  , simds `mod` s == 0
  ]
 where
  width = (targetCaps @t).capSimdWidth

-- | Output rows per threadgroup.
rowsPerGroup :: forall t. (Target t) => GemvVariant -> Natural
rowsPerGroup v = v.threadsPerGroup `div` (targetCaps @t).capSimdWidth `div` v.splitK * v.rowsPerSimd

-- | A shape the variant divides, small enough for the interpreter.
shapes :: forall t. (Target t) => GemvVariant -> Natural -> Gen GemvShape
shapes v maxBatch = do
  g <- Gen.integral (Range.linear 1 2)
  c <- Gen.integral (Range.linear 1 24)
  m <- Gen.integral (Range.linear 1 maxBatch)
  pure (GemvShape (g * rowsPerGroup @t v) (c * v.packWidth) m)

-- | Multiples of 1/8 in [-1, 1]: their products and sums are exact in f32.
eighths :: Gen BF16
eighths = narrowBF16 . (/ 8) . fromIntegral <$> Gen.int (Range.linear (-8) 8)

-- | bf16 values of moderate size, as activations and weights are.
values :: Gen BF16
values = narrowBF16 <$> Gen.float (Range.linearFracFrom 0 (-2) 2)

-- | y on the interpreter, row by row of the batch.
runGemv :: forall t. (Target t) => GemvShape -> GemvVariant -> [BF16] -> [BF16] -> Either String [BF16]
runGemv shape v xs ws = case gemv @t "gemv" shape v of
  Left fs -> Left (show fs)
  Right (k :: Kernel t) ->
    case runOn k [] [("x", map VBF16 xs), ("w", map VBF16 ws), ("y", replicate (fromIntegral (shape.batch * shape.outputs)) nan)] 1 of
      Left e -> Left (show e)
      Right bufs -> Right [b | VBF16 b <- Map.findWithDefault [] "y" bufs]
 where
  nan = VBF16 (BF16 0x7FC0)

-- | The exact sums, row by row of the batch, in f64.
dots :: GemvShape -> [BF16] -> [BF16] -> [(Double, Double)]
dots shape xs ws =
  [ (sum (zipWith (*) x w), sum (map abs (zipWith (*) x w)))
  | x <- chunks (map wide xs)
  , w <- chunks (map wide ws)
  ]
 where
  wide = realToFrac . widenBF16
  chunks [] = []
  chunks zs = let (a, b) = splitAt (fromIntegral shape.inputs) zs in a : chunks b

exact :: forall t. (Target t) => Property
exact = withTests 60 $ property $ do
  v <- forAll (Gen.element (variants @t))
  shape <- forAll (shapes @t v 3)
  xs <- forAll (Gen.list (Range.singleton (fromIntegral (shape.batch * shape.inputs))) eighths)
  ws <- forAll (Gen.list (Range.singleton (fromIntegral (shape.outputs * shape.inputs))) eighths)
  runGemv @t shape v xs ws === Right [narrowBF16 (realToFrac d) | (d, _) <- dots shape xs ws]

rounding :: Property
rounding = withTests 60 $ property $ do
  v <- forAll (Gen.element (variants @T32))
  shape <- forAll (shapes @T32 v 3)
  xs <- forAll (Gen.list (Range.singleton (fromIntegral (shape.batch * shape.inputs))) values)
  ws <- forAll (Gen.list (Range.singleton (fromIntegral (shape.outputs * shape.inputs))) values)
  case runGemv @T32 shape v xs ws of
    Left e -> annotateShow e >> failure
    Right ys -> do
      let errors =
            [ (got, d, err, bound)
            | (y, (d, magnitude)) <- zip ys (dots shape xs ws)
            , let got = realToFrac (widenBF16 y) :: Double
                  err = abs (got - d)
                  -- bf16 rounding, f32 accumulation over at most k + 64
                  -- sums in a row, and bf16's subnormals.
                  accumulation = fromIntegral (shape.inputs + 64) * 2 ** (-24) * magnitude
                  bound = 2 ** (-8) * abs d + 2 * accumulation + 2 ** (-126)
            , isNaN err || err > bound
            ]
      annotateShow errors
      length ys === fromIntegral (shape.batch * shape.outputs)
      assert (null errors)

batchInvariant :: Property
batchInvariant = withTests 25 $ property $ do
  v <- forAll (Gen.element (variants @T32))
  shape <- forAll (shapes @T32 v 1)
  let k = fromIntegral shape.inputs
  row <- forAll (Gen.list (Range.singleton k) values)
  others <- forAll (Gen.list (Range.singleton (15 * k)) values)
  ws <- forAll (Gen.list (Range.singleton (fromIntegral shape.outputs * k)) values)
  let alone = runGemv @T32 shape v row ws
  batches <- traverse (\m -> (m,) <$> forAll (Gen.int (Range.linear 0 (m - 1)))) [2, 5, 16]
  let inBatch (m, at) =
        let xs = take (at * k) others <> row <> take ((m - 1 - at) * k) (drop (at * k) others)
            n = fromIntegral shape.outputs
         in take n . drop (at * n) <$> runGemv @T32 shape {batch = fromIntegral m} v xs ws
  map inBatch batches === map (const alone) batches

scheduleInvariant :: Property
scheduleInvariant = withTests 20 $ property $ do
  p <- forAll (Gen.element [4, 8])
  s <- forAll (Gen.element [1, 2, 4])
  c <- forAll (Gen.integral (Range.linear 1 16))
  let shape = GemvShape 16 (c * p) 2
      ours = [v | v <- variants @T32, v.packWidth == p, v.splitK == s]
  xs <- forAll (Gen.list (Range.singleton (fromIntegral (2 * c * p))) values)
  ws <- forAll (Gen.list (Range.singleton (fromIntegral (16 * c * p))) values)
  let results = [runGemv @T32 shape v xs ws | v <- ours]
  assert (length ours > 1)
  results === map (const (runGemv @T32 shape (GemvVariant 1 (32 * s) p s) xs ws)) results
