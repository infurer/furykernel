-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE AllowAmbiguousTypes #-}

{- | Fragments: products of the 8×8 matrices a SIMD group holds, from
device and threadgroup memory, on the interpreter against their meaning in
f64, and the gate.
-}
module Test.FuryKernel.FragmentTest (tests) where

import Control.Monad (forM_)
import Data.Map.Strict qualified as Map
import Data.Word (Word32)
import FuryKernel.Build
import FuryKernel.Check (check)
import FuryKernel.Constraint (AlignOk, TgArrayOk, ThreadsOk)
import FuryKernel.Finding (Finding (..), Rule (..))
import FuryKernel.IR (DimRange (..))
import FuryKernel.Interp (Value (..))
import FuryKernel.Numerics (narrowBF16, narrowF16, widenBF16, widenF16)
import FuryKernel.Target (Target)
import FuryKernel.Type (BF16, F16, IsScalar)
import GHC.Float (float2Double)
import Test.FuryKernel.Run (runOn)
import Test.FuryKernel.Targets (T16, T32, T64)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (assertBool, assertFailure, testCase)

tests :: TestTree
tests =
  testGroup
    "fragments"
    [ testCase "a product of bf16 fragments into f32 is its f64 meaning, to f32's accumulation" $
        agrees (gemm @BF16) (VBF16 . narrowBF16) (\case VBF16 b -> widenBF16 b; _ -> 0 / 0)
    , testCase "of f16 fragments" $
        agrees (gemm @F16) (VF16 . narrowF16) (\case VF16 h -> widenF16 h; _ -> 0 / 0)
    , testCase "of f32 fragments" $
        agrees (gemm @Float) VF32 (\case VF32 x -> x; _ -> 0 / 0)
    , testCase "of fragments loaded from threadgroup memory, transposed" $
        agrees (staged @BF16) (VBF16 . narrowBF16) (\case VBF16 b -> widenBF16 b; _ -> 0 / 0)
    , testCase "of f32 fragments by bf16 ones widened as loaded, element by element" (mixedAgrees mixed 1)
    , testCase "an accumulator's rows scaled, each by its own" (mixedAgrees scaledRows 2)
    , testCase "a target without fragments takes none" $
        case check (gemm @BF16 @T16) of
          Left fs -> assertBool (show fs) (any ((== Fragments) . (.findingRule)) fs)
          Right _ -> assertFailure "T16 has no fragments"
    , testCase "a target takes products of the types it has, and only those" $ do
        case check (gemm @BF16 @T64) of
          Left fs -> assertBool (show fs) (any ((== Fragments) . (.findingRule)) fs)
          Right _ -> assertFailure "T64 has no bf16 fragments"
        either (assertFailure . show) (const (pure ())) (check (gemm @F16 @T64))
    ]

{- | C = A · Bᵀ, A and B 16 rows of k: four SIMD groups of a threadgroup an
8×8 tile of C each, over k in steps of 8, B's tiles loaded transposed.
-}
gemm :: forall a t. (Target t, IsScalar a, AlignOk a 16, FragInput a, ThreadsOk t 128) => Kernel t
gemm = kernel "gemm" $ do
  k <- dim "k" (DimRange 8 256 8)
  a <- buffer @ReadOnly @16 @a "a" (16 * k)
  b <- buffer @ReadOnly @16 @a "b" (16 * k)
  c <- buffer @WriteOnly @16 @Float "c" 256
  threads @128
  groups 1
  pure . pure $ do
    row0 <- let_ "row0" ((simdGroupIndex `quotE` 2) * 8)
    col0 <- let_ "col0" ((simdGroupIndex `remE` 2) * 8)
    acc <- fragment @Float "acc"
    fa <- fragment @a "fa"
    fb <- fragment @a "fb"
    forU "p" 0 (k `quotE` 8) $ \p -> do
      fragLoad fa a (row0 * k + p * 8) k
      fragLoadT fb b (col0 * k + p * 8) k
      fragMma acc fa fb acc
    fragStore c (row0 * 16 + col0) 16 acc

{- | 'gemm' with B's tiles staged in threadgroup memory: a step's 16 rows of
8 elements each, a thread an element, and loaded from there transposed.
-}
staged :: forall a t. (Target t, IsScalar a, AlignOk a 16, FragInput a, ThreadsOk t 128, TgArrayOk t 128 a) => Kernel t
staged = kernel "staged" $ do
  k <- dim "k" (DimRange 8 256 8)
  a <- buffer @ReadOnly @16 @a "a" (16 * k)
  b <- buffer @ReadOnly @16 @a "b" (16 * k)
  c <- buffer @WriteOnly @16 @Float "c" 256
  threads @128
  groups 1
  pure $ do
    tile <- tgArray @128 @a "tile"
    pure $ do
      row0 <- let_ "row0" ((simdGroupIndex `quotE` 2) * 8)
      col0 <- let_ "col0" ((simdGroupIndex `remE` 2) * 8)
      acc <- fragment @Float "acc"
      fa <- fragment @a "fa"
      fb <- fragment @a "fb"
      forU "p" 0 (k `quotE` 8) $ \p -> do
        tgStore tile threadIndex (load b ((threadIndex `quotE` 8) * k + p * 8 + threadIndex `remE` 8))
        barrier
        fragLoad fa a (row0 * k + p * 8) k
        fragLoadTgT fb tile (col0 * 8) 8
        fragMma acc fa fb acc
        barrier
      fragStore c (row0 * 16 + col0) 16 acc

{- | 'gemm' of an f32 A by a bf16 B, B's tiles loaded as bf16 fragments and
widened into f32 ones before they multiply.
-}
mixed :: Kernel T32
mixed = kernel "mixed" $ do
  k <- dim "k" (DimRange 8 256 8)
  a <- buffer @ReadOnly @16 @Float "a" (16 * k)
  b <- buffer @ReadOnly @16 @BF16 "b" (16 * k)
  c <- buffer @WriteOnly @16 @Float "c" 256
  threads @128
  groups 1
  pure . pure $ do
    row0 <- let_ "row0" ((simdGroupIndex `quotE` 2) * 8)
    col0 <- let_ "col0" ((simdGroupIndex `remE` 2) * 8)
    acc <- fragment @Float "acc"
    fa <- fragment @Float "fa"
    fbh <- fragment @BF16 "fbh"
    fb <- fragment @Float "fb"
    forU "p" 0 (k `quotE` 8) $ \p -> do
      fragLoad fa a (row0 * k + p * 8) k
      fragLoadT fbh b (col0 * k + p * 8) k
      fragConvert fb fbh
      fragMma acc fa fb acc
    fragStore c (row0 * 16 + col0) 16 acc

-- | 'mixed', its accumulators' row @r@ scaled by @2^r@ at the end.
scaledRows :: Kernel T32
scaledRows = kernel "scaled_rows" $ do
  k <- dim "k" (DimRange 8 256 8)
  a <- buffer @ReadOnly @16 @Float "a" (16 * k)
  b <- buffer @ReadOnly @16 @BF16 "b" (16 * k)
  c <- buffer @WriteOnly @16 @Float "c" 256
  threads @128
  groups 1
  pure $ do
    scales <- tgArray @8 @Float "scales"
    pure $ do
      forM_ [0 .. 7 :: Int] $ \r -> whenLane (threadIndex .== 0) (tgStore scales (fromIntegral r) (fromIntegral (2 ^ r :: Int)))
      barrier
      row0 <- let_ "row0" ((simdGroupIndex `quotE` 2) * 8)
      col0 <- let_ "col0" ((simdGroupIndex `remE` 2) * 8)
      acc <- fragment @Float "acc"
      fa <- fragment @Float "fa"
      fbh <- fragment @BF16 "fbh"
      fb <- fragment @Float "fb"
      forU "p" 0 (k `quotE` 8) $ \p -> do
        fragLoad fa a (row0 * k + p * 8) k
        fragLoadT fbh b (col0 * k + p * 8) k
        fragConvert fb fbh
        fragMma acc fa fb acc
      fragScaleRows acc scales 0
      fragStore c (row0 * 16 + col0) 16 acc

{- | A kernel of an f32 A by a bf16 B against f64, its row @i@ of C scaled
by @scale ^ (i mod 8)@.
-}
mixedAgrees :: Kernel T32 -> Double -> IO ()
mixedAgrees kernel' scale = do
  let k = 64 :: Int
      noise :: Int -> Int -> Float
      noise s i = fromIntegral ((s * 7919 + i * 104729) `mod` 2001 - 1000) / 1000
      as = [noise 1 i | i <- [0 .. 16 * k - 1]]
      bs = [widenBF16 (narrowBF16 (noise 2 i)) | i <- [0 .. 16 * k - 1]]
      exact i j = scale ^ (i `mod` 8) * sum [float2Double (as !! (i * k + p)) * float2Double (bs !! (j * k + p)) | p <- [0 .. k - 1]]
  case runOn kernel' [("k", VU32 (fromIntegral k :: Word32))] [("a", map VF32 as), ("b", map (VBF16 . narrowBF16) bs), ("c", replicate 256 (VF32 0))] 0 of
    Left e -> assertFailure (show e)
    Right out -> do
      let cs = Map.findWithDefault [] "c" out
          diffs = [abs (output v - exact i j) / scale ^ (i `mod` 8) | (n, v) <- zip [0 :: Int ..] cs, let (i, j) = n `divMod` 16]
      assertBool ("the differences are " <> describe diffs) (length cs == 256 && within 1e-4 diffs)

-- | A kernel on T32 against f64, inputs made of @value@ and read back with @widenV@.
agrees :: Kernel T32 -> (Float -> Value) -> (Value -> Float) -> IO ()
agrees kernel' value widenV = do
  let k = 64 :: Int
      noise :: Int -> Int -> Float
      noise s i = fromIntegral ((s * 7919 + i * 104729) `mod` 2001 - 1000) / 1000
      as = [value (noise 1 i) | i <- [0 .. 16 * k - 1]]
      bs = [value (noise 2 i) | i <- [0 .. 16 * k - 1]]
      exact i j = sum [float2Double (widenV (as !! (i * k + p))) * float2Double (widenV (bs !! (j * k + p))) | p <- [0 .. k - 1]]
  case runOn kernel' [("k", VU32 (fromIntegral k :: Word32))] [("a", as), ("b", bs), ("c", replicate 256 (VF32 0))] 0 of
    Left e -> assertFailure (show e)
    Right out -> do
      let cs = Map.findWithDefault [] "c" out
          diffs = [abs (output v - exact i j) | (n, v) <- zip [0 ..] cs, let (i, j) = n `divMod` 16]
      assertBool ("the differences are " <> describe diffs) (length cs == 256 && within 1e-4 diffs)

-- | An f32 output, as a Double; anything else is a NaN, which fails.
output :: Value -> Double
output = \case
  VF32 x -> float2Double x
  _ -> 0 / 0

-- | Every difference a number below the tolerance: a NaN fails.
within :: Double -> [Double] -> Bool
within tol = all (\d -> not (isNaN d) && d < tol)

describe :: [Double] -> String
describe ds = "at worst " <> show (maximum (0 : filter (not . isNaN) ds)) <> ", " <> show (length (filter isNaN ds)) <> " of them NaN"
