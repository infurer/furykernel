-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The tile library: a GEMM of tiles on the interpreter against its
meaning in f64, partial tiles included, and a row's bits whatever the
batch.
-}
module Test.FuryKernel.TileTest (tests) where

import Data.Map.Strict qualified as Map
import Data.Word (Word32)
import FuryKernel
import FuryKernel.Interp (Value (..))
import FuryKernel.Numerics (narrowBF16, widenBF16)
import GHC.Float (float2Double)
import Test.FuryKernel.Run (runOn)
import Test.FuryKernel.Targets (T32)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (assertBool, assertFailure, testCase, (@?=))

tests :: TestTree
tests =
  testGroup
    "Tile"
    [ testCase "a GEMM of 64 × 64 tiles, the last ones partial, is its f64 meaning to f32's accumulation" $ do
        cs <- run 70 72 48
        let exact i j = sum [float2Double (widen' (as 48 !! (i * 48 + p))) * float2Double (widen' (bs 48 !! (j * 48 + p))) | p <- [0 .. 47]]
            diffs = [abs (float2Double (f v) - exact i j) | (e, v) <- zip [0 ..] cs, let (i, j) = e `divMod` 72]
        length cs @?= 70 * 72
        assertBool ("the differences are " <> describe diffs) (within 1e-4 diffs)
    , testCase "a row's bits are the same whatever the rows with it" $ do
        many <- run 70 72 48
        few <- run 5 72 48
        map bits few @?= map bits (take (5 * 72) many)
    ]
 where
  widen' = \case
    VBF16 x -> widenBF16 x
    _ -> 0 / 0
  f = \case
    VF32 x -> x
    _ -> 0 / 0
  bits = \case
    VF32 x -> Just x
    _ -> Nothing
  noise :: Int -> Int -> Float
  noise s i = fromIntegral ((s * 7919 + i * 104729) `mod` 2001 - 1000) / 1000
  as k = [VBF16 (narrowBF16 (noise 1 i)) | i <- [0 .. 70 * k - 1]]
  bs k = [VBF16 (narrowBF16 (noise 2 i)) | i <- [0 .. 72 * k - 1]]
  run :: Int -> Int -> Int -> IO [Value]
  run m n k =
    case runOn tiled [("m", u m), ("n", u n), ("k", u k)] [("a", take (m * k) (as k)), ("b", take (n * k) (bs k)), ("c", replicate (m * n) (VF32 0))] 0 of
      Left e -> assertFailure (show e) >> pure []
      Right out -> pure (Map.findWithDefault [] "c" out)
  u = VU32 . (fromIntegral :: Int -> Word32)

{- | C = A · Bᵀ in tiles of 64 × 64, A @m × k@ and B @n × k@ of bf16, C
@m × n@ of f32; rows and columns past the edges read the last ones' and
store nothing.
-}
tiled :: Kernel T32
tiled = kernel "tiled" $ do
  m <- dim "m" (DimRange 1 256 1)
  n <- dim "n" (DimRange 8 256 8)
  k <- dim "k" (DimRange 16 256 16)
  a <- buffer @ReadOnly @16 @BF16 "a" (m * k)
  b <- buffer @ReadOnly @16 @BF16 "b" (n * k)
  c <- buffer @WriteOnly @16 @Float "c" (m * n)
  threads @256
  groups (((m + 63) `quotE` 64) * ((n + 63) `quotE` 64))
  pure $ do
    aSlice <- tgArray @1024 @Float "a_slice"
    bSlice <- tgArray @1024 @Float "b_slice"
    cTile <- tgArray @4096 @Float "c_tile"
    pure $ do
      tilesN <- let_ "tiles_n" ((n + 63) `quotE` 64)
      row0 <- let_ "row0" ((groupIndex `quotE` tilesN) * 64)
      col0 <- let_ "col0" ((groupIndex `remE` tilesN) * 64)
      gemmTile
        "t"
        (GemmTile 64 64 16 4)
        aSlice
        bSlice
        cTile
        k
        (\r p -> pure (widen @Float (load a (minE (row0 + r) (m - 1) * k + p))))
        (\col p -> pure (widen @Float (load b (minE (col0 + col) (n - 1) * k + p))))
        (\r col v -> whenLane (row0 + r .< m .&& col0 + col .< n) (store c ((row0 + r) * n + col0 + col) v))

-- | Every difference a number below the tolerance: a NaN fails.
within :: Double -> [Double] -> Bool
within tol = all (\d -> not (isNaN d) && d < tol)

describe :: [Double] -> String
describe ds = "at worst " <> show (maximum (0 : filter (not . isNaN) ds)) <> ", " <> show (length (filter isNaN ds)) <> " of them NaN"
