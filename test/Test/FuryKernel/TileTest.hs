-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE AllowAmbiguousTypes #-}

{- | The tile library: a GEMM of tiles on the interpreter against its
meaning in f64, partial tiles included, and a row's bits whatever the
batch, and whatever the tile's shape and step.
-}
module Test.FuryKernel.TileTest (tests) where

import Data.Either (fromLeft)
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe)
import Data.Proxy (Proxy (..))
import Data.Text qualified as T
import Data.Word (Word32)
import FuryKernel
import FuryKernel.Finding (Finding (..))
import FuryKernel.Interp (Value (..))
import FuryKernel.Numerics (narrowBF16, widenBF16)
import FuryKernel.Variant (Axis (..), Outcome (..), Profile (..), Shape (..), Space (..), enumerate, pointValue)
import GHC.Float (float2Double)
import GHC.TypeNats (KnownNat)
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
    , testCase "tiles of another shape and step, their results handed back a fragment a round, give the same bits" $ do
        wide <- runWith tiled 70 72 64
        thin <- runWith tiledNarrow 70 72 64
        length thin @?= 70 * 72
        map bits thin @?= map bits wide
    , testCase "a tile's broken precondition is a finding, and nothing of it is built" $ do
        refused (tiledAs @128 @1024 @4096 (GemmTile 8 64 16 4)) @?= ["The tile t: its 8 × 64 don't divide into the SIMD groups' blocks of 8×8 fragments."]
        refused (tiledAs @1 @1 @4096 (GemmTile 64 64 0 4)) @?= ["The tile t: its step, 0, is not a positive multiple of 8."]
        refused (tiledAs @768 @768 @4096 (GemmTile 64 64 12 4)) @?= ["The tile t: its step, 12, is not a positive multiple of 8."]
        refused (tiledAs @1024 @1024 @256 (GemmTile 64 64 16 4)) @?= ["The tile t: its arrays hold (1024,1024,256) elements."]
        refused (tiledAs @1024 @1024 @4096 (GemmTile 64 64 16 3)) @?= ["The tile t: 8 SIMD groups don't share 3 rows of groups."]
        refused (tiledAs @512 @1024 @4096 (GemmTile 64 64 16 4)) @?= ["The tile t: its arrays hold (512,1024,4096) elements."]
    , testCase "k known to be a multiple of the step, or the tile is refused" $ do
        refused (tiledK @1024 @2048 @512 (KDim (DimRange 16 256 16)) (GemmTile 32 64 32 1))
          @?= ["The tile t: k, the dimension k declared a multiple of 16, is not known to be a multiple of its step, 32."]
        refused (tiledK @1024 @2048 @512 (KLit 48) (GemmTile 32 64 32 1))
          @?= ["The tile t: k, 48, is not known to be a multiple of its step, 32."]
        refused (tiledK @1024 @2048 @512 (KLit 64) (GemmTile 32 64 32 1)) @?= []
    , testCase "a variant space prunes the tile whose SIMD groups' rows are too few, as illegal" $
        [ (pointValue "g" p, map (.findingMessage) fs)
        | (p, outcome) <- enumerate (Profile 1) tileSpace (Shape [("m", 256), ("n", 256), ("k", 256)])
        , let fs = case outcome of
                Illegal found -> found
                _ -> []
        ]
          @?= [(Just 4, []), (Just 8, ["The tile t: its 32 × 64 don't divide into the SIMD groups' blocks of 8×8 fragments."])]
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
  run = runWith tiled
  runWith :: Kernel T32 -> Int -> Int -> Int -> IO [Value]
  runWith kernel' m n k =
    case runOn kernel' [("m", u m), ("n", u n), ("k", u k)] [("a", take (m * k) (as k)), ("b", take (n * k) (bs k)), ("c", replicate (m * n) (VF32 0))] 0 of
      Left e -> assertFailure (show e) >> pure []
      Right out -> pure (Map.findWithDefault [] "c" out)
  u = VU32 . (fromIntegral :: Int -> Word32)

{- | C = A · Bᵀ in tiles of 64 × 64, A @m × k@ and B @n × k@ of bf16, C
@m × n@ of f32; rows and columns past the edges read the last ones' and
store nothing.
-}
tiled :: Kernel T32
tiled = tiledAs @1024 @1024 @4096 (GemmTile 64 64 16 4)

-- | 'tiled' in tiles of 32 × 64 in steps of 32, a fragment a round.
tiledNarrow :: Kernel T32
tiledNarrow = tiledAs @1024 @2048 @512 (GemmTile 32 64 32 1)

-- | A tile of the given shape, k a dimension of multiples of its step (16 at least).
tiledAs :: forall na nb nc. (KnownNat na, KnownNat nb, KnownNat nc, TgArrayOk T32 na Float, TgArrayOk T32 nb Float, TgArrayOk T32 nc Float) => GemmTile -> Kernel T32
tiledAs t = tiledK @na @nb @nc (KDim (DimRange (max 16 t.gtStep) 256 (max 16 t.gtStep))) t

-- | Where a test tile's k comes from: a dimension of a range, or a literal.
data KFrom = KDim DimRange | KLit Word32

tiledK :: forall na nb nc. (KnownNat na, KnownNat nb, KnownNat nc, TgArrayOk T32 na Float, TgArrayOk T32 nb Float, TgArrayOk T32 nc Float) => KFrom -> GemmTile -> Kernel T32
tiledK kFrom t = kernel "tiled" $ do
  m <- dim "m" (DimRange 1 256 1)
  n <- dim "n" (DimRange 8 256 8)
  k <- case kFrom of
    KDim r -> dim "k" r
    KLit v -> pure (fromIntegral v)
  a <- buffer @ReadOnly @16 @BF16 "a" (m * k)
  b <- buffer @ReadOnly @16 @BF16 "b" (n * k)
  c <- buffer @WriteOnly @16 @Float "c" (m * n)
  threads @256
  let rows = fromIntegral t.gtRows
      cols = fromIntegral t.gtCols
  groups (((m + rows - 1) `quotE` rows) * ((n + cols - 1) `quotE` cols))
  pure $ do
    aSlice <- tgArray @na @Float "a_slice"
    bSlice <- tgArray @nb @Float "b_slice"
    cTile <- tgArray @nc @Float "c_tile"
    pure $ do
      tilesN <- let_ "tiles_n" ((n + cols - 1) `quotE` cols)
      row0 <- let_ "row0" ((groupIndex `quotE` tilesN) * rows)
      col0 <- let_ "col0" ((groupIndex `remE` tilesN) * cols)
      gemmTile
        "t"
        t
        aSlice
        bSlice
        cTile
        k
        (\r p -> pure [widen @Float (load a (minE (row0 + r) (m - 1) * k + p + fromIntegral x)) | x <- [0 .. 3 :: Int]])
        (\col p -> pure [widen @Float (load b (minE (col0 + col) (n - 1) * k + p + fromIntegral x)) | x <- [0 .. 7 :: Int]])
        (\r col v -> whenLane (row0 + r .< m .&& col0 + col .< n) (store c ((row0 + r) * n + col0 + col) v))

-- | Every difference a number below the tolerance: a NaN fails.
within :: Double -> [Double] -> Bool
within tol = all (\d -> not (isNaN d) && d < tol)

describe :: [Double] -> String
describe ds = "at worst " <> show (maximum (0 : filter (not . isNaN) ds)) <> ", " <> show (length (filter isNaN ds)) <> " of them NaN"

-- | The messages of the findings that refuse a kernel.
refused :: Kernel T32 -> [T.Text]
refused k = map (.findingMessage) (fromLeft [] (check k))

-- | Tiles of 32 rows with 4 or 8 of the 8 SIMD groups to the rows: 8 leave each 4 rows.
tileSpace :: Space T32
tileSpace =
  Space
    { spaceFamily = "tiles"
    , spaceAxes = [Axis "rows" "r" [32] False, Axis "group_rows" "g" [4, 8] False]
    , spaceShapes = [Shape [("m", 256), ("n", 256), ("k", 256)]]
    , spaceBatch = ["m"]
    , spaceDivides = \_ _ -> []
    , spaceKernel = \_ _ p -> do
        let rows = fromMaybe 0 (pointValue "r" p)
        SomeArray (_ :: Proxy na) <- withTgArray @T32 @Float (rows * 16) SomeArray
        pure (tiledAs @na @1024 @4096 (GemmTile rows 64 16 (fromMaybe 0 (pointValue "g" p))))
    }

data SomeArray where
  SomeArray :: forall n. (KnownNat n, TgArrayOk T32 n Float) => Proxy n -> SomeArray
