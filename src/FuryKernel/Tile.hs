-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE AllowAmbiguousTypes #-}

{- | Tiles: a threadgroup's tile of a matrix product, its operands staged in
threadgroup memory and multiplied as fragments.

'gemmTile' computes @C = A · Bᵀ@ for a tile of C's rows and columns. The
caller makes A's and B's elements in f32, from whatever they are computed
from. The product goes over k in steps: each step's slices of A and B are
staged in threadgroup memory by every thread, then each SIMD group
multiplies its part of the tile as 8×8 f32 fragments. At the end the tile
is stored in threadgroup memory and handed back element by element, so
that how a target spreads a fragment over its lanes stays its own.

An element's sum goes over the steps in k order, and within a step its
fragments' products are added as the target adds them. That order depends
on the tile's shape and the step only, never on the other rows or columns,
so a row's result doesn't depend on the batch it is computed in.
-}
module FuryKernel.Tile (
  GemmTile (..),
  gemmTile,
) where

import Control.Monad (forM, forM_)
import Data.Proxy (Proxy (..))
import Data.Text qualified as T
import Data.Word (Word32)
import FuryKernel.Build
import FuryKernel.Finding (Rule (..))
import FuryKernel.IR (Name)
import FuryKernel.IR qualified as IR
import GHC.TypeNats (KnownNat, natVal)
import Numeric.Natural (Natural)

-- | A tile's shape, and how its SIMD groups share it.
data GemmTile = GemmTile
  { gtRows :: Natural
  -- ^ The tile's rows of C.
  , gtCols :: Natural
  -- ^ Its columns.
  , gtStep :: Natural
  -- ^ k per step, a multiple of 8.
  , gtGroupRows :: Natural
  {- ^ SIMD groups along the rows; the threadgroup's others go along the
  columns, each group taking an equal block.
  -}
  }
  deriving stock (Eq, Show)

{- | One tile of @C = A · Bᵀ@, its locals named from @name@.

@a@ and @b@ hold a step's slices, row by row: @gtRows × gtStep@ and
@gtCols × gtStep@ elements. @c@ holds the tile, @gtRows × gtCols@. @k@ is
the length of the products, known to be a multiple of the step: a literal
one, or a dimension declared a multiple of it ('knownMultiple'). A tile
that breaks one of these is a finding, and nothing of it is built.
@aAt row k@ and
@bAt col k@ make the elements, by row and column within the tile;
@element row col value@ takes each of the tile's results, every one once,
by one thread or another.

Barriers separate the stages, but none comes before the first: a caller
that stores what the element functions read to threadgroup memory puts a
barrier between, and one that reuses @c@ afterwards needs a barrier of its
own.
-}
gemmTile ::
  forall na nb nc.
  (KnownNat na, KnownNat nb, KnownNat nc) =>
  Name ->
  GemmTile ->
  TgArr na Float ->
  TgArr nb Float ->
  TgArr nc Float ->
  Exp Word32 ->
  (forall r. Exp Word32 -> Exp Word32 -> K r (Exp Float)) ->
  (forall r. Exp Word32 -> Exp Word32 -> K r (Exp Float)) ->
  (Exp Word32 -> Exp Word32 -> Exp Float -> K Divergent ()) ->
  K Uniform ()
gemmTile name t a b c k aAt bAt element = do
  groupsN <- simdGroups
  kMultiple <- knownMultiple k
  let groupCols = groupsN `div` max 1 t.gtGroupRows
      subRows = t.gtRows `div` max 1 t.gtGroupRows
      subCols = t.gtCols `div` max 1 groupCols
      fragRows = subRows `div` 8
      fragCols = subCols `div` 8
      sizes = (natVal (Proxy @na), natVal (Proxy @nb), natVal (Proxy @nc))
  -- The first broken precondition, if any: the later ones take the
  -- earlier as given.
  let broken =
        ["its step, " <> showT t.gtStep <> ", is not a positive multiple of 8" | not (t.gtStep > 0 && t.gtStep `mod` 8 == 0)]
          <> [showT groupsN <> " SIMD groups don't share " <> showT t.gtGroupRows <> " rows of groups" | not (t.gtGroupRows > 0 && groupsN `mod` t.gtGroupRows == 0)]
          <> ["its " <> showT t.gtRows <> " × " <> showT t.gtCols <> " don't divide into the SIMD groups' blocks of 8×8 fragments" | not (subRows > 0 && subCols > 0 && subRows * t.gtGroupRows == t.gtRows && subCols * groupCols == t.gtCols && subRows `mod` 8 == 0 && subCols `mod` 8 == 0)]
          <> ["its arrays hold " <> showT sizes <> " elements" | sizes /= (t.gtRows * t.gtStep, t.gtCols * t.gtStep, t.gtRows * t.gtCols)]
          <> ["k, " <> kWhat <> ", is not known to be a multiple of its step, " <> showT t.gtStep | maybe True (\m -> m `mod` t.gtStep /= 0) kMultiple]
      kWhat = case (expIR k, kMultiple) of
        (IR.EParam d, Just m) -> "the dimension " <> d <> " declared a multiple of " <> showT m
        (_, Just m) -> showT m
        (e, Nothing) -> IR.renderExpr e <> ", neither a literal nor a dimension"
  case broken of
    p : _ -> problem p
    [] -> do
      let step = lit t.gtStep
          cols = lit t.gtCols
      groupRow <- let_ (name <> "_group_row") (simdGroupIndex `quotE` lit groupCols)
      groupCol <- let_ (name <> "_group_col") (simdGroupIndex `remE` lit groupCols)
      accs <- forM [(i, j) | i <- [0 .. fragRows - 1], j <- [0 .. fragCols - 1]] $ \(i, j) ->
        ((i, j),) <$> fragment @Float (name <> "_acc" <> showT i <> "_" <> showT j)
      as <- forM [0 .. fragRows - 1] $ \i -> fragment @Float (name <> "_a" <> showT i)
      bs <- forM [0 .. fragCols - 1] $ \j -> fragment @Float (name <> "_b" <> showT j)
      forU (name <> "_step") 0 (k `quotE` step) $ \s -> do
        strided (name <> "_stage_a") (lit (t.gtRows * t.gtStep)) $ \e -> do
          v <- aAt (e `quotE` step) (s * step + e `remE` step)
          tgStore a e v
        strided (name <> "_stage_b") (lit (t.gtCols * t.gtStep)) $ \e -> do
          v <- bAt (e `quotE` step) (s * step + e `remE` step)
          tgStore b e v
        barrier
        forM_ [0, 8 .. t.gtStep - 8] $ \kk -> do
          forM_ (zip [0 ..] as) $ \(i, fa) ->
            fragLoadTg fa a ((groupRow * lit subRows + lit (8 * i)) * step + lit kk) step
          forM_ (zip [0 ..] bs) $ \(j, fb) ->
            fragLoadTgT fb b ((groupCol * lit subCols + lit (8 * j)) * step + lit kk) step
          forM_ accs $ \((i, j), acc) ->
            fragMma acc (as !! fromIntegral i) (bs !! fromIntegral j) acc
        barrier
      forM_ accs $ \((i, j), acc) ->
        fragStoreTg c ((groupRow * lit subRows + lit (8 * i)) * cols + groupCol * lit subCols + lit (8 * j)) cols acc
      barrier
      strided (name <> "_element") (lit (t.gtRows * t.gtCols)) $ \e ->
        element (e `quotE` cols) (e `remE` cols) (tgLoad c e)
 where
  problem m = note ThreadgroupMemory ("The tile " <> name <> ": " <> m <> ".")
  lit :: Natural -> Exp Word32
  lit = fromIntegral
  showT :: (Show s) => s -> T.Text
  showT = T.pack . show
