-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE AllowAmbiguousTypes #-}

{- | A bf16 GEMV family, as decode runs it: y = W x for @m@ rows of x. W is
@[n, k]@ (out_in), x @[m, k]@, y @[m, n]@, all bf16; products accumulate
in f32.

The shape (@n@, @k@, @m@) is baked into each kernel, and so is the
variant: output rows per SIMD group, threads per threadgroup, the pack
width of a lane's loads and the split-K factor.

The per-row reduction order is fixed by the pack width and the split
alone, never by the rows, the threads or @m@:

* Lane @l@ of split @q@ is virtual lane @q * w + l@ (@w@ the SIMD width);
  the virtual lanes take the row's packs in turn, so lane @v@ visits packs
  @v@, @v + s w@, and so on.
* Each pack's elements go in groups of four into one chain each, as
  @((x0 w0 + x1 w1) + (x2 w2 + x3 w3))@; a lane's chains are added in
  order, then summed over the SIMD group.
* With splits, each SIMD group's sum is a partial, and the partials are
  added in split order ('combineSplits').

So the variants for every @m@ give a row the same bits, and so do the
variants that differ only in rows and threads.
-}
module Test.FuryKernel.Gemv (
  GemvShape (..),
  GemvVariant (..),
  gemv,
) where

import Control.Monad (forM, forM_, unless)
import Data.Proxy (Proxy (..))
import Data.Text qualified as T
import Data.Word (Word32)
import FuryKernel.Build
import FuryKernel.Constraint (PackOk, TgArrayOk, ThreadsOk, withPack, withTgArray, withThreads)
import FuryKernel.Finding (Finding (..), Rule (..))
import FuryKernel.IR (Name)
import FuryKernel.Reduce (combineSplits)
import FuryKernel.Target (Target, TargetCaps (..), targetCaps)
import FuryKernel.Type (BF16)
import GHC.TypeNats (KnownNat)
import Numeric.Natural (Natural)

-- | What a kernel bakes in: W is @outputs × inputs@, x @batch × inputs@.
data GemvShape = GemvShape
  { outputs :: Natural
  , inputs :: Natural
  , batch :: Natural
  }
  deriving stock (Eq, Show)

-- | How a kernel computes.
data GemvVariant = GemvVariant
  { rowsPerSimd :: Natural
  -- ^ Output rows each SIMD group computes.
  , threadsPerGroup :: Natural
  , packWidth :: Natural
  -- ^ Elements a lane loads at once: 4, 8, ...
  , splitK :: Natural
  -- ^ SIMD groups that share a row, each with its part of the packs.
  }
  deriving stock (Eq, Show)

-- Static-mode evidence for numbers known at run time.
data SomeThreads t where
  SomeThreads :: forall t n. (KnownNat n, ThreadsOk t n) => Proxy n -> SomeThreads t

data SomePack t where
  SomePack :: forall t n. (KnownNat n, PackOk t n BF16 16) => Proxy n -> SomePack t

data SomeArray t where
  SomeArray :: forall t n. (KnownNat n, TgArrayOk t n Float) => Proxy n -> SomeArray t

data SomeScratch where
  SomeScratch :: (KnownNat n) => TgArr n Float -> SomeScratch

{- | The kernel @name@ for a shape and a variant, or why the variant is
illegal on @t@. The shape must divide: @n@ by the rows of a threadgroup,
@k@ by the pack width.
-}
gemv :: forall t. (Target t) => Name -> GemvShape -> GemvVariant -> Either [Finding] (Kernel t)
gemv name shape v = do
  unless (null problems) (Left problems)
  SomeThreads (_ :: Proxy threadsN) <- withThreads @t v.threadsPerGroup (SomeThreads @t)
  SomePack (_ :: Proxy p) <- withPack @t @BF16 @16 v.packWidth (SomePack @t)
  rowPack <- if r == 1 then pure Nothing else Just <$> withPack @t @BF16 @16 r (SomePack @t)
  scratch <- if s == 1 then pure Nothing else Just <$> withTgArray @t @Float (s * perGroup) (SomeArray @t)
  pure . kernel name $ do
    x <- buffer @ReadOnly @16 @BF16 "x" (lit (m * k))
    w <- buffer @ReadOnly @16 @BF16 "w" (lit (n * k))
    y <- buffer @WriteOnly @16 @BF16 "y" (lit (m * n))
    threads @threadsN
    groups (lit (n `div` rowsPerGroup))
    pure $ do
      partials <- traverse (\(SomeArray (_ :: Proxy slots)) -> SomeScratch <$> tgArray @slots @Float "partials") scratch
      pure $ do
        -- Which rows and which part of them this SIMD group takes.
        (split, rowGroup, lane) <-
          if s == 1
            then pure (0, simdGroupIndex, simdLane)
            else do
              split <- let_ "split" (simdGroupIndex `remE` lit s)
              rowGroup <- let_ "row_group" (simdGroupIndex `quotE` lit s)
              lane <- let_ "lane" (split * lit width + simdLane)
              pure (split, rowGroup, lane)
        row0 <- let_ "row0" (groupIndex * lit rowsPerGroup + rowGroup * lit r)
        accs <- forM rows $ \q -> forM xrows $ \i -> forM chains $ \c ->
          var ("acc" <> showT q <> "_" <> showT i <> "_" <> showT c) 0
        forStep "j" lane (lit packs) (lit (s * width)) $ \j -> do
          xs <- forM xrows $ \i -> do
            xp <- loadPack @p ("x" <> showT i) x (if i == 0 then j else lit (i * packs) + j)
            forM (zip [0 :: Int ..] (unpack xp)) $ \(e, xe) ->
              let_ ("xf" <> showT i <> "_" <> showT e) (widen @Float xe)
          forM_ rows $ \q -> do
            wp <- loadPack @p ("w" <> showT q) w ((row0 + lit q) * lit packs + j)
            ws <- forM (zip [0 :: Int ..] (unpack wp)) $ \(e, we) ->
              let_ ("wf" <> showT q <> "_" <> showT e) (widen @Float we)
            forM_ (zip xs (at q accs)) $ \(xv, accsI) ->
              forM_ (zip chains accsI) $ \(c, acc) -> do
                let prod o = at (4 * c + o) xv * at (4 * c + o) ws
                acc .= get acc + ((prod 0 + prod 1) + (prod 2 + prod 3))
        totals <- forM (zip rows accs) $ \(q, accsQ) -> forM (zip xrows accsQ) $ \(i, accsI) ->
          simdSum ("total" <> showT q <> "_" <> showT i) (inOrder (map get accsI))
        case partials of
          Nothing ->
            whenLane (simdLane .== 0) $
              forM_ xrows $ \i -> do
                let values = [narrow @BF16 (at i (at q totals)) | q <- rows]
                case rowPack of
                  Nothing -> store y (lit (i * n) + row0) (at 0 values)
                  Just (SomePack (_ :: Proxy rp)) ->
                    storePack @rp y (lit (i * (n `div` r)) + groupIndex * lit rowGroups + rowGroup) values
          Just (SomeScratch arr) -> do
            whenLane (simdLane .== 0) $
              forM_ rows $ \q -> forM_ xrows $ \i ->
                tgStore arr (((rowGroup * lit r + lit q) * lit m + lit i) * lit s + split) (at i (at q totals))
            combineSplits "total" arr s perGroup $ \o total ->
              store
                y
                ((o `remE` lit m) * lit n + groupIndex * lit rowsPerGroup + o `quotE` lit m)
                (narrow @BF16 total)
 where
  n = shape.outputs
  k = shape.inputs
  m = shape.batch
  r = v.rowsPerSimd
  s = v.splitK
  width = (targetCaps @t).capSimdWidth
  simds = v.threadsPerGroup `div` width
  rowGroups = simds `div` s
  rowsPerGroup = rowGroups * r
  perGroup = rowsPerGroup * m
  packs = k `div` v.packWidth
  rows = [0 .. r - 1]
  xrows = [0 .. m - 1]
  chains = [0 .. v.packWidth `div` 4 - 1]
  problems =
    [ Finding Threads ("The split, " <> showT s <> ", does not divide the " <> showT simds <> " SIMD groups of a threadgroup.")
    | s == 0 || simds `mod` s /= 0 || rowGroups == 0
    ]
      <> [Finding Packs ("The pack width, " <> showT v.packWidth <> ", is not a multiple of 4.") | v.packWidth `mod` 4 /= 0]

-- | A sum from left to right, with nothing added before the first term.
inOrder :: [Exp Float] -> Exp Float
inOrder = \case
  [] -> 0
  e : es -> foldl' (+) e es

lit :: Natural -> Exp Word32
lit = fromIntegral

at :: Natural -> [a] -> a
at i xs = xs !! fromIntegral i

showT :: (Show a) => a -> T.Text
showT = T.pack . show
