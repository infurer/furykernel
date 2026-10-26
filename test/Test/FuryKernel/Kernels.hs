-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

-- | Kernels the tests share.
module Test.FuryKernel.Kernels (
  scale,
  scalePacked,
  rmsnorm,
) where

import Data.Word (Word32)
import FuryKernel.Build
import FuryKernel.Constraint (PackOk, TgArrayOk, ThreadsOk)
import FuryKernel.IR (DimRange (..))
import FuryKernel.Reduce (tgReduceSum)
import FuryKernel.Target (Target)
import FuryKernel.Type (BF16)

-- | y = x * s, element by element, in bf16 with f32 arithmetic.
scale :: forall t. (Target t, ThreadsOk t 64) => Kernel t
scale = kernel "scale" $ do
  n <- dim "n" (DimRange 1 65536 1)
  s <- param @Float "s"
  x <- buffer @ReadOnly @16 @BF16 "x" n
  y <- buffer @WriteOnly @16 @BF16 "y" n
  threads @64
  groups 1
  pure . pure $
    strided "i" n $ \i -> do
      v <- let_ "v" (widen @Float (load x i))
      store y i (narrow @BF16 (v * s))

-- | 'scale' with each lane reading and writing eight elements at once.
scalePacked :: forall t. (Target t, ThreadsOk t 64, PackOk t 8 BF16 16) => Kernel t
scalePacked = kernel "scale_packed" $ do
  packs <- dim "packs" (DimRange 1 8192 1)
  s <- param @Float "s"
  x <- buffer @ReadOnly @16 @BF16 "x" (packs * 8)
  y <- buffer @WriteOnly @16 @BF16 "y" (packs * 8)
  threads @64
  groups 1
  pure . pure $
    strided "j" packs $ \j -> do
      v <- loadPack @8 "v" x j
      storePack @8 y j [narrow @BF16 (widen @Float e * s) | e <- unpack v]

{- | RMSNorm over rows of @hidden@ elements, one threadgroup per row:
y = x / sqrt (mean x² + eps) * w, in bf16 with f32 arithmetic.
-}
rmsnorm :: forall t. (Target t, ThreadsOk t 256, TgArrayOk t 32 Float) => Word32 -> Kernel t
rmsnorm hidden = kernel "rmsnorm" $ do
  rows <- dim "rows" (DimRange 1 4096 1)
  eps <- param @Float "eps"
  x <- buffer @ReadOnly @16 @BF16 "x" (rows * h)
  w <- buffer @ReadOnly @16 @BF16 "w" h
  y <- buffer @WriteOnly @16 @BF16 "y" (rows * h)
  threads @256
  groups rows
  pure $ do
    scratch <- tgArray @32 @Float "scratch"
    pure $ do
      base <- let_ "base" (groupIndex * h)
      acc <- var "acc" 0
      strided "i" h $ \i -> do
        v <- let_ "v" (widen @Float (load x (base + i)))
        acc .= get acc + v * v
      total <- tgReduceSum "sum_sq" scratch (get acc)
      scale' <- let_ "scale" (rsqrtE (total / toF32 h + eps))
      strided "j" h $ \j ->
        store y (base + j) (narrow @BF16 (widen @Float (load x (base + j)) * scale' * widen @Float (load w j)))
 where
  h = fromIntegral hidden
