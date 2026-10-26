-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

-- | Kernels the tests share.
module Test.FuryKernel.Kernels (
  scale,
) where

import FuryKernel.Build
import FuryKernel.Constraint (ThreadsOk)
import FuryKernel.IR (DimRange (..))
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
