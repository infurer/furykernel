-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | FuryKernel: a typed eDSL for GPU compute kernels.

This is the target-agnostic core. The language, its checks and the
interface that target libraries implement are added module by module.
-}
module FuryKernel () where
