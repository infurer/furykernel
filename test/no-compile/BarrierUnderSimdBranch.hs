-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: A barrier needs every thread of the threadgroup, and this code runs under a branch that is only uniform per SIMD group (whenSimd).
{-# LANGUAGE GHC2024 #-}

module BarrierUnderSimdBranch (oops) where

import FuryKernel.Build

oops :: K Uniform ()
oops = whenSimd (simdGroupIndex .== 0) barrier
