-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: A barrier needs every thread of the threadgroup, and this code runs under a per-lane branch
{-# LANGUAGE GHC2024 #-}

module BarrierUnderLaneBranch (oops) where

import FuryKernel.Build

oops :: K Uniform ()
oops = whenLane (threadIndex .== 0) barrier
