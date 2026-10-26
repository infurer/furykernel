-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: SIMD groups reduce f32, f16, i32 and u32: widen a storage type first.
{-# LANGUAGE GHC2024 #-}
{-# LANGUAGE OverloadedStrings #-}

module SimdSumOfInt8 (oops) where

import Data.Int (Int8)
import FuryKernel.Build

oops :: Exp Int8 -> K Uniform (Exp Int8)
oops = simdSum "total"
