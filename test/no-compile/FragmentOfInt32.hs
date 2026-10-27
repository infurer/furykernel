-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: Fragments multiply bf16, f16 and f32: no i32.
{-# LANGUAGE GHC2024 #-}

module FragmentOfInt32 (oops) where

import Data.Int (Int32)
import FuryKernel.Build

oops :: Frag Float -> Frag Int32 -> Frag Int32 -> K Uniform ()
oops acc a b = fragMma acc a b acc
