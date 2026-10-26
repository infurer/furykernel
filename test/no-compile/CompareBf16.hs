-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: bf16 is a storage type, with no comparisons: widen it to Float first.
{-# LANGUAGE GHC2024 #-}

module CompareBf16 (oops) where

import FuryKernel.Build
import FuryKernel.Type (BF16)

oops :: Exp BF16 -> Exp BF16 -> Exp Bool
oops a b = a .< b
