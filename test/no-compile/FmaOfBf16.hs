-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: bf16 is a storage type, with no arithmetic: widen it to Float,
{-# LANGUAGE GHC2024 #-}

module FmaOfBf16 (oops) where

import FuryKernel.Build (Exp, fmaE)
import FuryKernel.Type (BF16)

oops :: Exp BF16 -> Exp BF16 -> Exp BF16 -> Exp BF16
oops = fmaE
