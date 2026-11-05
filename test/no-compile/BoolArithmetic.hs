-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: Bool has no arithmetic: use .&&, .|| and notE.
{-# LANGUAGE GHC2024 #-}

module BoolArithmetic (oops) where

import FuryKernel.Build

oops :: Exp Bool -> Exp Bool -> Exp Bool
oops a b = a + b
