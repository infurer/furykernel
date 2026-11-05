-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: i32 is an integer type, with no floating-point operations.
{-# LANGUAGE GHC2024 #-}

module FmaOfInt32 (oops) where

import Data.Int (Int32)
import FuryKernel.Build

oops :: Exp Int32 -> Exp Int32 -> Exp Int32 -> Exp Int32
oops = fmaE
