-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: An i32 narrows to i8 with narrowClamped or narrowWrapped,
{-# LANGUAGE GHC2024 #-}

module NarrowIntImplicitly (oops) where

import Data.Int (Int32, Int8)
import FuryKernel.Build (Exp, narrow)

oops :: Exp Int32 -> Exp Int8
oops = narrow @Int8
