-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: i8 is a storage type, with no arithmetic: widen it to Int32 first.
{-# LANGUAGE GHC2024 #-}

module Int8Arithmetic (oops) where

import Data.Int (Int8)
import FuryKernel.Build

oops :: Exp Int8 -> Exp Int8 -> Exp Int8
oops a b = a + b
