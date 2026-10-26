-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: u8 is a storage type, with no arithmetic: widen it to Word32 first.
{-# LANGUAGE GHC2024 #-}

module Word8Arithmetic (oops) where

import Data.Word (Word8)
import FuryKernel.Build

oops :: Exp Word8 -> Exp Word8 -> Exp Word8
oops a b = a * b
