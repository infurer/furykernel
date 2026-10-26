-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: This buffer is write-only: declare it ReadWrite to load from it.
{-# LANGUAGE GHC2024 #-}

module LoadFromWriteOnly (oops) where

import FuryKernel.Build

oops :: Buf WriteOnly 16 Float -> Exp Float
oops b = load b 0
