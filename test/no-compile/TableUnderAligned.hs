-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: An alignment of 4 B for elements of 2 B; the least is 16 B.
{-# LANGUAGE GHC2024 #-}
{-# LANGUAGE OverloadedStrings #-}

module TableUnderAligned (oops) where

import FuryKernel.Build
import FuryKernel.Type (BF16)

oops :: Sig t (Tab t 4 BF16)
oops = table @4 @BF16 "experts" 4 64
