-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: An alignment of 24 B, which is not a power of two.
{-# LANGUAGE GHC2024 #-}
{-# LANGUAGE OverloadedStrings #-}

module BufferAlignNotPowerOfTwo (oops) where

import FuryKernel.Build

oops :: Sig t (Buf t WriteOnly 24 Float)
oops = buffer @WriteOnly @24 @Float "y" 64
