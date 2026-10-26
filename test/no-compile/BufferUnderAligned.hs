-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: An alignment of 8 B for elements of 4 B; the least is 16 B.
{-# LANGUAGE GHC2024 #-}
{-# LANGUAGE OverloadedStrings #-}

module BufferUnderAligned (oops) where

import FuryKernel.Build

oops :: Sig t (Buf t ReadOnly 8 Float)
oops = buffer @ReadOnly @8 @Float "x" 64
