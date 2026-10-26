-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: This buffer is read-only: declare it ReadWrite to store to it.
{-# LANGUAGE GHC2024 #-}

module StoreToReadOnly (oops) where

import FuryKernel.Build

oops :: Buf ReadOnly 16 Float -> K Uniform ()
oops b = store b 0 1
