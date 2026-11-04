-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: This buffer or table is read-only: declare it ReadWrite to store to it.
{-# LANGUAGE GHC2024 #-}

module StoreViaReadOnly (oops) where

import FuryKernel.Build

oops :: Tab t ReadOnly 16 Float -> K Uniform ()
oops t = storeVia t 0 0 1
