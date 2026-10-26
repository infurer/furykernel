-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: Couldn't match type
{-# LANGUAGE GHC2024 #-}

module CoerceChecked (oops) where

import Data.Coerce (coerce)
import FuryKernel.Check (Checked)

data T8

data T32

-- A certificate of one target's gate, coerced into another's.
oops :: Checked T32 -> Checked T8
oops = coerce
