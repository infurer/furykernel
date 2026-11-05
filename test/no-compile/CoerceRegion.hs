-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: Couldn't match type
{-# LANGUAGE GHC2024 #-}

module CoerceRegion (oops) where

import Data.Coerce (coerce)
import FuryKernel.Build

-- Per-lane code coerced into code every thread of the threadgroup runs.
oops :: K Divergent () -> K Uniform ()
oops = coerce
