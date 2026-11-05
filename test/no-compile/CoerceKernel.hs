-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: Couldn't match type
{-# LANGUAGE GHC2024 #-}

module CoerceKernel (oops) where

import Data.Coerce (coerce)
import FuryKernel.Build (Kernel)

data T8

data T32

-- A kernel built for one target, coerced into a kernel for another.
oops :: Kernel T32 -> Kernel T8
oops = coerce
