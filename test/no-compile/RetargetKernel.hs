-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: Not in scope: record field
{-# LANGUAGE GHC2024 #-}

module RetargetKernel (oops) where

import FuryKernel.Build (Kernel, kernelProblems)

data T8

data T32

-- A record update that keeps the fields and changes the target.
oops :: Kernel T32 -> Kernel T8
oops k = k {kernelProblems = kernelProblems k}
