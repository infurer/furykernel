-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: Bool has no arithmetic: use .&&, .|| and notE.
{-# LANGUAGE GHC2024 #-}

module FmaOfBool (oops) where

import FuryKernel.Build (Exp, fmaE)

oops :: Exp Bool -> Exp Bool -> Exp Bool -> Exp Bool
oops = fmaE
