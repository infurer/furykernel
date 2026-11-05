-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: SIMD-group operations need every lane of the group, and this code
{-# LANGUAGE GHC2024 #-}
{-# LANGUAGE OverloadedStrings #-}

module SimdUnderLaneBranch (oops) where

import FuryKernel.Build

oops :: Exp Float -> K Uniform ()
oops x = whenLane (simdLane .== 0) $ do
  _ <- simdSum "total" x
  pure ()
