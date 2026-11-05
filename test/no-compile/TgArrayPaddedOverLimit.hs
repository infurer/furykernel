-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: Threadgroup array of 4093 elements of 4 B = 16372 B, which takes 16384 B aligned to 16 B; the threadgroup memory of Uneven is 16380 B.
{-# LANGUAGE GHC2024 #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}

module TgArrayPaddedOverLimit (oops) where

import FuryKernel.Constraint (TgArrayOk)
import FuryKernel.Target (Target (..))

data Uneven

instance Target Uneven where
  type TargetName Uneven = "Uneven"
  type SimdWidth Uneven = 32
  type MaxThreads Uneven = 1024
  type TgMemBytes Uneven = 16380
  type MaxBuffers Uneven = 31
  type LaneLoadMax Uneven = 16

needs :: forall n a. (TgArrayOk Uneven n a) => ()
needs = ()

oops :: ()
oops = needs @4093 @Float
