-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: Threadgroup array of 9216 elements of 4 B = 36864 B, which takes 36864 B aligned to 16 B; the threadgroup memory of T32 is 32768 B.
{-# LANGUAGE GHC2024 #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}

module TgArrayTooLarge (oops) where

import FuryKernel.Constraint (TgArrayOk)
import FuryKernel.Target (Target (..))

data T32

instance Target T32 where
  type TargetName T32 = "T32"
  type SimdWidth T32 = 32
  type MaxThreads T32 = 1024
  type TgMemBytes T32 = 32768
  type MaxBuffers T32 = 31
  type LaneLoadMax T32 = 16

needs :: forall n a. (TgArrayOk T32 n a) => ()
needs = ()

oops :: ()
oops = needs @9216 @Float
