-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: A threadgroup array needs at least one element.
{-# LANGUAGE GHC2024 #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}

module TgArrayEmpty (oops) where

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
oops = needs @0 @Float
