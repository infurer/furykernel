-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: A threadgroup needs at least one thread.
{-# LANGUAGE GHC2024 #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}

module ThreadsNone (oops) where

import FuryKernel.Constraint (ThreadsOk)
import FuryKernel.Target (Target (..))

data T32

instance Target T32 where
  type TargetName T32 = "T32"
  type SimdWidth T32 = 32
  type MaxThreads T32 = 1024
  type TgMemBytes T32 = 32768
  type MaxBuffers T32 = 31
  type LaneLoadMax T32 = 16

needs :: forall n. (ThreadsOk T32 n) => ()
needs = ()

oops :: ()
oops = needs @0
