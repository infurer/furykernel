-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: Pack of 3 elements of 4 B = 12 B, which is not a power of two.
{-# LANGUAGE GHC2024 #-}
{-# LANGUAGE TypeFamilies #-}

module PackNotPowerOfTwo (oops) where

import FuryKernel.Build
import FuryKernel.Target (Target (..))

data T32

instance Target T32 where
  type TargetName T32 = "T32"
  type SimdWidth T32 = 32
  type MaxThreads T32 = 1024
  type TgMemBytes T32 = 32768
  type MaxBuffers T32 = 31
  type LaneLoadMax T32 = 16

oops :: Buf T32 WriteOnly 16 Float -> K Uniform ()
oops y = storePack @3 y 0 [1, 2, 3]
