-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: Pack of 16 elements of 2 B = 32 B; the widest lane load on T32 is 16 B.
{-# LANGUAGE GHC2024 #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}

module PackTooWide (oops) where

import FuryKernel.Build
import FuryKernel.Target (Target (..))
import FuryKernel.Type (BF16)

data T32

instance Target T32 where
  type TargetName T32 = "T32"
  type SimdWidth T32 = 32
  type MaxThreads T32 = 1024
  type TgMemBytes T32 = 32768
  type MaxBuffers T32 = 31
  type LaneLoadMax T32 = 16

oops :: Buf T32 ReadOnly 32 BF16 -> K Uniform (Pack 16 BF16)
oops x = loadPack @16 "v" x 0
