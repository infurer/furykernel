-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
-- EXPECT: Pack of 8 elements of 2 B = 16 B from a buffer aligned to 8 B: align it to a multiple of 16 B.
{-# LANGUAGE GHC2024 #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}

module PackUnaligned (oops) where

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

oops :: Buf T32 ReadOnly 8 BF16 -> K Uniform (Pack 8 BF16)
oops x = loadPack @8 "v" x 0
