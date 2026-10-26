-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | Targets for the tests: SIMD widths from 8 to 64, so that portable code
is exercised on widths other than the ones in use today.
-}
module Test.FuryKernel.Targets (
  T8,
  T16,
  T32,
  T64,
) where

import FuryKernel.Target (Target (..))

data T8

data T16

data T32

data T64

instance Target T8 where
  type TargetName T8 = "T8"
  type SimdWidth T8 = 8
  type MaxThreads T8 = 256
  type TgMemBytes T8 = 16384
  type MaxBuffers T8 = 31
  type LaneLoadMax T8 = 16

instance Target T16 where
  type TargetName T16 = "T16"
  type SimdWidth T16 = 16
  type MaxThreads T16 = 512
  type TgMemBytes T16 = 32768
  type MaxBuffers T16 = 31
  type LaneLoadMax T16 = 16

instance Target T32 where
  type TargetName T32 = "T32"
  type SimdWidth T32 = 32
  type MaxThreads T32 = 1024
  type TgMemBytes T32 = 32768
  type MaxBuffers T32 = 31
  type LaneLoadMax T32 = 16

instance Target T64 where
  type TargetName T64 = "T64"
  type SimdWidth T64 = 64
  type MaxThreads T64 = 1024
  type TgMemBytes T64 = 65536
  type MaxBuffers T64 = 31
  type LaneLoadMax T64 = 16
