-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE AllowAmbiguousTypes #-}

{- | Targets: what a GPU family makes legal.

A target is a type whose associated types state its limits. Static checks
read them at the type level (see "FuryKernel.Constraint"); the gate and
dynamic mode read them reflected once into 'TargetCaps'. Each target
library defines its own (furykernel-metal's @AppleGPU9@, for one); the core
defines one, "FuryKernel.Cpu"'s @Cpu@, and its portable code is generic in
the target.

What a target makes legal is not what makes it fast: measured profiles
(bandwidth, cores, power state) live with the tuning, not here.

Later stages add fragments, features, atomics, forward progress and the
minimum shading-language version.
-}
module FuryKernel.Target (
  Target (..),
  TargetCaps (..),
  targetCaps,
) where

import Data.Kind (Constraint, Type)
import Data.Proxy (Proxy (..))
import Data.Text (Text)
import Data.Text qualified as T
import GHC.TypeLits (KnownSymbol, Symbol, symbolVal)
import GHC.TypeNats (KnownNat, Nat, natVal)
import Numeric.Natural (Natural)

type Target :: Type -> Constraint
class
  ( KnownSymbol (TargetName t)
  , KnownNat (SimdWidth t)
  , KnownNat (MaxThreads t)
  , KnownNat (TgMemBytes t)
  , KnownNat (MaxBuffers t)
  , KnownNat (LaneLoadMax t)
  ) =>
  Target t
  where
  -- | The name in messages, e.g. @"AppleGPU9"@.
  type TargetName t :: Symbol

  -- | Lanes per SIMD group.
  type SimdWidth t :: Nat

  -- | Threads per threadgroup.
  type MaxThreads t :: Nat

  -- | Threadgroup memory per threadgroup, in bytes.
  type TgMemBytes t :: Nat

  -- | Buffer bindings per kernel, the parameter struct included.
  type MaxBuffers t :: Nat

  -- | The widest load or store of one lane, in bytes.
  type LaneLoadMax t :: Nat

-- | A target's limits as terms.
data TargetCaps = TargetCaps
  { capName :: Text
  , capSimdWidth :: Natural
  , capMaxThreads :: Natural
  , capTgMemBytes :: Natural
  , capMaxBuffers :: Natural
  , capLaneLoadMax :: Natural
  }
  deriving stock (Eq, Show)

targetCaps :: forall t. (Target t) => TargetCaps
targetCaps =
  TargetCaps
    { capName = T.pack (symbolVal (Proxy @(TargetName t)))
    , capSimdWidth = nat @(SimdWidth t)
    , capMaxThreads = nat @(MaxThreads t)
    , capTgMemBytes = nat @(TgMemBytes t)
    , capMaxBuffers = nat @(MaxBuffers t)
    , capLaneLoadMax = nat @(LaneLoadMax t)
    }
 where
  nat :: forall n. (KnownNat n) => Natural
  nat = natVal (Proxy @n)
