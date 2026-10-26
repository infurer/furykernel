-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE AllowAmbiguousTypes #-}

{- | Reductions across a threadgroup, from SIMD-group reductions and a
threadgroup array of partials.

The order of the arithmetic depends only on the thread count and the SIMD
width, never on the data or on the threadgroup that runs it, so a row's
result is bitwise the same whatever else is in the batch.
-}
module FuryKernel.Reduce (
  tgReduceSum,
  tgReduceMax,
) where

import Control.Monad (when)
import Data.Proxy (Proxy (..))
import Data.Text qualified as T
import FuryKernel.Build
import FuryKernel.Finding (Rule (..))
import FuryKernel.IR (Name)
import GHC.TypeNats (KnownNat, natVal)

{- | The sum of @x@ over the threads of the threadgroup, in every thread,
bound to the local @name@.

@scratch@ needs a slot per SIMD group. Each SIMD group sums its lanes and
lane 0 stores the sum in the group's slot. After a barrier, the first SIMD
group reduces the slots: lane @l@ adds slots @l@, @l + w@, … in order to
-0, the identity of addition (so a sum of -0s is -0), and the lanes'
results are summed once more, so any number of SIMD groups works. A last
barrier lets the caller reuse @scratch@ at once.
-}
tgReduceSum :: forall n. (KnownNat n) => Name -> TgArr n Float -> Exp Float -> K Uniform (Exp Float)
tgReduceSum = tgReduce @n (+) simdSum (f32 (-0.0))

{- | The maximum of @x@ over the threads of the threadgroup, in every thread,
by 'maxE': a NaN gives way to any number, and over NaNs alone the maximum
is -inf, the reduction's identity.
-}
tgReduceMax :: forall n. (KnownNat n) => Name -> TgArr n Float -> Exp Float -> K Uniform (Exp Float)
tgReduceMax = tgReduce @n maxE simdMax (f32 (negate (1 / 0)))

tgReduce ::
  forall n.
  (KnownNat n) =>
  (Exp Float -> Exp Float -> Exp Float) ->
  (forall r. (Converged r) => Name -> Exp Float -> K r (Exp Float)) ->
  Exp Float ->
  Name ->
  TgArr n Float ->
  Exp Float ->
  K Uniform (Exp Float)
tgReduce combine simdOp neutral name scratch x = do
  groupsN <- simdGroups
  let slots = natVal (Proxy @n)
  when (slots < groupsN) $
    note ThreadgroupMemory $
      "The scratch of reduction "
        <> name
        <> " has "
        <> T.pack (show slots)
        <> " slots for "
        <> T.pack (show groupsN)
        <> " SIMD groups."
  count <- simdGroupCount
  width <- simdWidth
  s <- simdOp (name <> "_simd") x
  whenLane (simdLane .== 0) (tgStore scratch simdGroupIndex s)
  barrier
  whenSimd (simdGroupIndex .== 0) $ do
    acc <- var (name <> "_lane") neutral
    forStep (name <> "_slot") simdLane count width $ \k ->
      acc .= combine (get acc) (tgLoad scratch k)
    t <- simdOp (name <> "_group") (get acc)
    whenLane (simdLane .== 0) (tgStore scratch 0 t)
  barrier
  total <- let_ name (tgLoad scratch 0)
  barrier
  pure total
