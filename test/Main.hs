-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

module Main (main) where

import Control.Applicative ((<|>))
import Hedgehog.Internal.Property (Skip (SkipNothing))
import Hedgehog.Internal.Seed qualified as Seed
import Test.FuryKernel.AbiTest qualified as AbiTest
import Test.FuryKernel.BuildTest qualified as BuildTest
import Test.FuryKernel.CheckTest qualified as CheckTest
import Test.FuryKernel.ConstraintTest qualified as ConstraintTest
import Test.FuryKernel.CpuTest qualified as CpuTest
import Test.FuryKernel.EmitRustTest qualified as EmitRustTest
import Test.FuryKernel.GemvTest qualified as GemvTest
import Test.FuryKernel.InterpTest qualified as InterpTest
import Test.FuryKernel.JsonTest qualified as JsonTest
import Test.FuryKernel.NumericsTest qualified as NumericsTest
import Test.FuryKernel.ProgramTest qualified as ProgramTest
import Test.FuryKernel.ReduceTest qualified as ReduceTest
import Test.FuryKernel.SelectionTest qualified as SelectionTest
import Test.FuryKernel.VariantTest qualified as VariantTest
import Test.Tasty (adjustOption, defaultMain, testGroup)
import Test.Tasty.Hedgehog (HedgehogReplay (..))

-- | Every property's cases from one seed, unless @--hedgehog-replay@ gives another, so that runs repeat.
fixedSeed :: HedgehogReplay -> HedgehogReplay
fixedSeed (HedgehogReplay given) = HedgehogReplay (given <|> Just (SkipNothing, Seed.from 2026))

main :: IO ()
main =
  defaultMain . adjustOption fixedSeed $
    testGroup
      "furykernel"
      [ NumericsTest.tests
      , ConstraintTest.tests
      , BuildTest.tests
      , CheckTest.tests
      , InterpTest.tests
      , ReduceTest.tests
      , GemvTest.tests
      , VariantTest.tests
      , SelectionTest.tests
      , AbiTest.tests
      , EmitRustTest.tests
      , JsonTest.tests
      , ProgramTest.tests
      , CpuTest.tests
      ]
