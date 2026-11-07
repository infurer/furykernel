-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | What a target library provides: the lowering of checked kernels to one
translation unit, and the launch tables its runtime needs.

The core defines the interface and one backend, the CPU target of
"FuryKernel.Cpu"; furykernel-metal's is the first for a GPU. Backends emit
source only. Runtimes compile it with the vendor's compiler at run time,
so no backend needs a GPU to build or to be tested.
-}
module FuryKernel.Backend (
  Backend (..),
  Lowered (..),
  CompileOptions (optionsLanguageVersion, optionsMath),
  compileOptions,
  Math (..),
  LoweredKernel (..),
) where

import Data.Text (Text)
import Data.Word (Word64)
import FuryKernel.Abi (Abi)
import FuryKernel.Check (Checked)
import FuryKernel.Finding (Finding)
import FuryKernel.IR (Expr, Name)
import Numeric.Natural (Natural)

data Backend t = Backend
  { backendName :: Text
  , backendLower :: [Checked t] -> Either [Finding] Lowered
  {- ^ One translation unit for a set of kernels, which the runtime
  compiles at once.
  -}
  }

data Lowered = Lowered
  { loweredLanguage :: Text
  -- ^ The source language, e.g. @"msl"@.
  , loweredOptions :: CompileOptions
  -- ^ How the runtime compiles the source, for it to mean what the IR does.
  , loweredSource :: Text
  , loweredKernels :: [LoweredKernel]
  }

{- | How a runtime compiles a translation unit, for its kernels to compute
what their IR means: the version of the source language, and the
floating-point semantics the compiler may assume. A target library starts
from 'compileOptions' and sets what it needs, which options that come
later leave as it is.
-}
data CompileOptions = CompileOptions
  { optionsLanguageVersion :: Maybe Text
  {- ^ The version of the source language the source is written in, e.g.
  @"3.1"@ for MSL 3.1, where the language has versions.
  -}
  , optionsMath :: Math
  }
  deriving stock (Eq, Show)

-- | No language version, 'Strict' arithmetic.
compileOptions :: CompileOptions
compileOptions = CompileOptions {optionsLanguageVersion = Nothing, optionsMath = Strict}

-- | The floating-point semantics a compiler may assume.
data Math
  = {- | Arithmetic as the IR defines it: no fast-math assumptions (no NaNs
    or infinities, no reassociation, no approximate reciprocals or
    functions) and no multiplication and addition contracted into one
    rounding, but an 'FuryKernel.IR.EFma', which is one.
    -}
    Strict
  deriving stock (Eq, Show)

-- | One kernel's launch table.
data LoweredKernel = LoweredKernel
  { lkName :: Name
  , lkEntry :: Name
  -- ^ The entry point's name in the source.
  , lkAbi :: Abi
  , lkIrHash :: Word64
  , lkThreads :: Natural
  , lkGroups :: Expr
  {- ^ Threadgroups in the grid, over the parameters. A launch whose grid
  comes to 0 does nothing, and runtimes skip it: a dispatch of no
  threadgroups is refused under Metal's API validation, and does nothing
  without it.
  -}
  , lkBindings :: [(Name, Natural)]
  -- ^ The binding index of every buffer, then every table, by name.
  , lkParamsBinding :: Maybe Natural
  {- ^ The binding index of the parameter struct, if the kernel has one and
  the target binds it: the CPU target passes it as its functions' first
  argument instead.
  -}
  , lkTgMemory :: Natural
  {- ^ Threadgroup memory, in bytes: the total of the kernel's layout
  ('FuryKernel.IR.tgLayout'), each array aligned to 16 bytes at least. A
  target may allocate less: Metal drops the arrays a kernel doesn't use.
  -}
  }
