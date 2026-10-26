-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | FuryKernel: a typed eDSL for GPU compute kernels.

What a kernel author imports: the typed surface, the gate, the library's
building blocks, the scalar types, targets and their rules. Backends and
tools import the modules they need ("FuryKernel.IR", "FuryKernel.Abi",
"FuryKernel.Backend", "FuryKernel.Interp", ...) directly.
-}
module FuryKernel (
  module FuryKernel.Build,
  module FuryKernel.Check,
  module FuryKernel.Constraint,
  module FuryKernel.Reduce,
  module FuryKernel.Target,
  module FuryKernel.Type,
  DimRange (..),
  Name,
) where

import FuryKernel.Build
import FuryKernel.Check (Checked, check)
import FuryKernel.Constraint (AlignOk, TgArrayOk, ThreadsOk, withTgArray, withThreads)
import FuryKernel.IR (DimRange (..), Name)
import FuryKernel.Reduce
import FuryKernel.Target (Target (..))
import FuryKernel.Type (BF16 (..), F16 (..), IsScalar)
