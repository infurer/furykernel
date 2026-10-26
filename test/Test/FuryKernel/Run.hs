-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

-- | Checking and running kernels in tests.
module Test.FuryKernel.Run (
  runOn,
) where

import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text qualified as T
import Data.Word (Word64)
import FuryKernel.Build (Kernel)
import FuryKernel.Check (check)
import FuryKernel.IR (Name)
import FuryKernel.Interp (RunError (..), Value, launch, run, withSeed)
import FuryKernel.Target (Target)

-- | Checks a kernel and runs it; a kernel the gate rejects is an error.
runOn ::
  (Target t) =>
  Kernel t ->
  [(Name, Value)] ->
  [(Name, [Value])] ->
  Word64 ->
  Either RunError (Map Name [Value])
runOn k ps bs seed = case check k of
  Left fs -> Left (Internal (T.pack (show fs)))
  Right c -> run c (withSeed seed (launch (Map.fromList ps) (Map.fromList bs)))
