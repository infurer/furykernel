-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The program layer: what a model computes in a step, as dataflow
("FuryKernel.Program.Graph"), written with "FuryKernel.Program.Build",
checked by "FuryKernel.Program.Check", given its meaning by
"FuryKernel.Program.Eval", and planned into kernel launches by
"FuryKernel.Program.Plan".
-}
module FuryKernel.Program (
  module FuryKernel.Program.Build,
  module FuryKernel.Program.Check,
  module FuryKernel.Program.Graph,
  module FuryKernel.Program.Plan,
) where

import FuryKernel.Program.Build
import FuryKernel.Program.Check
import FuryKernel.Program.Graph
import FuryKernel.Program.Plan
