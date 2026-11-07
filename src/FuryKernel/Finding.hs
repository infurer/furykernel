-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE PatternSynonyms #-}

{- | What the gate reports when a kernel breaks a rule.

A finding is made with 'Finding', a rule and a message, and given a place
with 'locatedAt'; its fields are read by name. Fields that come later
leave both as they are.
-}
module FuryKernel.Finding (
  Rule (..),
  Finding (Finding, findingRule, findingMessage, findingLocation),
  Location (..),
  locatedAt,
  renderFinding,
) where

import Data.Text (Text)
import Data.Text qualified as T

-- | The rules, one constructor per family of checks.
data Rule
  = {- | Threads per threadgroup: positive, a multiple of the SIMD width,
    within the target's limit.
    -}
    Threads
  | -- | Threadgroup memory, per array and in total.
    ThreadgroupMemory
  | -- | The number of buffer bindings.
    Buffers
  | {- | Alignments of arrays in device memory: a power of two, at least
    16 B and the element size.
    -}
    Alignment
  | {- | Packs: at least two elements, a power of two in size, no wider than
    a lane loads, from buffers aligned to them.
    -}
    Packs
  | -- | Names: unique, and valid identifiers in every target language.
    Names
  | -- | Well-typed expressions and statements.
    Typing
  | {- | Launch sizes: every dimension with legal values, and the grid and
    the buffers' and tables' sizes in u32 at every step, dividing by no
    zero, for every legal value of their dimensions.
    -}
    LaunchSizes
  | -- | Conditions and loop bounds as uniform as their constructs require.
    Uniformity
  | {- | Barriers only where every thread arrives, SIMD-group operations
    only where every lane does.
    -}
    Regions
  | {- | Hazards: no two threads touch one element of memory unordered while
    one of them stores to it.
    -}
    Hazards
  | {- | Programs: sources, values and nodes of unique ids, increasing in
    program order, and every value made by one node.
    -}
    Structure
  | {- | Programs: every operation's values and sources of the shapes and
    types it takes, and in order.
    -}
    Shapes
  | {- | Programs: every index below the size of what it indexes: routed
    ids, tokens, slots, blocks and positions.
    -}
    Bounds
  | {- | Fragment products only of the element types the target takes, and
    fragments' rows at least 8 elements apart in memory, for every legal
    value of the dimensions.
    -}
    Fragments
  | -- | The kernel checked for the target it was built for.
    Target
  | {- | Overrides: the ABI of the kernel they replace, together every legal
    value of its dimensions, and contracts at least as permissive.
    -}
    Overrides
  deriving stock (Eq, Ord, Show, Enum, Bounded)

data Finding = MkFinding
  { findingRule :: Rule
  , findingMessage :: Text
  , findingLocation :: Maybe Location
  }
  deriving stock (Eq, Show)

-- | A finding of a rule, with its message and no place.
pattern Finding :: Rule -> Text -> Finding
pattern Finding rule message <- MkFinding rule message _
  where
    Finding rule message = MkFinding rule message Nothing

{-# COMPLETE Finding #-}

{- | Where a finding is: the kernel or program it is about, and in it the
places down to the one at fault, outermost first (a statement, a node, an
operand).
-}
data Location = Location
  { locationIn :: Text
  , locationPath :: [Text]
  }
  deriving stock (Eq, Show)

-- | The finding, at a place.
locatedAt :: Location -> Finding -> Finding
locatedAt place f = f {findingLocation = Just place}

-- | @Rule: message@, and where the finding has a place, @(in unit, place, …)@.
renderFinding :: Finding -> Text
renderFinding f =
  T.pack (show f.findingRule)
    <> ": "
    <> f.findingMessage
    <> maybe "" (\l -> " (in " <> T.intercalate ", " (l.locationIn : l.locationPath) <> ")") f.findingLocation
