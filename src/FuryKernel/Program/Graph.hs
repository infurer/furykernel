-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | Programs: what a model computes in one step, as dataflow.

A program is straight-line dataflow over /row values/. A step runs @m@
rows at once, one per token, and every value holds a row of elements for
each of them: the residual stream, a layer's queries, a router's logits.
Rows never mix: a row's elements depend on that row's inputs alone, and on
what the step shares, the weights and the state.

Values are made by nodes, each one operation of a small vocabulary:
linear layers (plain, or routed through a table of experts), RMSNorm,
rotary embeddings, routing, attention over a KV cache, and elementwise
work. Their meaning is exact arithmetic ("FuryKernel.Program.Eval");
how they become kernels is the planner's business
("FuryKernel.Program.Plan"), which no program states.

A value's elements fall into segments: a head of a layer's queries, an
expert's rank of a routed layer's output, or the whole row. Per-segment
operations (a head's norm, its rotation, a rank's gated activation) work on
each segment alone.

What a program reads comes from its sources: per-row activations and
integers from the engine (the residual stream, tokens, positions, the
sequence's slot), weights and tables of weights, named as the model names
its tensors, and state: the KV caches. Outputs name the values the engine
gets back.
-}
module FuryKernel.Program.Graph (
  -- * Shapes
  Shape (..),
  wholeRow,
  segmentCount,

  -- * Sources
  SourceId (..),
  Source (..),
  SourceKind (..),
  CacheLayout (..),
  cacheKvHeads,
  cacheHeadDim,

  -- * Values, nodes and operations
  ValueId (..),
  Value (..),
  NodeId (..),
  Node (..),
  Op (..),
  RopeSpec (..),
  AttnSpec (..),
  RouteSpec (..),
  RouteWeights (..),
  RopePairs (..),
  Gelu (..),
  StoreAt (..),
  AttendTo (..),
  opValues,
  opSources,

  -- * Programs
  Output (..),
  Program (..),
  programValue,
  programSource,
  programNode,
  producerOf,
  renderProgram,
) where

import Data.List (find)
import Data.Text (Text)
import Data.Text qualified as T
import FuryKernel.IR (Name)
import FuryKernel.Type (ScalarType, scalarTypeName)
import GHC.Float (castDoubleToWord64)
import Numeric (showHex)
import Numeric.Natural (Natural)

------------------------------------------------------------------------------
-- Shapes

{- | A row's elements: @width@ of them in segments of @segment@, which
divides the width.
-}
data Shape = Shape
  { shapeWidth :: Natural
  , shapeSegment :: Natural
  }
  deriving stock (Eq, Ord, Show)

-- | A row that is one segment.
wholeRow :: Natural -> Shape
wholeRow n = Shape n n

segmentCount :: Shape -> Natural
segmentCount s = if s.shapeSegment == 0 then 0 else s.shapeWidth `div` s.shapeSegment

------------------------------------------------------------------------------
-- Sources

newtype SourceId = SourceId Int
  deriving stock (Eq, Ord, Show)

data Source = Source
  { sourceId :: SourceId
  , sourceName :: Name
  , sourceKind :: SourceKind
  }
  deriving stock (Eq, Show)

data SourceKind
  = -- | A per-row activation from the engine, of this element type.
    SInput Shape ScalarType
  | {- | Per row, this many u32, each below the bound: a token, a position,
    a slot, a block table. The engine gives none outside it.
    -}
    SRowInts Natural Natural
  | {- | A weight: its shape, outputs first, its element type and the id of
    its layout in the model's format.
    -}
    SWeight [Natural] ScalarType Text
  | {- | A table of weights, one per entry (an expert's), each of this shape,
    type and layout. Kernels reach an entry through its address.
    -}
    STable Natural [Natural] ScalarType Text
  | -- | A KV cache: state that steps write and read, of this element type, bf16.
    SCache CacheLayout ScalarType
  deriving stock (Eq, Show)

{- | How a KV cache holds a sequence's keys or values: each position's heads
of @headDim@ elements.
-}
data CacheLayout
  = {- | @Ring slots kvHeads length headDim@: a ring of @length@ positions
    for each of @slots@ sequences. Position @p@ of the sequence in slot @s@,
    head @h@, starts at @((s · kvHeads + h) · length + p mod length) ·
    headDim@.
    -}
    Ring Natural Natural Natural Natural
  | {- | @Paged blocks blockSize kvHeads headDim@: blocks of @blockSize@
    positions from a pool of @blocks@, which each sequence lists in its
    block table. Each block is an array of its own, which a lowering
    reaches through a table of the blocks' addresses, wherever the engine
    allocated them: position @p@, head @h@, starts at @(h · blockSize + p
    mod blockSize) · headDim@ in block @table[p / blockSize]@. So a block
    is indexed in u32, however many the pool has.
    -}
    Paged Natural Natural Natural Natural
  deriving stock (Eq, Show)

cacheKvHeads :: CacheLayout -> Natural
cacheKvHeads = \case
  Ring _ h _ _ -> h
  Paged _ _ h _ -> h

cacheHeadDim :: CacheLayout -> Natural
cacheHeadDim = \case
  Ring _ _ _ d -> d
  Paged _ _ _ d -> d

------------------------------------------------------------------------------
-- Values, nodes and operations

newtype ValueId = ValueId Int
  deriving stock (Eq, Ord, Show)

-- | A row value, computed in its element type: f32 for activations, u32 for ids.
data Value = Value
  { valueId :: ValueId
  , valueName :: Name
  , valueShape :: Shape
  , valueType :: ScalarType
  }
  deriving stock (Eq, Show)

newtype NodeId = NodeId Int
  deriving stock (Eq, Ord, Show)

-- | One operation, and the values it makes.
data Node = Node
  { nodeId :: NodeId
  , nodeName :: Name
  , nodeOp :: Op
  , nodeOutputs :: [ValueId]
  }
  deriving stock (Eq, Show)

{- | A rotary position embedding on each head, in @axes@ equal parts, the
@a@th turned by the row's @a@th position (one, the usual embedding; two,
an image patch's column and row): in a part of width @d@, the pairs
'ropePairs' names, @(i, i + d/2)@, for @i < rotated@ turn by
@position · theta^(-2i/d)@; the other pairs stay as they are. Every pair rotates when @rotated = d/2@.
-}
data RopeSpec = RopeSpec
  { ropeTheta :: Double
  , ropeRotated :: Natural
  , ropeAxes :: Natural
  , ropePairs :: RopePairs
  }
  deriving stock (Eq, Show)

{- | Decode attention: @heads@ query heads share @kvHeads@ key and value
heads in groups; a query sees the positions of its row's sequence up to its
own, the last @window@ of them if it has one; scores are scaled by @scale@.
-}
data AttnSpec = AttnSpec
  { attnHeads :: Natural
  , attnKvHeads :: Natural
  , attnHeadDim :: Natural
  , attnWindow :: Maybe Natural
  , attnScale :: Double
  }
  deriving stock (Eq, Show)

{- | Routing to experts: the @k@ largest of a row's logits, ties to the
lower index, in rank order, weighed as 'routeWeights' says (a softmax
over those @k@), each times its expert's scale if there is one.
-}
data RouteSpec = RouteSpec
  { routeK :: Natural
  , routeExpertScale :: Maybe SourceId
  , routeWeights :: RouteWeights
  }
  deriving stock (Eq, Show)

{- | Where a step's rows store into a cache: the cache, each row's position,
and its sequence's slot or block table.
-}
data StoreAt s = StoreAt
  { storeCache :: s
  , storePositions :: s
  , storeSequences :: s
  }
  deriving stock (Eq, Show, Functor)

{- | What an attention reads: the keys' and the values' caches, each row's
position, and its sequence's slot or block table.
-}
data AttendTo s = AttendTo
  { attendKeys :: s
  , attendValues :: s
  , attendPositions :: s
  , attendSequences :: s
  }
  deriving stock (Eq, Show, Functor)

-- | Which elements of a head turn together.
data RopePairs
  = -- | @(i, i + d/2)@ in a part of width @d@: its halves, as GPT-NeoX pairs them.
    RopeHalves
  deriving stock (Eq, Show, Enum, Bounded)

-- | The GELU an activation computes.
data Gelu
  = -- | @x/2 · (1 + tanh (√(2/π) · (x + 0.044715 x³)))@.
    GeluTanh
  deriving stock (Eq, Show, Enum, Bounded)

-- | How a route weighs the experts it takes.
data RouteWeights
  = -- | A softmax over the @k@ logits taken.
    SoftmaxOfTopK
  deriving stock (Eq, Show, Enum, Bounded)

data Op
  = -- | The value of an input source: the engine's activation.
    Input SourceId
  | {- | @y = W x@ per row: the weight @[n, k]@, the input of width @k@; the
    output has width @n@, in segments of the given size.
    -}
    Linear SourceId Natural ValueId
  | {- | Per rank @r@ of the route, @y_r = W[ids_r] x_r@: the table's entries
    @[n, k]@, the route's ids, the input (one segment of @k@ that every rank
    takes, or one per rank). The output has a segment of @n@ per rank.
    -}
    LinearRouted SourceId ValueId ValueId
  | {- | RMSNorm of each segment, @x · (mean x² + eps)^(-1/2)@, times its
    gamma if given (a segment wide), times a constant factor.
    -}
    RmsNorm Double (Maybe SourceId) Double ValueId
  | Add ValueId ValueId
  | {- | Each row's first value, then its second: their segments, which are
    as wide, in that order.
    -}
    Concat ValueId ValueId
  | -- | Times a weight of one element.
    ScaleBy SourceId ValueId
  | {- | Per segment, @gelu(a) · b@ of its halves @a@ and @b@, the GELU of
    the given form: the output's segments are half as wide.
    -}
    GatedGelu Gelu ValueId
  | -- | @gelu(a) · b@, elementwise, the GELU of the given form.
    GeluMul Gelu ValueId ValueId
  | {- | Each head (segment) rotated by the row's positions, from the source,
    as many a row as the embedding has axes.
    -}
    Rope RopeSpec SourceId ValueId
  | -- | @cap · tanh(x / cap)@.
    Softcap Double ValueId
  | -- | The route of each row: its ids (u32) and weights (f32), @k@ each.
    Route RouteSpec ValueId
  | {- | Per row, the sum over ranks in rank order of the weight times the
    rank's segment: the weights (one per rank) and the input (a segment
    per rank).
    -}
    CombineRanks ValueId ValueId
  | {- | Stores each row's heads (segments) into the cache, at the row's
    position (a source) of its sequence (its slot, or its block table).
    The value is rounded to bf16, the cache's element type.
    -}
    KvStore (StoreAt SourceId) ValueId
  | {- | Attention of each row's queries (heads as segments) over its
    sequence's keys and values in the caches, as of the row's position.
    -}
    Attend AttnSpec ValueId (AttendTo SourceId)
  | -- | Each row's token's row of the table, times a constant factor.
    Gather SourceId SourceId Double
  | -- | The index of the row's largest element, ties to the lowest.
    ArgMax ValueId
  deriving stock (Eq, Show)

-- | The values an operation reads, in order.
opValues :: Op -> [ValueId]
opValues = \case
  Input _ -> []
  Linear _ _ x -> [x]
  LinearRouted _ ids x -> [ids, x]
  RmsNorm _ _ _ x -> [x]
  Add a b -> [a, b]
  Concat a b -> [a, b]
  ScaleBy _ x -> [x]
  GatedGelu _ x -> [x]
  GeluMul _ a b -> [a, b]
  Rope _ _ x -> [x]
  Softcap _ x -> [x]
  Route _ x -> [x]
  CombineRanks w x -> [w, x]
  KvStore _ x -> [x]
  Attend _ q _ -> [q]
  Gather {} -> []
  ArgMax x -> [x]

-- | The sources an operation reads or, for a cache, writes.
opSources :: Op -> [SourceId]
opSources = \case
  Input s -> [s]
  Linear w _ _ -> [w]
  LinearRouted t _ _ -> [t]
  RmsNorm _ g _ _ -> maybe [] pure g
  Add {} -> []
  Concat {} -> []
  ScaleBy w _ -> [w]
  GatedGelu {} -> []
  GeluMul {} -> []
  Rope _ p _ -> [p]
  Softcap {} -> []
  Route spec _ -> maybe [] pure spec.routeExpertScale
  CombineRanks {} -> []
  KvStore (StoreAt c p s) _ -> [c, p, s]
  Attend _ _ (AttendTo k v p s) -> [k, v, p, s]
  Gather t tok _ -> [t, tok]
  ArgMax _ -> []

------------------------------------------------------------------------------
-- Programs

-- | A value the engine gets back, by name.
data Output = Output
  { outputName :: Name
  , outputValue :: ValueId
  }
  deriving stock (Eq, Show)

{- | A program: its sources, values and nodes each with ids that increase
in the list's order, every value made by one node.
-}
data Program = Program
  { programName :: Name
  , programSources :: [Source]
  , programValues :: [Value]
  , programNodes :: [Node]
  -- ^ In order: a node reads only sources and the values of nodes before it.
  , programOutputs :: [Output]
  }
  deriving stock (Eq, Show)

programValue :: Program -> ValueId -> Maybe Value
programValue p v = find ((== v) . (.valueId)) p.programValues

programSource :: Program -> SourceId -> Maybe Source
programSource p s = find ((== s) . (.sourceId)) p.programSources

programNode :: Program -> NodeId -> Maybe Node
programNode p n = find ((== n) . (.nodeId)) p.programNodes

-- | The node that makes a value.
producerOf :: Program -> ValueId -> Maybe Node
producerOf p v = find ((v `elem`) . (.nodeOutputs)) p.programNodes

{- | A deterministic text form, for golden tests and messages. Floats are
written by their bits.
-}
renderProgram :: Program -> Text
renderProgram p =
  T.unlines $
    ["program " <> p.programName]
      <> map source p.programSources
      <> map node p.programNodes
      <> ["  output " <> o.outputName <> " = " <> value o.outputValue | o <- p.programOutputs]
 where
  source s = "  source " <> s.sourceName <> " : " <> kind s.sourceKind
  kind = \case
    SInput sh t -> "input " <> shape sh <> " " <> scalarTypeName t
    SRowInts n b -> "row_ints " <> showT n <> " below " <> showT b
    SWeight dims t layout -> "weight " <> dimsT dims <> " " <> scalarTypeName t <> " " <> layout
    STable n dims t layout -> "table " <> showT n <> " x " <> dimsT dims <> " " <> scalarTypeName t <> " " <> layout
    SCache l t -> "cache " <> layoutT l <> " " <> scalarTypeName t
  layoutT = \case
    Ring s h n d -> "ring slots " <> showT s <> " heads " <> showT h <> " length " <> showT n <> " dim " <> showT d
    Paged b n h d -> "paged blocks " <> showT b <> " size " <> showT n <> " heads " <> showT h <> " dim " <> showT d
  shape sh = "[" <> showT sh.shapeWidth <> "/" <> showT sh.shapeSegment <> "]"
  dimsT ds = "[" <> T.intercalate ", " (map showT ds) <> "]"
  node n =
    "  "
      <> T.intercalate ", " [value v <> " : " <> valueInfo v | v <- n.nodeOutputs]
      <> (if null n.nodeOutputs then "" else " = ")
      <> n.nodeName
      <> " "
      <> op n.nodeOp
  valueInfo v = case programValue p v of
    Just val -> shape val.valueShape <> " " <> scalarTypeName val.valueType
    Nothing -> "?"
  value v = maybe ("%" <> showT (unValue v)) (.valueName) (programValue p v)
  unValue (ValueId i) = i
  src s = maybe "?" (.sourceName) (programSource p s)
  op = \case
    Input s -> "input(" <> src s <> ")"
    Linear w seg x -> "linear(" <> src w <> ", " <> value x <> ") segment " <> showT seg
    LinearRouted t ids x -> "linear_routed(" <> src t <> ", " <> value ids <> ", " <> value x <> ")"
    RmsNorm eps g f x ->
      "rmsnorm(" <> value x <> ", eps " <> double eps <> maybe "" ((", gamma " <>) . src) g <> ", factor " <> double f <> ")"
    Add a b -> "add(" <> value a <> ", " <> value b <> ")"
    Concat a b -> "concat(" <> value a <> ", " <> value b <> ")"
    ScaleBy w x -> "scale_by(" <> src w <> ", " <> value x <> ")"
    GatedGelu GeluTanh x -> "gated_gelu(" <> value x <> ")"
    GeluMul GeluTanh a b -> "gelu_mul(" <> value a <> ", " <> value b <> ")"
    Rope r pos x ->
      "rope("
        <> value x
        <> ", "
        <> src pos
        <> ", theta "
        <> double r.ropeTheta
        <> ", rotated "
        <> showT r.ropeRotated
        <> (if r.ropeAxes == 1 then "" else ", axes " <> showT r.ropeAxes)
        <> pairing r.ropePairs
        <> ")"
    Softcap c x -> "softcap(" <> value x <> ", " <> double c <> ")"
    Route r x -> "route(" <> value x <> ", k " <> showT r.routeK <> maybe "" ((", scale " <>) . src) r.routeExpertScale <> weighing r.routeWeights <> ")"
    CombineRanks w x -> "combine_ranks(" <> value w <> ", " <> value x <> ")"
    KvStore (StoreAt c pos slot) x -> "kv_store(" <> src c <> ", " <> src pos <> ", " <> src slot <> ", " <> value x <> ")"
    Attend a q (AttendTo k v pos slot) ->
      "attend("
        <> T.intercalate ", " [value q, src k, src v, src pos, src slot]
        <> ", heads "
        <> showT a.attnHeads
        <> "/"
        <> showT a.attnKvHeads
        <> ", dim "
        <> showT a.attnHeadDim
        <> maybe "" ((", window " <>) . showT) a.attnWindow
        <> ", scale "
        <> double a.attnScale
        <> ")"
    Gather t tok f -> "gather(" <> src t <> ", " <> src tok <> ", factor " <> double f <> ")"
    ArgMax x -> "argmax(" <> value x <> ")"
  double d = "f64:0x" <> T.pack (showHex (castDoubleToWord64 d) "")
  -- The parameters with one value so far print as nothing.
  pairing RopeHalves = ""
  weighing SoftmaxOfTopK = ""

showT :: (Show a) => a -> Text
showT = T.pack . show
