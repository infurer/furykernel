-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | Writing programs: a builder that records sources, nodes and values in
program order, and hands back the values to compute with.

Shapes follow from the operations: a linear layer's output is as wide as
its weight has rows, a gated activation's half as wide as its input. The
builder trusts its arguments; "FuryKernel.Program.Check" checks the result,
as the kernel gate does for kernels. Sources keep the names they are given,
which the engine binds them by; values and nodes take a suffix where a name
is taken.
-}
module FuryKernel.Program.Build (
  -- * Building
  P,
  program,

  -- * Handles
  Val (..),
  Src (..),

  -- * Sources
  input,
  rowInts,
  weight,
  rowsF32,
  table,
  cache,

  -- * Operations
  linear,
  linearRouted,
  rmsnorm,
  add,
  concatenate,
  scaleBy,
  gatedGelu,
  geluMul,
  rope,
  softcap,
  route,
  combineRanks,
  kvStore,
  attend,
  gather,
  standardize,
  argmax,

  -- * Outputs
  output,
) where

import Control.Monad (void)
import Control.Monad.Trans.State.Strict (State, execState, gets, modify', state)
import Data.Foldable (toList)
import Data.List (find)
import Data.Maybe (fromMaybe)
import Data.Sequence (Seq)
import Data.Sequence qualified as Seq
import Data.Set (Set)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as T
import FuryKernel.IR (Name)
import FuryKernel.Program.Graph
import FuryKernel.Type (ScalarType (..))
import Numeric.Natural (Natural)

data PState = PState
  { names :: Set Name
  , sources :: Seq Source
  , values :: Seq Value
  , nodes :: Seq Node
  , outputs :: Seq Output
  }

-- | A program being written.
newtype P a = P (State PState a)
  deriving newtype (Functor, Applicative, Monad)

-- | A value: its id and shape.
data Val = Val
  { valId :: ValueId
  , valShape :: Shape
  }
  deriving stock (Eq, Show)

-- | A source: its id and kind.
data Src = Src
  { srcId :: SourceId
  , srcKind :: SourceKind
  }
  deriving stock (Eq, Show)

program :: Name -> P () -> Program
program name (P body) =
  Program
    { programName = name
    , programSources = toList final.sources
    , programValues = toList final.values
    , programNodes = toList final.nodes
    , programOutputs = toList final.outputs
    }
 where
  final = execState body (PState Set.empty Seq.empty Seq.empty Seq.empty Seq.empty)

-- | A value's or a node's name: the author's, or with a suffix if already taken.
fresh :: Name -> State PState Name
fresh base = state $ \s ->
  let candidates = base : [base <> "_" <> T.pack (show i) | i <- [1 :: Int ..]]
      n = fromMaybe base (find (`Set.notMember` s.names) candidates)
   in (n, s {names = Set.insert n s.names})

{- | A source, named exactly as given: the engine binds sources by name. A
name given twice reaches the gate, which refuses it.
-}
addSource :: Name -> SourceKind -> P Src
addSource name kind = P $ do
  i <- gets (Seq.length . (.sources))
  let s = Source (SourceId i) name kind
  modify' (\st -> st {names = Set.insert name st.names, sources = st.sources Seq.|> s})
  pure (Src s.sourceId kind)

{- | A node with the given outputs (name, shape, type), returning them. A
node is named after its first output; one without outputs, an effect,
takes a name of its own.
-}
addNode :: Name -> Op -> [(Name, Shape, ScalarType)] -> P [Val]
addNode name op outs = P $ do
  vals <- traverse value outs
  n <- case vals of
    (_, vn) : _ -> pure vn
    [] -> fresh name
  i <- gets (Seq.length . (.nodes))
  let node = Node (NodeId i) n op (map ((.valId) . fst) vals)
  modify' (\st -> st {nodes = st.nodes Seq.|> node})
  pure (map fst vals)
 where
  value (vname, sh, t) = do
    vn <- fresh vname
    j <- gets (Seq.length . (.values))
    let v = Value (ValueId j) vn sh t
    modify' (\st -> st {values = st.values Seq.|> v})
    pure (Val v.valueId sh, vn)

one :: Name -> Op -> Shape -> ScalarType -> P Val
one name op sh t =
  addNode name op [(name, sh, t)] >>= \case
    [v] -> pure v
    _ -> error "FuryKernel.Program.Build: a node with one output made another number"

------------------------------------------------------------------------------
-- Sources

-- | An activation the engine provides per row, and its value, of the same name.
input :: Name -> Shape -> ScalarType -> P Val
input name sh t = do
  s <- addSource name (SInput sh t)
  P $ do
    i <- gets (Seq.length . (.nodes))
    j <- gets (Seq.length . (.values))
    let v = Value (ValueId j) name sh t
    modify' (\st -> st {values = st.values Seq.|> v, nodes = st.nodes Seq.|> Node (NodeId i) name (Input s.srcId) [v.valueId]})
    pure (Val v.valueId sh)

-- | Per row, @n@ u32, each below @bound@.
rowInts :: Name -> Natural -> Natural -> P Src
rowInts name n bound = addSource name (SRowInts n bound)

{- | A bf16 weight: a linear layer's in the model's @out_in@ layout, outputs
first, anything else @row_major@.
-}
weight :: Name -> [Natural] -> P Src
weight name dims = addSource name (SWeight dims TBF16 (if length dims == 2 then "out_in" else "row_major"))

{- | f32 rows another program wrote, @[rows, width]@ row-major: a table
that a gather reads, the pooling of a vision encoder's patches.
-}
rowsF32 :: Name -> [Natural] -> P Src
rowsF32 name dims = addSource name (SWeight dims TF32 "row_major")

-- | A table of @n@ bf16 weights in the @out_in@ layout, each of this shape.
table :: Name -> Natural -> [Natural] -> P Src
table name n dims = addSource name (STable n dims TBF16 "out_in")

-- | A bf16 KV cache.
cache :: Name -> CacheLayout -> P Src
cache name l = addSource name (SCache l TBF16)

------------------------------------------------------------------------------
-- Operations

weightRows :: Src -> Natural
weightRows s = case s.srcKind of
  SWeight (n : _) _ _ -> n
  STable _ (n : _) _ _ -> n
  _ -> 0

-- | @linear "q" w seg x@: @W x@, in segments of @seg@ (0 for one segment).
linear :: Name -> Src -> Natural -> Val -> P Val
linear name w seg x = one name (Linear w.srcId seg' x.valId) (Shape n seg') TF32
 where
  n = weightRows w
  seg' = if seg == 0 then n else seg

-- | Each rank's expert of the table applied to its input: a segment per rank.
linearRouted :: Name -> Src -> Val -> Val -> P Val
linearRouted name t ids x = one name (LinearRouted t.srcId ids.valId x.valId) (Shape (ranks * n) n) TF32
 where
  n = weightRows t
  ranks = ids.valShape.shapeWidth

-- | @rmsnorm "norm" eps gamma factor x@, per segment of @x@.
rmsnorm :: Name -> Double -> Maybe Src -> Double -> Val -> P Val
rmsnorm name eps g f x = one name (RmsNorm eps (fmap (.srcId) g) f x.valId) x.valShape TF32

add :: Name -> Val -> Val -> P Val
add name a b = one name (Add a.valId b.valId) a.valShape TF32

-- | Each row's @a@, then its @b@, whose segments are as wide.
concatenate :: Name -> Val -> Val -> P Val
concatenate name a b = one name (Concat a.valId b.valId) (Shape (a.valShape.shapeWidth + b.valShape.shapeWidth) a.valShape.shapeSegment) TF32

-- | Times a weight of one element.
scaleBy :: Name -> Src -> Val -> P Val
scaleBy name w x = one name (ScaleBy w.srcId x.valId) x.valShape TF32

-- | @gelu_tanh(a) · b@ of each segment's halves.
gatedGelu :: Name -> Val -> P Val
gatedGelu name x = one name (GatedGelu GeluTanh x.valId) (Shape (x.valShape.shapeWidth `div` 2) (x.valShape.shapeSegment `div` 2)) TF32

-- | @gelu_tanh(a) · b@, elementwise.
geluMul :: Name -> Val -> Val -> P Val
geluMul name a b = one name (GeluMul GeluTanh a.valId b.valId) a.valShape TF32

rope :: Name -> RopeSpec -> Src -> Val -> P Val
rope name r pos x = one name (Rope r pos.srcId x.valId) x.valShape TF32

softcap :: Name -> Double -> Val -> P Val
softcap name c x = one name (Softcap c x.valId) x.valShape TF32

-- | The route to @k@ experts, a softmax over their logits, scaled per expert if given: ids and weights.
route :: Name -> Natural -> Maybe Src -> Val -> P (Val, Val)
route name k scale x =
  addNode name (Route (RouteSpec k (fmap (.srcId) scale) SoftmaxOfTopK) x.valId) [(name <> "_ids", wholeRow k, TU32), (name <> "_weights", wholeRow k, TF32)] >>= \case
    [ids, ws] -> pure (ids, ws)
    _ -> error "FuryKernel.Program.Build: a route without its two outputs"

-- | The rank-weighted sum of a value's segments.
combineRanks :: Name -> Val -> Val -> P Val
combineRanks name w x = one name (CombineRanks w.valId x.valId) (wholeRow x.valShape.shapeSegment) TF32

-- | Stores each row's heads in the cache, at its position of its sequence.
kvStore :: Name -> StoreAt Src -> Val -> P ()
kvStore name at x = void $ addNode name (KvStore ((.srcId) <$> at) x.valId) []

-- | Each row's attention over its sequence's keys and values, as of its position.
attend :: Name -> AttnSpec -> Val -> AttendTo Src -> P Val
attend name spec q to = one name (Attend spec q.valId ((.srcId) <$> to)) q.valShape TF32

{- | Per row, the sum of the table's rows its tokens name (as many as
@tok@ has a row), times @factor@.
-}
gather :: Name -> Src -> Src -> Double -> P Val
gather name t tok f = one name (Gather t.srcId tok.srcId f) (wholeRow width) TF32
 where
  width = case t.srcKind of
    SWeight [_, w] _ _ -> w
    _ -> 0

-- | @(factor · x − bias) · scale@, elementwise, the weights a row wide.
standardize :: Name -> Src -> Src -> Double -> Val -> P Val
standardize name b s f x = one name (Standardize b.srcId s.srcId f x.valId) x.valShape TF32

argmax :: Name -> Val -> P Val
argmax name x = one name (ArgMax x.valId) (wholeRow 1) TU32

------------------------------------------------------------------------------
-- Outputs

-- | A value the engine gets back.
output :: Text -> Val -> P ()
output name v = P (modify' (\st -> st {outputs = st.outputs Seq.|> Output name v.valId}))
