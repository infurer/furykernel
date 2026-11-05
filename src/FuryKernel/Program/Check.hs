-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The program gate: sources, values and nodes of unique ids, increasing
in program order, every value made by one node and every cache stored into
by one; every operation's values and sources of the shapes and types it
takes, rows and caches of at least one element in every dimension, caches
of bf16 and weights of a float type, finite parameters, every value made
before it is read; every index below the size of what it indexes; and
names that generated code can carry, the program's without the @_x_@ that
joins it to its steps' names.

Indices are a routed layer's ids, below the width of the route's or the
argmax's row they come from, and row integers (tokens, slots, block
tables, positions), below the bounds they declare, which the engine keeps
to. Attention over a ring reads only positions the ring holds at once:
its window, or with none every position below its bound, at most the
ring's length.

'checkProgram' returns a 'CheckedProgram', whose constructor is private,
or the findings, each placed in its program and, where a node's, in the
node; the planner takes only checked programs.
-}
module FuryKernel.Program.Check (
  CheckedProgram,
  checkedProgram,
  checkProgram,
) where

import Control.Monad (unless, when)
import Control.Monad.Trans.State.Strict (State, execState, gets, modify')
import Data.Foldable (for_)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (catMaybes, isNothing)
import Data.Text (Text)
import Data.Text qualified as T
import FuryKernel.Check (nameProblem)
import FuryKernel.Finding (Finding (..), Location (..), Rule (..), locatedAt)
import FuryKernel.Program.Graph
import FuryKernel.Type (ScalarType (..), scalarTypeName)
import Numeric.Natural (Natural)

-- | A program that passed the gate.
newtype CheckedProgram = CheckedProgram Program

checkedProgram :: CheckedProgram -> Program
checkedProgram (CheckedProgram p) = p

checkProgram :: Program -> Either [Finding] CheckedProgram
checkProgram p = case map (placedIn p.programName) (names p <> structure p <> rowBounds p <> layouts p <> elementTypes p <> shapes p) of
  [] -> Right (CheckedProgram p)
  fs -> Left fs

-- | A finding placed in its program, if nothing placed it more closely.
placedIn :: Text -> Finding -> Finding
placedIn program f = case f.findingLocation of
  Nothing -> locatedAt (Location program []) f
  Just _ -> f

{- | Ids the planner keys on: unique and increasing in program order, every
value made by one node; and every cache stored into by one node.
-}
structure :: Program -> [Finding]
structure p =
  increasing "source" [i | Source {sourceId = SourceId i} <- p.programSources]
    <> increasing "value" [i | Value {valueId = ValueId i} <- p.programValues]
    <> increasing "node" [i | Node {nodeId = NodeId i} <- p.programNodes]
    <> [ Finding Structure ("The value " <> v.valueName <> " is made by " <> count c <> ".")
       | v <- p.programValues
       , let c = length [() | n <- p.programNodes, v.valueId `elem` n.nodeOutputs]
       , c /= 1
       ]
    <> [ Finding Structure (again.nodeName <> " stores into " <> maybe "a cache" (.sourceName) (programSource p c) <> ", which " <> first.nodeName <> " already does: a program stores into a cache once.")
       | (c, first : others) <- Map.toList stores
       , again <- others
       ]
 where
  stores = Map.fromListWith (flip (<>)) [(c, [n]) | n <- p.programNodes, KvStore (StoreAt c _ _) _ <- [n.nodeOp]]
  increasing what is = case [(a, b) | (a, b) <- zip is (drop 1 is), a >= b] of
    (a, b) : _ -> [Finding Structure ("The " <> what <> " ids don't increase in program order: " <> showT a <> ", then " <> showT b <> ".")]
    [] -> []
  count = \case
    0 -> "no node"
    1 -> "one node"
    c -> showT c <> " nodes"

-- | Caches of at least one of everything.
layouts :: Program -> [Finding]
layouts p =
  [ Finding Shapes ("The cache " <> s.sourceName <> " has " <> what <> ".")
  | s <- p.programSources
  , SCache layout _ <- [s.sourceKind]
  , what <- case layout of
      Ring slots heads len d -> zeros [("0 slots", slots), ("0 heads", heads), ("a length of 0", len), ("a head dimension of 0", d)]
      Paged blocks size heads d -> zeros [("0 blocks", blocks), ("a block size of 0", size), ("0 heads", heads), ("a head dimension of 0", d)]
  ]
 where
  zeros fields = [what | (what, 0) <- fields]

{- | Caches of bf16, the type the program's evaluation rounds their stores to
("FuryKernel.Program.Eval"); weights and tables of a float type, each
naming its layout.
-}
elementTypes :: Program -> [Finding]
elementTypes p =
  [ Finding Shapes ("The cache " <> s.sourceName <> " holds " <> scalarTypeName t <> ", not bf16.")
  | s <- p.programSources
  , SCache _ t <- [s.sourceKind]
  , t /= TBF16
  ]
    <> [ Finding Shapes ("The " <> what <> " " <> s.sourceName <> " holds " <> scalarTypeName t <> ", not bf16, f16 or f32.")
       | s <- p.programSources
       , Just (what, t, _) <- [weighted s.sourceKind]
       , t `notElem` [TBF16, TF16, TF32]
       ]
    <> [ Finding Shapes ("The " <> what <> " " <> s.sourceName <> " names no layout.")
       | s <- p.programSources
       , Just (what, _, layout) <- [weighted s.sourceKind]
       , T.null layout
       ]
 where
  weighted = \case
    SWeight _ t layout -> Just ("weight" :: Text, t, layout)
    STable _ _ t layout -> Just ("table", t, layout)
    _ -> Nothing

{- | Names become identifiers in generated code: the program's prefixes its
kernels, the others name buffers and locals, which must differ.
-}
names :: Program -> [Finding]
names p =
  [Finding Names m | Just m <- map nameProblem (p.programName : everything <> map (.outputName) p.programOutputs)]
    <> [Finding Names ("The program's name " <> p.programName <> " holds _x_, which joins it to its steps' names.") | "_x_" `T.isInfixOf` p.programName]
    <> [ Finding Names ("The name " <> n <> " is used " <> showT c <> " times.")
       | (n, c) <- Map.toList (Map.fromListWith (+) [(n, 1 :: Int) | n <- everything])
       , c > 1
       ]
    <> [ Finding Names ("The output " <> n <> " is named " <> showT c <> " times.")
       | (n, c) <- Map.toList (Map.fromListWith (+) [(o.outputName, 1 :: Int) | o <- p.programOutputs])
       , c > 1
       ]
 where
  -- Inputs share their source's name.
  everything =
    map (.sourceName) p.programSources
      <> [v.valueName | v <- p.programValues, not (isInputValue v.valueId)]
      <> [n.nodeName | n <- p.programNodes, null n.nodeOutputs]
  isInputValue v = case producerOf p v of
    Just Node {nodeOp = Input _} -> True
    _ -> False

-- | No row integer is below a bound of 0.
rowBounds :: Program -> [Finding]
rowBounds p =
  [ Finding Bounds ("The bound of " <> s.sourceName <> ", 0, admits no value.")
  | s <- p.programSources
  , SRowInts _ 0 <- [s.sourceKind]
  ]

-- | Neither infinite nor NaN.
finite :: Double -> Bool
finite v = not (isNaN v || isInfinite v)

------------------------------------------------------------------------------
-- Shapes, types and indices

data St = St
  { made :: Map ValueId Value
  , findings :: [Finding]
  , placeNode :: [Text]
  -- ^ The node checked, if any.
  }

shapes :: Program -> [Finding]
shapes p = reverse (execState (mapM_ node p.programNodes >> outputs) (St Map.empty [] [])).findings
 where
  sources = Map.fromList [(s.sourceId, s) | s <- p.programSources]
  values = Map.fromList [(v.valueId, v) | v <- p.programValues]

  -- What a u32 value is below: a route's ids, the width of its logits; an
  -- argmax, the width of its row. Nothing bounds an input.
  idBound v = case producerOf p v of
    Just Node {nodeOp = Route _ x} -> width x
    Just Node {nodeOp = ArgMax x} -> width x
    _ -> Nothing
  width x = (.valueShape.shapeWidth) <$> Map.lookup x values

  bad :: Text -> State St ()
  bad = finding Shapes

  finding :: Rule -> Text -> State St ()
  finding r m = modify' (\s -> s {findings = locatedAt (Location p.programName s.placeNode) (Finding r m) : s.findings})

  outputs = do
    modify' (\s -> s {placeNode = []})
    for_ p.programOutputs $ \o -> do
      known <- gets (Map.member o.outputValue . (.made))
      unless known $ bad ("The output " <> o.outputName <> " names no value of the program.")

  node :: Node -> State St ()
  node n = do
    modify' (\s -> s {placeNode = [n.nodeName]})
    ins <- traverse (val n) (opValues n.nodeOp)
    srcs <- traverse (src n) (opSources n.nodeOp)
    let outs = [Map.lookup v values | v <- n.nodeOutputs]
    when (any isNothing outs) $ finding Structure (n.nodeName <> " makes a value the program doesn't list.")
    -- What is missing has its finding; the rest is checked with it.
    unless (any isNothing ins || any isNothing srcs || any isNothing outs) $
      op n (catMaybes ins) (catMaybes srcs) (catMaybes outs)
    modify' (\s -> s {made = foldr (\v -> maybe id (Map.insert v) (Map.lookup v values)) s.made n.nodeOutputs})

  val n v = do
    m <- gets (Map.lookup v . (.made))
    when (isNothing m) $ bad (n.nodeName <> " reads a value no earlier node makes.")
    pure m

  src n s = do
    let m = Map.lookup s sources
    when (isNothing m) $ bad (n.nodeName <> " reads a source the program doesn't have.")
    pure m

  op :: Node -> [Value] -> [Source] -> [Value] -> State St ()
  op n ins srcs outs = case (n.nodeOp, ins, srcs, outs) of
    (Input _, [], [Source {sourceKind = SInput sh t}], [o]) -> shaped o sh t
    (Input _, _, _, _) -> expect False "it reads a source that isn't an input"
    (Linear _ seg _, [x], [w], [o]) -> case weight2 w of
      Just (rows, cols) -> do
        f32 x
        expect (x.valueShape.shapeWidth == cols) ("its input has " <> showT x.valueShape.shapeWidth <> " elements for a weight of " <> showT cols <> " columns")
        expect (seg > 0 && rows `mod` seg == 0) ("its segments of " <> showT seg <> " don't divide its " <> showT rows <> " outputs")
        shaped o (Shape rows seg) TF32
      Nothing -> expect False (w.sourceName <> " is not a two-dimensional weight")
    (LinearRouted {}, [ids, x], [t], [o]) -> case t.sourceKind of
      STable entries [rows, cols] _ _ -> do
        u32 ids
        f32 x
        let ranks = ids.valueShape.shapeWidth
            shared = x.valueShape == wholeRow cols
            perRank = x.valueShape == Shape (ranks * cols) cols
        expect (shared || perRank) ("its input is neither one segment of " <> showT cols <> " nor one per rank")
        case idBound ids.valueId of
          Just b -> within (b <= entries) ("the bound of its ids, " <> showT b <> ", is more than the " <> showT entries <> " entries of " <> t.sourceName)
          Nothing -> within False "nothing bounds its ids, which are neither a route's nor an argmax's"
        shaped o (Shape (ranks * rows) rows) TF32
      _ -> expect False (t.sourceName <> " is not a table of two-dimensional weights")
    (RmsNorm eps g factor _, [x], gs, [o]) -> do
      f32 x
      expect (eps > 0 && finite eps) "its epsilon isn't positive and finite"
      expect (finite factor) "its factor isn't finite"
      for_ (zip (maybe [] pure g) gs) $ \(_, gamma) ->
        expect (weight1 gamma == Just x.valueShape.shapeSegment) (gamma.sourceName <> " is not a weight as wide as a segment")
      shaped o x.valueShape TF32
    (Add _ _, [a, b], [], [o]) -> do
      f32 a
      f32 b
      expect (a.valueShape == b.valueShape) "its operands have different shapes"
      shaped o a.valueShape TF32
    (Concat _ _, [a, b], [], [o]) -> do
      f32 a
      f32 b
      expect (a.valueShape.shapeSegment == b.valueShape.shapeSegment) "its operands' segments differ"
      shaped o (Shape (a.valueShape.shapeWidth + b.valueShape.shapeWidth) a.valueShape.shapeSegment) TF32
    (ScaleBy _ _, [x], [w], [o]) -> do
      f32 x
      expect (weight1 w == Just 1) (w.sourceName <> " is not a weight of one element")
      shaped o x.valueShape TF32
    (GatedGelu _ _, [x], [], [o]) -> do
      f32 x
      expect (even x.valueShape.shapeSegment) "its segments have no halves"
      shaped o (Shape (x.valueShape.shapeWidth `div` 2) (x.valueShape.shapeSegment `div` 2)) TF32
    (GeluMul {}, [a, b], [], [o]) -> do
      f32 a
      f32 b
      expect (a.valueShape == b.valueShape) "its operands have different shapes"
      shaped o a.valueShape TF32
    (Rope r _ _, [x], [pos], [o]) -> do
      f32 x
      expect (r.ropeAxes > 0) "it has no axes"
      expect (rowCount pos == Just r.ropeAxes) (pos.sourceName <> " is not " <> showT r.ropeAxes <> " u32 per row, a position an axis")
      expect (x.valueShape.shapeSegment `mod` (2 * max 1 r.ropeAxes) == 0) "its heads have no halves for each axis"
      expect (2 * r.ropeAxes * r.ropeRotated <= x.valueShape.shapeSegment) "it rotates more pairs than a head has"
      expect (r.ropeTheta > 0 && finite r.ropeTheta) "its theta isn't positive and finite"
      shaped o x.valueShape TF32
    (Softcap c _, [x], [], [o]) -> do
      f32 x
      expect (c > 0 && finite c) "its cap isn't positive and finite"
      shaped o x.valueShape TF32
    (Route r _, [x], scales, [ids, ws]) -> do
      f32 x
      whole x
      expect (r.routeK > 0 && r.routeK <= x.valueShape.shapeWidth) ("it takes " <> showT r.routeK <> " of " <> showT x.valueShape.shapeWidth <> " logits")
      for_ scales $ \s -> expect (weight1 s == Just x.valueShape.shapeWidth) (s.sourceName <> " is not a weight with a scale per expert")
      shaped ids (wholeRow r.routeK) TU32
      shaped ws (wholeRow r.routeK) TF32
    (CombineRanks _ _, [w, x], [], [o]) -> do
      f32 w
      f32 x
      whole w
      expect (segmentCount x.valueShape == w.valueShape.shapeWidth) "its weights don't match its value's segments"
      shaped o (wholeRow x.valueShape.shapeSegment) TF32
    (KvStore {}, [x], [c, pos, slot], []) -> case c.sourceKind of
      SCache layout _ -> do
        f32 x
        rowInts1 pos
        sequenceSource c layout slot
        positions layout slot pos
        expect
          (x.valueShape == Shape (cacheKvHeads layout * cacheHeadDim layout) (cacheHeadDim layout))
          ("its value isn't the " <> showT (cacheKvHeads layout) <> " heads of " <> showT (cacheHeadDim layout) <> " of " <> c.sourceName)
      _ -> expect False (c.sourceName <> " is not a cache")
    (Attend a _ _, [q], k : v : pos : slot : reach, [o]) -> case (k.sourceKind, v.sourceKind) of
      (SCache kl _, SCache vl _) -> do
        f32 q
        rowInts1 pos
        mapM_ rowInts1 reach
        expect (length reach == maybe 0 (const 1) a.attnReach) "its sources aren't a cache's, a position's, a sequence's and its reach's"
        sequenceSource k kl slot
        mapM_ (positions kl slot) (pos : reach)
        expect (kl == vl) "its key and value caches are laid out differently"
        expect (a.attnHeads > 0) "it has no query heads"
        expect (a.attnKvHeads > 0 && a.attnHeads `mod` a.attnKvHeads == 0) "its query heads don't share its key and value heads evenly"
        expect (cacheKvHeads kl == a.attnKvHeads && cacheHeadDim kl == a.attnHeadDim) "its caches don't hold its key and value heads"
        expect (q.valueShape == Shape (a.attnHeads * a.attnHeadDim) a.attnHeadDim) "its queries aren't its heads"
        expect (maybe True (> 0) a.attnWindow) "its window is empty"
        expect (a.attnScale > 0 && finite a.attnScale) "its scale isn't positive and finite"
        ringFit k kl a pos reach
        shaped o q.valueShape TF32
      _ -> expect False "it reads keys or values from something that isn't a cache"
    (Gather _ _ factor, [], [t, tok], [o]) -> case weight2 t of
      Just (rows, cols) -> do
        expect (finite factor) "its factor isn't finite"
        expect (isRowInts tok.sourceKind) (tok.sourceName <> " is not u32 per row")
        bounded tok rows ("the " <> showT rows <> " rows of " <> t.sourceName)
        shaped o (wholeRow cols) TF32
      Nothing -> expect False (t.sourceName <> " is not a two-dimensional weight")
    (Standardize _ _ factor _, [x], [b, s], [o]) -> do
      f32 x
      expect (finite factor) "its factor isn't finite"
      expect (weight1 b == Just x.valueShape.shapeWidth) (b.sourceName <> " is not a weight a row wide")
      expect (weight1 s == Just x.valueShape.shapeWidth) (s.sourceName <> " is not a weight a row wide")
      shaped o x.valueShape TF32
    (ArgMax _, [x], [], [o]) -> do
      f32 x
      whole x
      shaped o (wholeRow 1) TU32
    _ -> expect False "it has the wrong number of values or sources"
   where
    expect ok what = unless ok $ bad (n.nodeName <> ": " <> what <> ".")
    within ok what = unless ok $ finding Bounds (n.nodeName <> ": " <> what <> ".")
    f32 v = expect (v.valueType == TF32) (v.valueName <> " is " <> scalarTypeName v.valueType <> ", not f32")
    u32 v = expect (v.valueType == TU32) (v.valueName <> " is " <> scalarTypeName v.valueType <> ", not u32")
    whole v = expect (v.valueShape.shapeSegment == v.valueShape.shapeWidth) (v.valueName <> " is in segments")
    shaped o sh t
      | o.valueShape /= sh || o.valueType /= t = bad (n.nodeName <> ": " <> o.valueName <> " should be " <> showShape sh <> " " <> scalarTypeName t <> ".")
      | sh.shapeWidth == 0 = bad (n.nodeName <> ": " <> o.valueName <> " is " <> showShape sh <> ", a row of no elements.")
      | sh.shapeSegment == 0 || sh.shapeWidth `mod` sh.shapeSegment /= 0 =
          bad (n.nodeName <> ": " <> o.valueName <> " is " <> showShape sh <> ", its segments not dividing its row.")
      | otherwise = pure ()
    {- A ring holds a sequence's last positions, as many as its length:
    attention over one reads no more of them at once than that. A row
    reads from its window's start, or the sequence's, to its reach: at
    most its ahead past its own position, and below its bound. -}
    ringFit c layout a pos reach = case layout of
      Ring _ _ len _ -> do
        let holds = ", more than the " <> showT len <> " that " <> c.sourceName <> "'s ring holds"
            boundOf s = case s.sourceKind of
              SRowInts _ b -> Just b
              _ -> Nothing
        case (a.attnWindow, a.attnReach, reach) of
          (Just w, Nothing, _) ->
            within (w <= len) ("its window of " <> showT w <> " positions is longer than the " <> showT len <> " that " <> c.sourceName <> "'s ring holds")
          (Just w, Just r, [rs]) | Just rb <- boundOf rs -> do
            let spanned = min (w + r.reachAhead) (max w rb)
            within (spanned <= len) ("its window of " <> showT w <> " and its reach of up to " <> showT r.reachAhead <> " past a row's own position read " <> showT spanned <> " positions" <> holds)
          (Nothing, Nothing, _)
            | Just b <- boundOf pos ->
                within (b <= len) ("with no window it reads every position below the bound of " <> pos.sourceName <> ", " <> showT b <> holds)
          (Nothing, Just r, [rs])
            | Just b <- boundOf pos
            , Just rb <- boundOf rs -> do
                let below = max b (min (b + r.reachAhead) rb)
                within (below <= len) ("with no window, and a reach of up to " <> showT r.reachAhead <> " past a row's own position, it reads every position below " <> showT below <> holds)
          -- Sources of other kinds have their findings.
          _ -> pure ()
      Paged {} -> pure ()
    rowInts1 s = expect (rowCount s == Just 1) (s.sourceName <> " is not one u32 per row")
    sequenceSource c layout s = case layout of
      Ring slots _ _ _ -> do
        rowInts1 s
        bounded s slots ("the " <> showT slots <> " slots of " <> c.sourceName)
      Paged blocks _ _ _ -> do
        expect (isRowInts s.sourceKind) (s.sourceName <> " is not a block table")
        bounded s blocks ("the " <> showT blocks <> " blocks of " <> c.sourceName)
    -- A paged cache holds a sequence's positions as far as its block table
    -- reaches.
    positions layout s pos = case (layout, rowCount s) of
      (Paged _ blockSize _ _, Just k) -> bounded pos (k * blockSize) ("the " <> showT (k * blockSize) <> " positions of its block table")
      _ -> pure ()
    bounded s limit what = case s.sourceKind of
      SRowInts _ b -> within (b <= limit) ("the bound of " <> s.sourceName <> ", " <> showT b <> ", is more than " <> what)
      _ -> pure ()
    isRowInts = \case
      SRowInts k _ -> k > 0
      _ -> False

weight2 :: Source -> Maybe (Natural, Natural)
weight2 s = case s.sourceKind of
  SWeight [r, c] _ _ -> Just (r, c)
  _ -> Nothing

rowCount :: Source -> Maybe Natural
rowCount s = case s.sourceKind of
  SRowInts k _ -> Just k
  _ -> Nothing

weight1 :: Source -> Maybe Natural
weight1 s = case s.sourceKind of
  SWeight [n] _ _ -> Just n
  _ -> Nothing

showShape :: Shape -> Text
showShape sh = "[" <> showT sh.shapeWidth <> "/" <> showT sh.shapeSegment <> "]"

showT :: (Show a) => a -> Text
showT = T.pack . show
