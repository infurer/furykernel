-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The planner: a program as steps, each one kernel launch, and the cut
points where steps meet.

The rule that places the cuts is the GPU's: no threadgroup can wait for
another inside a kernel. A value is /stored/ once the step that writes it
has finished, and from the next step on any threadgroup may read it.
Within a step a threadgroup has only what it computes itself.

* __Anchors__ set a step's grid: a linear layer's threadgroups each make a
  tile of its outputs, attention's each a split of a row's positions for
  one key and value head. Anchors are what must not be computed twice:
  they read the weights and the caches. Independent anchors of one family
  share a step, each threadgroup working on one of them.
* __Recomputed__ operations run in every threadgroup that needs them,
  before the anchors, from stored values: a norm's sum of squares over a
  stored row, elementwise work, a route from stored logits. An anchor
  reads its input through them, never stored. A value whose recomputation
  takes a pass over a row and that more than this step needs is stored by
  the step that computes it first, once. Asked to ('optStoreGelus'), a
  plan stores a GELU an anchor reads first, by a light step: a linear
  layer's threadgroups would each recompute it for their outputs, a tanh
  an element each time.
* __Epilogues__ run on a linear layer's tile: elementwise work on its
  outputs. Attention's tile holds its splits' partials, which nothing but
  its combine reads.
* __Cuts__ come where an operation needs what several threadgroups of a
  step made: a norm of a linear layer's output row, a rotation pairing
  elements of different tiles, attention's combine of its splits. The
  tile is stored, and the operation is recomputed from it later.
* __Light steps__ hold what no anchor step can: stores into caches, which
  must run once per element and land before attention reads them, or
  are staged for it; attention's combine, whose cost grows with the
  context, before the linear layer that reads it; and the program's
  outputs that no anchor computes. An output a combine makes that a
  later step reads too is stored by the step that combines it, so that
  no step combines it again; a program's output made only from what that
  step has is made and stored there.
* __Staged stores__: a store into a ring lands before the attention that
  reads the ring if the ring has room for a step's rows of one sequence
  ('optSequenceRows'), at consecutive positions, past the positions a row
  reads (its window, or with none every position below the bound): rows
  at most the ring's length less that, and one. Otherwise storing first
  would overwrite keys the step's first rows read, and the store is
  staged: a light step keeps its values as the cache holds them, the
  attention reads the step's own positions from there and the rest from
  the ring, and the first light step after the last attention reading the
  ring lands them in it.

Every value's arithmetic is fixed by the program, so a placement changes
where it runs, never its bits: with the canonical reduction orders of the
lowering, a plan is free to recompute or store anything.
-}
module FuryKernel.Program.Plan (
  -- * Plans
  Plan (..),
  Step (..),
  Work (..),
  Stored (..),
  Form (..),
  workNode,
  stepAnchors,
  isLightStep,
  isStepName,

  -- * Planning
  PlanOptions (..),
  defaultPlanOptions,
  plan,

  -- * What the rules see
  Family (..),
  anchorFamily,
  Cost (..),
  nodeCost,
  renderPlan,
) where

import Control.Monad (forM_, unless, void, when)
import Control.Monad.Trans.State.Strict (State, execState, get, gets, modify')
import Data.Char (isDigit)
import Data.Foldable (toList)
import Data.List (nub, sortOn)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe, listToMaybe, mapMaybe)
import Data.Sequence (Seq)
import Data.Sequence qualified as Seq
import Data.Set (Set)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as T
import FuryKernel.IR (Name)
import FuryKernel.Program.Check (CheckedProgram, checkedProgram)
import FuryKernel.Program.Graph
import FuryKernel.Type (ScalarType (..), scalarTypeName)
import Numeric.Natural (Natural)

------------------------------------------------------------------------------
-- Plans

data Plan = Plan
  { planProgram :: Program
  , planSteps :: [Step]
  }

-- | One kernel launch.
data Step = Step
  { stepIndex :: Int
  , stepName :: Name
  , stepWork :: [Work]
  -- ^ In program order.
  , stepStores :: [Stored]
  -- ^ What it stores, for later steps and the engine.
  , stepReads :: [Stored]
  -- ^ What earlier steps stored that it reads.
  , stepCaches :: [SourceId]
  -- ^ The caches it reads or writes.
  }

-- | A node's part in a step.
data Work
  = -- | An anchor: its tiles set the grid.
    Anchor NodeId
  | -- | Computed in every threadgroup before the anchors, from stored values.
    Recompute NodeId
  | -- | Computed on a linear layer's tiles.
    Epilogue NodeId
  | -- | Attention's combine of its stored partials.
    Combine NodeId
  | -- | A store into a cache, once per element.
    Effect NodeId
  | -- | A staged store into a ring: its values kept for attention, once per element.
    Stage NodeId
  | -- | A staged store landing in its ring, once per element.
    Land NodeId
  deriving stock (Eq, Show)

workNode :: Work -> NodeId
workNode = \case
  Anchor n -> n
  Recompute n -> n
  Epilogue n -> n
  Combine n -> n
  Effect n -> n
  Stage n -> n
  Land n -> n

stepAnchors :: Step -> [NodeId]
stepAnchors s = [n | Anchor n <- s.stepWork]

isLightStep :: Step -> Bool
isLightStep = null . stepAnchors

{- | Whether a name is a step's, as 'plan' names them: @s@, the step's index,
@_@ and a word, as @s0_x@ or @s3_combine@. A name that joins a program's to
a step's with @_x_@ splits back at the first @_x_@ whose right side is one
("FuryKernel.Variant"'s @splitJoin@): a program's name holds no @_x_@.
-}
isStepName :: Text -> Bool
isStepName t = case T.span isDigit <$> T.stripPrefix "s" t of
  Just (index, rest) -> not (T.null index) && maybe False (not . T.null) (T.stripPrefix "_" rest)
  Nothing -> False

-- | A value in memory between steps.
data Stored = Stored
  { storedValue :: ValueId
  , storedForm :: Form
  }
  deriving stock (Eq, Ord, Show)

data Form
  = -- | Its elements, of this type.
    Elements ScalarType
  | -- | Attention's partials: per split, its maximum, sum and accumulator.
    Partials
  | -- | A staged store's values, as the cache holds them.
    Staged SourceId
  deriving stock (Eq, Ord, Show)

------------------------------------------------------------------------------
-- What the rules see

-- | Anchors that can share a step.
data Family
  = -- | Linear layers, plain or routed.
    Gemv
  | Attention
  deriving stock (Eq, Ord, Show)

anchorFamily :: Op -> Maybe Family
anchorFamily = \case
  Linear {} -> Just Gemv
  LinearRouted {} -> Just Gemv
  Attend {} -> Just Attention
  _ -> Nothing

-- | What recomputing an operation costs a threadgroup.
data Cost
  = -- | Elementwise, from what it reads.
    Cheap
  | -- | A pass over a row or a segment first: a reduction.
    Pass
  | -- | More than any anchor should repeat: it grows with the context.
    Expensive
  deriving stock (Eq, Ord, Show)

{- | What recomputing an operation of the program costs a threadgroup: what
its kind costs, and more where its sources say so (a gather that sums
rows, which an anchor would repeat for every output it makes).
-}
nodeCost :: Program -> Op -> Cost
nodeCost p = \case
  RmsNorm {} -> Pass
  Route {} -> Pass
  ArgMax {} -> Pass
  Gather _ tok _
    | Just Source {sourceKind = SRowInts k _} <- programSource p tok
    , k > 1 ->
        Expensive
  _ -> Cheap

-- | Whether an operation reads a tiled value elementwise.
pointwise :: Op -> Bool
pointwise = \case
  Add {} -> True
  GeluMul {} -> True
  ScaleBy {} -> True
  Softcap {} -> True
  KvStore {} -> True
  _ -> False

isEffect :: Op -> Bool
isEffect = \case
  KvStore {} -> True
  _ -> False

------------------------------------------------------------------------------
-- Planning

data PlanOptions = PlanOptions
  { optHorizontal :: Bool
  -- ^ Independent anchors of one family share a step.
  , optGated :: Bool
  {- ^ Routed layers wait for their route: the engine fetches experts in
  between, so the route's ids are stored by an earlier step. So are the
  ids of routes the program outputs, which the engine reads then too: a
  prediction of a later layer's experts, say.
  -}
  , optStoreGelus :: Bool
  {- ^ A GELU an anchor reads is stored first, by a light step, not
  recomputed for every output of the anchor: worth a launch where a
  linear layer's tiles are a few rows of outputs each, a vector's
  product.
  -}
  , optSequenceRows :: Natural
  {- ^ The most rows of one sequence a step holds, at consecutive
  positions: a store into a ring without room for them past what a row
  reads is staged.
  -}
  }
  deriving stock (Eq, Show)

-- | A step of one row per sequence, as decode's.
defaultPlanOptions :: PlanOptions
defaultPlanOptions = PlanOptions {optHorizontal = True, optGated = False, optStoreGelus = False, optSequenceRows = 1}

{- | How many rows of one sequence, at consecutive positions, a step can
store into a ring of this length before an attention reads it, with this
window over positions below this bound: the length less the positions a
row reads (the window, or with none every position below the bound), and
one.
-}
ringRoom :: Natural -> Maybe Natural -> Natural -> Natural
ringRoom len window bound
  | seen > len = 0
  | otherwise = len - seen + 1
 where
  seen = fromMaybe bound window

-- | Where a value is.
data Avail
  = -- | Stored by this step; -1 for the program's inputs.
    AStored Int
  | -- | Recomputable from stored values, at this cost.
    AVirtual Cost
  | -- | In the tiles of a step's anchor.
    ATile Int NodeId
  | -- | Attention's output, stored as partials by this step.
    APartials Int
  deriving stock (Eq, Show)

data StepB = StepB
  { sbIndex :: Int
  , sbWork :: Seq Work
  , sbStores :: Seq Stored
  , sbFamily :: Maybe Family
  , sbWrites :: Set SourceId
  }

data PS = PS
  { avail :: Map ValueId Avail
  , closed :: Seq StepB
  , open :: Maybe StepB
  , next :: Int
  , landing :: [(Int, NodeId, SourceId)]
  {- ^ Staged stores to land: each after this step, the last attention's
  that reads the ring.
  -}
  }

plan :: PlanOptions -> CheckedProgram -> Plan
plan opts checked = Plan p (map finish (toList final.closed))
 where
  p = checkedProgram checked
  final = execState (mapM_ node p.programNodes >> outputs >> landRest >> closeOpen) (PS Map.empty Seq.empty Nothing 0 [])

  nodes = Map.fromList [(n.nodeId, n) | n <- p.programNodes]
  producer = Map.fromList [(v, n) | n <- p.programNodes, v <- n.nodeOutputs]
  consumers :: Map ValueId [NodeId]
  consumers = Map.fromListWith (flip (<>)) [(v, [n.nodeId]) | n <- p.programNodes, v <- opValues n.nodeOp]
  outputValues = Set.fromList (map (.outputValue) p.programOutputs)

  opOf n = maybe (Input (SourceId 0)) (.nodeOp) (Map.lookup n nodes)
  typeOf v = maybe TF32 (.valueType) (programValue p v)
  outsOf n = maybe [] (.nodeOutputs) (Map.lookup n nodes)

  -- Stores into rings without room for a step's rows of one sequence
  -- past what the rows of an attention later in the program read.
  staged :: Set NodeId
  staged =
    Set.fromList
      [ n.nodeId
      | (i, n) <- zip [0 :: Int ..] p.programNodes
      , KvStore (StoreAt c _ _) _ <- [n.nodeOp]
      , Just Source {sourceKind = SCache (Ring _ _ len _) _} <- [programSource p c]
      , or
          [ opts.optSequenceRows > ringRoom len a.attnWindow (positionBound pos)
          | r <- drop (i + 1) p.programNodes
          , Attend a _ (AttendTo k v pos _) <- [r.nodeOp]
          , c `elem` [k, v]
          ]
      ]
  positionBound s = case programSource p s of
    Just Source {sourceKind = SRowInts _ bound} -> bound
    _ -> 0

  -- A ring's staged store, and the last attention that reads the ring.
  stagedInto c = listToMaybe [n.nodeId | n <- p.programNodes, KvStore (StoreAt c' _ _) _ <- [n.nodeOp], c' == c, n.nodeId `Set.member` staged]
  lastReader c = listToMaybe (reverse [n.nodeId | n <- p.programNodes, Attend _ _ (AttendTo k v _ _) <- [n.nodeOp], c `elem` [k, v]])

  -- The cost of a virtual value's chain: the dearest node in it.
  availOf :: ValueId -> State PS Avail
  availOf v = gets (fromMaybe (AStored (-1)) . Map.lookup v . (.avail))

  setAvail :: ValueId -> Avail -> State PS ()
  setAvail v a = modify' (\s -> s {avail = Map.insert v a s.avail})

  openIndex :: State PS (Maybe Int)
  openIndex = gets (fmap (.sbIndex) . (.open))

  ------------------------------------------------------------------------
  -- Steps

  closeOpen :: State PS ()
  closeOpen =
    gets (.open) >>= \case
      Nothing -> pure ()
      Just sb -> do
        outputsHere
        storeShared
        sb1 <- gets (fromMaybe sb . (.open))
        -- Tiles that anything outside the step reads, or the engine, are stored.
        let inStep = Set.fromList (map workNode (toList sb1.sbWork))
            tiled = [v | w <- toList sb1.sbWork, isTiling w, v <- outsOf (workNode w)]
            isTiling = \case
              Anchor _ -> True
              Epilogue _ -> True
              _ -> False
        stores <- fmap concat . traverse (storeTile sb1 inStep) $ tiled
        let sb' = sb1 {sbStores = sb1.sbStores <> Seq.fromList stores}
        modify' (\s -> s {closed = s.closed Seq.|> sb', open = Nothing})

  storeTile :: StepB -> Set NodeId -> ValueId -> State PS [Stored]
  storeTile sb inStep v = do
    let outside = any (`Set.notMember` inStep) (Map.findWithDefault [] v consumers)
        wanted = outside || v `Set.member` outputValues
    if not wanted
      then pure []
      else case Map.lookup v producer of
        Just n | Attend {} <- n.nodeOp -> do
          setAvail v (APartials sb.sbIndex)
          pure [Stored v Partials]
        _ -> do
          setAvail v (AStored sb.sbIndex)
          pure [Stored v (Elements (typeOf v))]

  newStep :: Maybe Family -> State PS StepB
  newStep fam = do
    closeOpen
    i <- gets (.next)
    let sb = StepB i Seq.empty Seq.empty fam Set.empty
    modify' (\s -> s {open = Just sb, next = i + 1})
    pure sb

  -- The open light step, or a new one; staged stores due land in it.
  lightStep :: State PS ()
  lightStep = do
    gets (.open) >>= \case
      Just sb | null sb.sbFamily, not (any isAnchor sb.sbWork) -> pure ()
      _ -> void (newStep Nothing)
    i <- openIndex
    due <- gets (\s -> [l | l@(after, _, _) <- s.landing, maybe False (> after) i])
    forM_ due $ \(_, n, c) -> do
      addWork (Land n)
      modify' $ \s -> case s.open of
        Just sb -> s {open = Just sb {sbWrites = Set.insert c sb.sbWrites}}
        Nothing -> s
    modify' (\s -> s {landing = filter (`notElem` due) s.landing})
   where
    isAnchor = \case
      Anchor _ -> True
      _ -> False

  -- Staged stores still to land do, after the last step.
  landRest :: State PS ()
  landRest = gets (null . (.landing)) >>= \done -> unless done lightStep

  addWork :: Work -> State PS ()
  addWork w = modify' $ \s -> case s.open of
    Just sb
      | workNode w `notElem` map workNode (toList sb.sbWork) -> s {open = Just sb {sbWork = sb.sbWork Seq.|> w}}
    _ -> s

  addStore :: Stored -> State PS ()
  addStore st = modify' $ \s -> case s.open of
    Just sb | st `notElem` sb.sbStores -> s {open = Just sb {sbStores = sb.sbStores Seq.|> st}}
    _ -> s

  ------------------------------------------------------------------------
  -- Chains: what the open step recomputes to have a value

  {- The work that makes a value in the open step, from stored values:
  recomputed nodes and combines, in program order. A value stored by the
  open step itself is recomputed there, like any virtual one. -}
  chain :: ValueId -> State PS ()
  chain v = do
    a <- availOf v
    i <- openIndex
    case (a, Map.lookup v producer) of
      (AStored s, _) | Just s /= i -> pure ()
      (APartials s, Just n) | Just s /= i -> addWork (Combine n.nodeId)
      (_, Just n) -> do
        -- A node in the step's work has its chain there already.
        planned <- gets (maybe False (elem n.nodeId . fmap workNode . (.sbWork)) . (.open))
        unless planned $ do
          mapM_ chain (opValues n.nodeOp)
          addWork (Recompute n.nodeId)
      (_, Nothing) -> pure ()

  {- When a light step that combines attention's output ends: the program's
  outputs made only from what it has, its combines' outputs and values
  stored before it, are made and stored by it, so that no step after it
  is needed for them. -}
  outputsHere :: State PS ()
  outputsHere = do
    sb <- gets (.open)
    forM_ sb $ \st -> do
      let combined = Set.fromList [n | Combine n <- toList st.sbWork]
          has v =
            availOf v >>= \case
              AStored _ -> pure True
              APartials _ -> pure (maybe False ((`Set.member` combined) . (.nodeId)) (Map.lookup v producer))
              AVirtual _ | Just n <- Map.lookup v producer -> and <$> traverse has (opValues n.nodeOp)
              _ -> pure False
      when (null st.sbFamily && not (Set.null combined)) $
        forM_ p.programOutputs $ \o ->
          availOf o.outputValue >>= \case
            AVirtual _ -> has o.outputValue >>= \ok -> when ok (materialize o.outputValue)
            _ -> pure ()

  {- When a step ends: the values it recomputed that took a pass, and
  attention's outputs it combined, that later steps need too are stored by
  it, once, so that no later step recomputes them or combines again. -}
  storeShared :: State PS ()
  storeShared = do
    sb <- gets (.open)
    forM_ sb $ \st -> do
      let inStep = Set.fromList (map workNode (toList st.sbWork))
          needed v = any (`Set.notMember` inStep) (Map.findWithDefault [] v consumers) || v `Set.member` outputValues
          store v = do
            addStore (Stored v (Elements (typeOf v)))
            setAvail v (AStored st.sbIndex)
      forM_ [n | Recompute n <- toList st.sbWork] $ \n ->
        forM_ (outsOf n) $ \v -> do
          a <- availOf v
          case a of
            AVirtual c | c >= Pass, needed v -> store v
            _ -> pure ()
      forM_ [n | Combine n <- toList st.sbWork] $ \n ->
        forM_ (outsOf n) $ \v -> do
          a <- availOf v
          case a of
            APartials _ | needed v -> store v
            _ -> pure ()

  -- A light step that stores a value, recomputing it.
  materialize :: ValueId -> State PS ()
  materialize v = do
    lightStep
    chain v
    i <- openIndex
    forM_ i $ \s -> do
      addStore (Stored v (Elements (typeOf v)))
      setAvail v (AStored s)

  ------------------------------------------------------------------------
  -- Nodes

  node :: Node -> State PS ()
  node n = case n.nodeOp of
    Input _ -> mapM_ (`setAvail` AStored (-1)) n.nodeOutputs
    op
      | Just fam <- anchorFamily op -> anchor n fam
      | isEffect op -> effect n
      | otherwise -> light n

  anchor :: Node -> Family -> State PS ()
  anchor n fam = do
    let ins = opValues n.nodeOp
    -- An anchor that needs the open step's tiles waits for the step to end.
    tiled <- or <$> traverse touchesOpen ins
    when tiled closeOpen
    -- Inputs too dear to recompute, and gated routes, are stored first;
    -- and GELUs, if asked.
    forM_ ins $ \v -> do
      dear <- expensive v
      gelu <- storedGelu v
      when (dear || gelu) (materialize v)
    case n.nodeOp of
      LinearRouted _ ids _ | opts.optGated -> do
        a <- availOf ids
        unless (stored a) (materialize ids)
      _ -> pure ()
    st <- get
    let readsCaches = [c | c <- opSources n.nodeOp, Just sb <- [st.open], c `Set.member` sb.sbWrites]
        gatedHere = case n.nodeOp of
          LinearRouted {} -> opts.optGated
          _ -> False
        joins = case st.open of
          Just sb -> opts.optHorizontal && sb.sbFamily == Just fam && null readsCaches && not gatedHere
          Nothing -> False
    unless joins (void (newStep (Just fam)))
    mapM_ chain ins
    addWork (Anchor n.nodeId)
    i <- openIndex
    forM_ i $ \s -> mapM_ (`setAvail` tile s) n.nodeOutputs
    -- The staged stores into the rings it is the last to read land after it.
    forM_ i $ \s ->
      forM_ [(store, c) | Attend _ _ (AttendTo k v _ _) <- [n.nodeOp], c <- nub [k, v], lastReader c == Just n.nodeId, Just store <- [stagedInto c]] $ \(store, c) ->
        modify' (\ps -> ps {landing = ps.landing <> [(s, store, c)]})
   where
    -- Attention's tile holds its splits' partials, which nothing but its
    -- combine reads: its outputs are partials from the start.
    tile s = case n.nodeOp of
      Attend {} -> APartials s
      _ -> ATile s n.nodeId
    stored = \case
      AStored _ -> True
      _ -> False

  -- Whether a value needs a tile of the open step, directly or through its chain.
  touchesOpen :: ValueId -> State PS Bool
  touchesOpen v0 = do
    i <- openIndex
    let go :: Set ValueId -> [ValueId] -> State PS Bool
        go _ [] = pure False
        go seen (v : vs)
          | v `Set.member` seen = go seen vs
          | otherwise =
              availOf v >>= \case
                ATile s _ | Just s == i -> pure True
                APartials s | Just s == i -> pure True
                AVirtual _ | Just n <- Map.lookup v producer -> go (Set.insert v seen) (opValues n.nodeOp <> vs)
                _ -> go (Set.insert v seen) vs
    go Set.empty [v0]

  -- Whether a value is a GELU to store before the anchor that reads it.
  storedGelu :: ValueId -> State PS Bool
  storedGelu v
    | opts.optStoreGelus
    , Just n <- Map.lookup v producer
    , isGelu n.nodeOp =
        availOf v >>= \case
          AVirtual _ -> pure True
          _ -> pure False
    | otherwise = pure False
   where
    isGelu = \case
      GeluMul {} -> True
      GatedGelu {} -> True
      _ -> False

  -- Whether recomputing a value would repeat attention's combine.
  expensive :: ValueId -> State PS Bool
  expensive v =
    availOf v >>= \case
      APartials _ -> pure True
      AVirtual c -> pure (c >= Expensive)
      _ -> pure False

  light :: Node -> State PS ()
  light n = do
    let op = n.nodeOp
        ins = opValues op
    avails <- traverse availOf ins
    i <- openIndex
    let onTile = [(v, a) | (v, ATile s a) <- zip ins avails, Just s == i]
        anchors = nub (map snd onTile)
    -- An epilogue is elementwise on one anchor's tile; its other operands
    -- come from stored values.
    othersOk <- and <$> traverse other [(v, a) | (v, a) <- zip ins avails, v `notElem` map fst onTile]
    if length anchors == 1 && pointwise op && othersOk
      then do
        mapM_ chain [v | (v, a) <- zip ins avails, isVirtual a]
        addWork (Epilogue n.nodeId)
        forM_ ((,) <$> i <*> listToMaybe anchors) $ \(s, a) -> mapM_ (`setAvail` ATile s a) n.nodeOutputs
      else do
        costs <- traverse inputCost ins
        let c = maximum (nodeCost p op : costs)
        mapM_ (`setAvail` AVirtual c) n.nodeOutputs
        -- Gated, a route is stored as soon as it can be, so that the engine
        -- fetches its experts while the steps before them run; and so is a
        -- route the engine reads, which the program outputs.
        when (opts.optGated && (feedsRouted n || output n)) $ do
          early <- or <$> traverse touchesOpen ins
          when early closeOpen
          mapM_ materialize (take 1 n.nodeOutputs)
   where
    feedsRouted r =
      or
        [ ids == v
        | Route {} <- [r.nodeOp]
        , v <- take 1 r.nodeOutputs
        , c <- Map.findWithDefault [] v consumers
        , LinearRouted _ ids _ <- [opOf c]
        ]
    output r =
      or
        [ v `Set.member` outputValues
        | Route {} <- [r.nodeOp]
        , v <- take 1 r.nodeOutputs
        ]
    isVirtual = \case
      AVirtual _ -> True
      _ -> False
    other (v, a) = case a of
      AStored _ -> pure True
      AVirtual Cheap -> not <$> touchesOpen v
      _ -> pure False

  -- What depending on a value adds to a chain's cost.
  inputCost :: ValueId -> State PS Cost
  inputCost v =
    availOf v >>= \case
      AVirtual c -> pure c
      APartials _ -> pure Expensive
      _ -> pure Cheap

  effect :: Node -> State PS ()
  effect n
    | n.nodeId `Set.member` staged = do
        lightStep
        mapM_ chain ins
        addWork (Stage n.nodeId)
        sequence_ [addStore (Stored x (Staged c)) | KvStore (StoreAt c _ _) x <- [n.nodeOp]]
    | otherwise = do
        avails <- traverse availOf ins
        i <- openIndex
        let onTile = [v | (v, ATile s _) <- zip ins avails, Just s == i]
        if not (null onTile) && pointwise n.nodeOp && length onTile == length ins
          then addWork (Epilogue n.nodeId)
          else do
            lightStep
            mapM_ chain ins
            addWork (Effect n.nodeId)
        modify' $ \s -> case s.open of
          Just sb -> s {open = Just sb {sbWrites = sb.sbWrites <> Set.fromList [c | c@(SourceId _) <- take 1 (opSources n.nodeOp)]}}
          Nothing -> s
   where
    ins = opValues n.nodeOp

  -- The engine's values: whatever isn't stored yet is, at the end.
  outputs :: State PS ()
  outputs = forM_ p.programOutputs $ \o -> do
    a <- availOf o.outputValue
    case a of
      AVirtual _ -> materialize o.outputValue
      APartials _ -> materialize o.outputValue
      _ -> pure ()

  finish :: StepB -> Step
  finish sb =
    Step
      { stepIndex = sb.sbIndex
      , stepName = name sb
      , stepWork = sortOn (nodeOrder . workNode) (toList sb.sbWork)
      , stepStores = sortOn (valueOrder . (.storedValue)) (toList sb.sbStores)
      , stepReads = sortOn (valueOrder . (.storedValue)) (readsOf sb)
      , stepCaches = nub [c | w <- toList sb.sbWork, c <- caches w]
      }

  nodeOrder (NodeId k) = k
  valueOrder (ValueId k) = k

  -- A stage holds its values apart from the cache, and a combine reads only
  -- attention's partials.
  caches = \case
    Stage _ -> []
    Combine _ -> []
    w -> case opOf (workNode w) of
      KvStore (StoreAt c _ _) _ -> [c]
      Attend _ _ (AttendTo k v _ _) -> [k, v]
      _ -> []

  -- Values stored by earlier steps that the step's work reads.
  readsOf :: StepB -> [Stored]
  readsOf sb =
    nub
      ( [ st
        | w <- toList sb.sbWork
        , v <- readsOfWork w
        , st <- storedBefore sb.sbIndex v
        , wanted w st.storedForm
        ]
          <> [st | w <- toList sb.sbWork, c <- stagesRead w, st <- stagedBefore sb.sbIndex c]
      )
   where
    -- A combine reads partials; everything else reads elements.
    wanted w form = case (w, form) of
      (Combine _, Partials) -> True
      (Combine _, _) -> False
      (_, Partials) -> False
      (_, Staged _) -> False
      _ -> True
    -- Attention reads the stages of the rings it reads, a land its own.
    stagesRead = \case
      Anchor n | Attend _ _ (AttendTo k v _ _) <- opOf n -> nub [k, v]
      Land n | KvStore (StoreAt c _ _) _ <- opOf n -> [c]
      _ -> []
    inStep = Set.fromList [v | w <- toList sb.sbWork, isMaking w, v <- outsOf (workNode w)]
    isMaking = \case
      Combine _ -> True
      Anchor _ -> True
      Recompute _ -> True
      Epilogue _ -> True
      Effect _ -> False
      Stage _ -> False
      Land _ -> False
    readsOfWork = \case
      Combine n -> outsOf n
      Land _ -> []
      w -> [v | v <- opValues (opOf (workNode w)), v `Set.notMember` inStep]

  storedBefore :: Int -> ValueId -> [Stored]
  storedBefore s v =
    [ st
    | sb <- toList final.closed
    , sb.sbIndex < s
    , st <- toList sb.sbStores
    , st.storedValue == v
    ]
      <> [Stored v (Elements (typeOf v)) | isInput v]

  isInput v = case Map.lookup v producer of
    Just Node {nodeOp = Input _} -> True
    _ -> False

  stagedBefore :: Int -> SourceId -> [Stored]
  stagedBefore s c =
    [ st
    | sb <- toList final.closed
    , sb.sbIndex < s
    , st <- toList sb.sbStores
    , st.storedForm == Staged c
    ]

  name :: StepB -> Name
  name sb =
    "s"
      <> T.pack (show sb.sbIndex)
      <> "_"
      <> case [opOf n | Anchor n <- toList sb.sbWork] of
        [] -> case mapMaybe effectOrStore (toList sb.sbWork) of
          (w : _) -> w
          []
            | not (null [() | Land _ <- toList sb.sbWork]) -> "land"
            | otherwise -> "light"
        anchors -> T.intercalate "_" (nub (map familyWord anchors))
   where
    effectOrStore = \case
      Effect _ -> Just "store"
      Stage _ -> Just "store"
      Combine _ -> Just "combine"
      Recompute _ -> case [st | st <- toList sb.sbStores, Elements _ <- [st.storedForm]] of
        st : _ -> (.valueName) <$> programValue p st.storedValue
        [] -> Nothing
      _ -> Nothing
    familyWord = \case
      Attend {} -> "attend"
      _ -> "gemv"

-- | A deterministic text form, for golden tests and reading.
renderPlan :: Plan -> Text
renderPlan pl =
  T.unlines $
    ["plan " <> pl.planProgram.programName]
      <> concatMap step pl.planSteps
 where
  p = pl.planProgram
  step s =
    ["  step " <> s.stepName]
      <> ["    " <> work w | w <- s.stepWork]
      <> ["    reads " <> stored st | st <- s.stepReads]
      <> ["    stores " <> stored st | st <- s.stepStores]
  work w =
    let n = workNode w
        label = maybe "?" (.nodeName) (programNode p n)
     in case w of
          Anchor _ -> "anchor " <> label
          Recompute _ -> "recompute " <> label
          Epilogue _ -> "epilogue " <> label
          Combine _ -> "combine " <> label
          Effect _ -> "effect " <> label
          Stage _ -> "stage " <> label
          Land _ -> "land " <> label
  stored st =
    maybe "?" (.valueName) (programValue p st.storedValue) <> case st.storedForm of
      Elements t -> " " <> scalarTypeName t
      Partials -> " partials"
      Staged c -> " staged for " <> maybe "?" (.sourceName) (programSource p c)
