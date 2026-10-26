-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | A plan's dependencies, checked apart from the planner, by its rules:

* a step reads the program's inputs, the elements earlier steps stored,
  and what its own recomputations and combines make; an anchor's outputs
  are on its tile, which only the step's epilogues read;
* an epilogue is elementwise, on one linear layer's tile: attention's tile
  holds partials;
* a combine reads the partials an earlier step stored of its attention, in
  a step of no anchor;
* a step stores partials only of its own attention, and elements, of the
  type the program gives them, only of what it computes or a linear
  layer's tile holds;
* the reads a step lists are what its work reads;
* a step doesn't store into a cache its attention reads, nor into one
  cache twice; a store into a cache that the program makes before an
  attention reading it lands in an earlier step, one it makes after, in a
  later one;
* every anchor and effect is placed once, and every output is stored;
* gated, a routed layer's route is stored by an earlier light step.
-}
module Test.FuryKernel.PlanCheck (
  planIssues,
) where

import Data.List (nub)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (isNothing)
import Data.Set (Set)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as T
import FuryKernel.Program

-- | What is wrong with a plan the planner made with these options.
planIssues :: PlanOptions -> Plan -> [Text]
planIssues opts pl = stepIssues <> cacheIssues <> placementIssues <> outputIssues <> gatedIssues
 where
  p = pl.planProgram
  steps = pl.planSteps

  opOf n = maybe (Input (SourceId 0)) (.nodeOp) (programNode p n)
  outsOf n = maybe [] (.nodeOutputs) (programNode p n)
  nodeName n = maybe "?" (.nodeName) (programNode p n)
  valueName v = maybe "?" (.valueName) (programValue p v)
  isInput v = case producerOf p v of
    Just Node {nodeOp = Input _} -> True
    _ -> False
  inputs = Set.fromList [v.valueId | v <- p.programValues, isInput v.valueId]

  -- Walk the steps in order, each against what the steps before it stored.
  (stepIssues, placed, storedBy) = go inputs Set.empty Map.empty Map.empty steps
  go _ _ seen by [] = ([], seen, by)
  go elems partials seen by (s : rest) =
    let found = stepCheck elems partials s
        elems' = elems <> Set.fromList [st.storedValue | st <- s.stepStores, st.storedForm /= Partials]
        partials' = partials <> Set.fromList [st.storedValue | st <- s.stepStores, st.storedForm == Partials]
        seen' = Map.unionWith (<>) seen (Map.fromListWith (<>) [(workNode x, [s.stepIndex]) | x <- s.stepWork, places x])
        by' = Map.unionWith (<>) by (Map.fromList [(st.storedValue, [s.stepIndex]) | st <- s.stepStores, st.storedForm /= Partials])
        (more, seenAll, byAll) = go elems' partials' seen' by' rest
     in (map (\i -> s.stepName <> ": " <> i) found <> more, seenAll, byAll)

  -- Work that places its node: an anchor, an effect, or a store into a
  -- cache on a tile.
  places = \case
    Anchor _ -> True
    Effect _ -> True
    Epilogue n -> isStore (opOf n)
    _ -> False

  stepCheck :: Set ValueId -> Set ValueId -> Step -> [Text]
  stepCheck elems partials s = reverse w.wIssues <> storeIssues <> listIssues <> cacheStepIssues
   where
    w = foldl work (W [] Set.empty Map.empty Set.empty Set.empty) s.stepWork
    work st = \case
      Recompute n -> computes n (readAll ("recompute " <> nodeName n) st (opValues (opOf n)))
      Combine n
        | isAttend (opOf n) ->
            let outs = outsOf n
                st1 =
                  if all (`Set.member` partials) outs
                    then st {wReadP = st.wReadP <> Set.fromList outs}
                    else issue st ("combine " <> nodeName n <> " of partials no earlier step stored")
                st2 = if null (stepAnchors s) then st1 else issue st1 ("combine " <> nodeName n <> " in a step of anchors")
             in computes n st2
        | otherwise -> issue st ("combine of " <> nodeName n <> ", which isn't attention")
      Anchor n -> onTile n n (readAll ("anchor " <> nodeName n) st (opValues (opOf n)))
      Epilogue n ->
        let ins = opValues (opOf n)
            tiled = [(v, a) | v <- ins, Just a <- [Map.lookup v st.wTile]]
            st1 = if elementwise (opOf n) then st else issue st ("epilogue " <> nodeName n <> " isn't elementwise")
            st2 = readAll ("epilogue " <> nodeName n) st1 [v | v <- ins, v `notElem` map fst tiled]
         in case nub (map snd tiled) of
              [a]
                | isLinear (opOf a) -> onTile a n st2
                | otherwise -> onTile a n (issue st2 ("epilogue " <> nodeName n <> " on the tile of " <> nodeName a <> ", which holds partials"))
              [] -> issue st2 ("epilogue " <> nodeName n <> " on no tile")
              _ -> issue st2 ("epilogue " <> nodeName n <> " on several tiles")
      Effect n -> readAll ("effect " <> nodeName n) st (opValues (opOf n))
    readAll what = foldl (readOne what)
    readOne what st v
      | v `Set.member` st.wLocal = st
      | v `Set.member` elems = st {wReadE = Set.insert v st.wReadE}
      | Just a <- Map.lookup v st.wTile = issue st (what <> " reads " <> valueName v <> ", on the tile of " <> nodeName a)
      | otherwise = issue st (what <> " reads " <> valueName v <> ", which no earlier step stored and the step doesn't compute")
    computes n st = st {wLocal = st.wLocal <> Set.fromList (outsOf n)}
    onTile a n st = st {wTile = foldr (`Map.insert` a) st.wTile (outsOf n)}
    issue st m = st {wIssues = m : st.wIssues}

    storeIssues = concatMap stored s.stepStores
    stored st = case st.storedForm of
      Partials -> case Map.lookup st.storedValue w.wTile of
        Just a | isAttend (opOf a) -> []
        _ -> ["stores " <> valueName st.storedValue <> " as partials, which aren't its attention's"]
      Elements t ->
        ["stores " <> valueName st.storedValue <> " as another type" | Just t /= fmap (.valueType) (programValue p st.storedValue)]
          <> case Map.lookup st.storedValue w.wTile of
            _ | st.storedValue `Set.member` w.wLocal -> []
            Just a
              | isLinear (opOf a) -> []
              | otherwise -> ["stores " <> valueName st.storedValue <> " as elements, from the partials of " <> nodeName a]
            Nothing -> ["stores " <> valueName st.storedValue <> ", which it doesn't compute"]

    listed form = Set.fromList [st.storedValue | st <- s.stepReads, form st.storedForm]
    listIssues =
      [ "lists the reads "
          <> names (listed (/= Partials))
          <> " and partials "
          <> names (listed (== Partials))
          <> ", but its work reads "
          <> names w.wReadE
          <> " and partials "
          <> names w.wReadP
      | listed (/= Partials) /= w.wReadE || listed (== Partials) /= w.wReadP
      ]
    names = T.intercalate ", " . map valueName . Set.toList

    stores = [c | x <- s.stepWork, places x, c <- written (opOf (workNode x))]
    attends = [c | Anchor n <- s.stepWork, c <- attended (opOf n)]
    cacheStepIssues =
      ["stores into a cache its attention reads" | any (`elem` attends) stores]
        <> ["stores into one cache twice" | length stores /= length (nub stores)]

  -- Where the program puts a node, and the one step a node is placed in.
  order = Map.fromList (zip (map (.nodeId) p.programNodes) [0 :: Int ..])
  placedOnce n = case Map.lookup n placed of
    Just [s] -> Just s
    _ -> Nothing

  cacheIssues =
    [ nodeName k.nodeId <> " and " <> nodeName a.nodeId <> ": " <> why
    | a <- p.programNodes
    , isAttend a.nodeOp
    , k <- p.programNodes
    , any (`elem` attended a.nodeOp) (written k.nodeOp)
    , Just sa <- [placedOnce a.nodeId]
    , Just sk <- [placedOnce k.nodeId]
    , Just why <-
        [ if Map.lookup k.nodeId order < Map.lookup a.nodeId order
            then if sk < sa then Nothing else Just "a store before attention in the program, not in an earlier step"
            else if sk > sa then Nothing else Just "a store after attention in the program, not in a later step"
        ]
    ]

  placementIssues =
    [ nodeName n.nodeId <> ": placed " <> T.pack (show (length (Map.findWithDefault [] n.nodeId placed))) <> " times"
    | n <- p.programNodes
    , isLinear n.nodeOp || isAttend n.nodeOp || isStore n.nodeOp
    , isNothing (placedOnce n.nodeId)
    ]

  outputIssues =
    [ "output " <> o.outputName <> " is never stored"
    | o <- p.programOutputs
    , not (isInput o.outputValue)
    , Map.notMember o.outputValue storedBy
    ]

  gatedIssues
    | not opts.optGated = []
    | otherwise =
        [ "gated: " <> nodeName n <> " reads its route " <> valueName ids <> ", which no earlier light step stored"
        | s <- steps
        , n <- stepAnchors s
        , LinearRouted _ ids _ <- [opOf n]
        , not (isInput ids)
        , not (any (\i -> i < s.stepIndex && i `Set.member` lightSteps) (Map.findWithDefault [] ids storedBy))
        ]
  lightSteps = Set.fromList [s.stepIndex | s <- steps, isLightStep s]

-- | What a step's work has seen so far.
data W = W
  { wIssues :: [Text]
  , wLocal :: Set ValueId
  , wTile :: Map ValueId NodeId
  , wReadE :: Set ValueId
  , wReadP :: Set ValueId
  }

isLinear :: Op -> Bool
isLinear = \case
  Linear {} -> True
  LinearRouted {} -> True
  _ -> False

isAttend :: Op -> Bool
isAttend = \case
  Attend {} -> True
  _ -> False

isStore :: Op -> Bool
isStore = \case
  KvStore {} -> True
  _ -> False

{- | Operations whose every output element needs only the same element of
their inputs, which a tile has whole.
-}
elementwise :: Op -> Bool
elementwise = \case
  Add {} -> True
  ScaleBy {} -> True
  GeluMul {} -> True
  Softcap {} -> True
  KvStore {} -> True
  _ -> False

-- | The caches an attention reads, and the one a store writes.
attended, written :: Op -> [SourceId]
attended = \case
  Attend _ _ (AttendTo k v _ _) -> [k, v]
  _ -> []
written = \case
  KvStore (StoreAt c _ _) _ -> [c]
  _ -> []
