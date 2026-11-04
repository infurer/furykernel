-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The hazard check: no two threads touch one element of memory unordered
while one of them stores to it.

* Threadgroup memory: a barrier orders what every thread did before it
  against what any thread does after it, and a SIMD-group barrier does so
  among its group's lanes; nothing else orders two threads, lanes of one
  SIMD group included. Two accesses to one element of a threadgroup array
  by two threads between the same barriers race, unless both load. A
  fragment operation is its SIMD group's lanes together, any of which may
  touch any of its elements.
* Device memory: nothing orders the threads of a launch, so no thread
  loads or stores an element that another one stores. Of a buffer the
  kernel declares apart, the threadgroups' elements are disjoint by the
  launch's data, and only the threads of one threadgroup are checked.
* Through a table the kernel writes, an element is an entry and an index:
  where the table declares its entries distinct, two accesses meet only at
  one entry and one index; where it doesn't, any two may meet. A table
  declared apart is checked within each threadgroup only, as a buffer is.

Every pair of accesses that could race becomes a system of affine
constraints over both threads' positions, their loops' counters, the
dimensions and what the branches around the accesses require, and the pair
is safe when the system has no solution in the integers
("FuryKernel.Check.Affine"). Indices are taken as computed without
wrapping around, which the bounds check of a later stage verifies. What
the check cannot write as an affine form (a value loaded from memory, a
product of two unknowns) can be anything, so a race the check cannot rule
out is a finding; but what the two threads compute from values that are
the same in both is the same too, and a minimum or a maximum is one of its
operands.

The barriers before an access count those of its loops' passes. Where
the passes of a uniform loop hold different numbers of barriers, a pair
of accesses is either in one pass of it, where both threads see the same
barriers and the same iterations of the uniform loops inside, or in two
passes: then at least the barriers after the earlier access in its pass,
those before the later one in its and those of every pass between them
separate the two. In one pass, a local the divergence analysis finds
uniform holds the same in both threads. After a loop or a uniform branch
whose barriers vary, the number is at least each access's inside plus
those after it there.
-}
module FuryKernel.Check.Hazard (
  checkHazards,
) where

import Control.Monad (forM_, when)
import Control.Monad.Trans.State.Strict (State, evalState, gets, modify')
import Data.List (nub, partition)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe, isJust, isNothing, listToMaybe)
import Data.Set (Set)
import Data.Set qualified as Set
import Data.Text (Text)
import FuryKernel.Check.Affine
import FuryKernel.Finding (Finding (..), Rule (..))
import FuryKernel.IR hiding (Kernel (..))
import FuryKernel.IR qualified as IR
import FuryKernel.Target (TargetCaps (..))

-- | Which of the two threads of a pair a variable belongs to.
data Side = One | Two
  deriving stock (Eq, Ord, Show)

data Var
  = -- | A dimension, the same for every thread.
    VDim Name
  | -- | A dimension divided by what it is a multiple of.
    VDimUnit Name
  | -- | A u32 scalar parameter.
    VParam Name
  | -- | The thread's threadgroup, its SIMD group, its lane in it.
    VGroup Side
  | VSimd Side
  | VLane Side
  | -- | The iteration of a loop the thread is in, and the loop's iterations.
    VIter Side Int
  | VTrips Side Int
  | {- | Anything else: quotients, remainders, an element of a pack or a
    fragment, an unknown value.
    -}
    VAux Side Int
  | {- | An unknown value the same in every thread of a threadgroup, and one
    the same in every thread of the launch.
    -}
    VUniform Side Int
  | VLaunch Int
  | {- | A number of barriers, or of a SIMD group's, that what comes before
    doesn't fix: at a pass's start, after a loop, after a branch.
    -}
    VEpoch Side Int
  | {- | A dimension times the difference of what multiplies it in a pair's
    two indices.
    -}
    VProduct Name
  deriving stock (Eq, Ord, Show)

type Lin = Affine Var

type Con = Constraint Var

{- | A value: its affine form, the dimensions it multiplies (@Σ n · L_n@
over them, each with its multiplier), and what defines the variables it
adds.
-}
data Term = Term Lin (Map Name Lin) [Con]

linear :: Lin -> [Con] -> Term
linear l = Term l Map.empty

data Mem = Tg Name | Dev Name | Via Name
  deriving stock (Eq, Ord, Show)

data Touch = Touch
  { aMem :: Mem
  , aStore :: Bool
  , aWhat :: Text
  , aIndex :: Lin
  , aEntry :: Lin
  -- ^ The entry of a table it goes through; 0 for a buffer or an array.
  , aProducts :: Map Name Lin
  -- ^ The dimensions the index multiplies, and by what.
  , aCons :: [Con]
  , aEpoch :: Lin
  -- ^ The threadgroup barriers before it.
  , aSimdEpoch :: Lin
  -- ^ The barriers of the thread's SIMD group before it.
  , aWide :: Bool
  -- ^ A fragment operation: its SIMD group's lanes together.
  , aStmt :: Int
  , aDigits :: [Var]
  -- ^ The element of a fragment operation it is, for its pairs with itself.
  , aLoops :: [Int]
  , aSince :: Map Int Fewest
  -- ^ For each loop around it, the fewest barriers since its pass began.
  , aUntil :: Map Int Fewest
  -- ^ For each loop around it, the fewest barriers until its pass ends.
  }

-- | Barriers, and a SIMD group's barriers (its threadgroup's included).
data Fewest = Fewest Integer Integer
  deriving stock (Eq, Show)

instance Semigroup Fewest where
  Fewest a b <> Fewest c d = Fewest (a + c) (b + d)

instance Monoid Fewest where
  mempty = Fewest 0 0

lesser :: Fewest -> Fewest -> Fewest
lesser (Fewest a b) (Fewest c d) = Fewest (min a c) (min b d)

fewestOf :: Bool -> Fewest -> Integer
fewestOf simd (Fewest a b) = pick simd (a, b)

-- | Of a pair for barriers and for a SIMD group's, the one asked.
pick :: Bool -> (a, a) -> a
pick simd = if simd then snd else fst

data Loop = Loop
  { loopUniform :: Bool
  {- ^ Declared uniform: in one pass of the loops around it, the same
  iterations in every thread of a threadgroup.
  -}
  , loopKnown :: Bool
  -- ^ Uniform, of bounds the same in every pass of the loops around it.
  , loopLaunch :: Bool
  -- ^ The same iterations in every thread of the launch.
  , loopParent :: Maybe Int
  , loopFixed :: (Bool, Bool)
  {- ^ Whether every pass holds the same number of barriers, and of a
  SIMD group's.
  -}
  , loopLeast :: Fewest
  -- ^ The fewest barriers a pass holds.
  }

data Ctx = Ctx
  { cDefs :: Map Name Term
  , cGuards :: [Con]
  , cEpoch :: Lin
  , cSimdEpoch :: Lin
  , cLoops :: [Int]
  , cSince :: Map Int Fewest
  -- ^ For each loop around, the fewest barriers since its pass began.
  , cUntil :: Map Int Fewest
  {- ^ For each loop around, the fewest barriers from the statement's end
  to its pass's end.
  -}
  , cAhead :: Fewest
  -- ^ The fewest barriers of the statement itself, after what it reads.
  , cExits :: Map Int (Maybe Lin, Maybe Lin)
  {- ^ For each loop around whose passes' barriers, or a SIMD group's,
  vary: the number after it.
  -}
  }

data WS = WS
  { wsNext :: Int
  , wsAccesses :: [Touch]
  , wsLoops :: Map Int Loop
  , wsStmt :: Int
  , wsDivisions :: [Division]
  , wsQuotients :: Map (Lin, Integer) (Var, Var)
  -- ^ The quotient and the remainder of a value by a constant.
  , wsEpochs :: Map Int (Maybe Int, Bool)
  {- ^ Of each 'VEpoch': the loop whose pass it counts in, if any, and
  whether it counts a SIMD group's barriers.
  -}
  , wsInputs :: Map Int [Var]
  -- ^ Of each 'VAux' computed from values: the variables of those values.
  , wsChoices :: Map Var [[Con]]
  -- ^ Of each minimum and maximum: its value as either operand.
  }

-- | A quotient or a remainder by a value that isn't a constant.
data Division = Division
  { divDividend :: Lin
  , divDivisor :: Lin
  , divQuotient :: Bool
  , divResult :: Var
  }

type W = State WS

{- | The races of a kernel that passed every other rule, given the locals
the same in every thread of a threadgroup where they're read.
-}
checkHazards :: TargetCaps -> Set Name -> IR.Kernel -> [Finding]
checkHazards caps uniformLocals k = nub [finding a b | ((a, _), (b, _)) <- pairs, races a b]
 where
  (accesses, loops, divisions, epochs, inputs, choices) = evalState collect (WS 0 [] Map.empty 0 [] Map.empty Map.empty Map.empty Map.empty)
  collect = do
    _ <- block (Ctx Map.empty [] (constant 0) (constant 0) [] Map.empty Map.empty mempty Map.empty) k.kernelBody
    st <- gets id
    pure (reverse st.wsAccesses, st.wsLoops, st.wsDivisions, st.wsEpochs, st.wsInputs, st.wsChoices)
  -- What each computed unknown is a function of, down to what isn't one;
  -- an unknown is computed from earlier ones only.
  basis = foldl' (\m (n, from) -> Map.insert n (nub (concatMap (expand m) from)) m) Map.empty (Map.toAscList inputs)
   where
    expand m = \case
      VAux _ i | Just b <- Map.lookup i m -> b
      v -> [v]
  -- A quotient and a remainder of one value by one divisor the same in
  -- the launch: equal in two threads only where the value is.
  splits =
    [ (q.divResult, r.divResult, q.divDividend)
    | q <- divisions
    , q.divQuotient
    , r <- divisions
    , not r.divQuotient
    , q.divDividend == r.divDividend
    , q.divDivisor == r.divDivisor
    , all ((== Launch) . varLevel) (Map.keys (coefficients q.divDivisor))
    ]
  stored = Set.fromList [a.aMem | a <- accesses, a.aStore]
  -- Each access with the range of its index, where the constraints bound it.
  -- Through a table of entries that may share arrays, any index may meet.
  indexed = zip [0 :: Int ..] [(a, if Map.null a.aProducts && not (shared a.aMem) then range (bounds (a.aCons <> position One <> dimensions)) a.aIndex else Nothing) | a <- accesses]
  shared = \case
    Via n -> not (Set.member n distinctTables)
    _ -> False
  pairs =
    [ (x, y)
    | (i, x@(a, sa)) <- indexed
    , a.aMem `Set.member` stored
    , (j, y@(b, sb)) <- indexed
    , i <= j
    , a.aMem == b.aMem
    , a.aStore || b.aStore
    , overlap sa sb
    ]
  overlap x y = case (x, y) of
    (Just (lo, hi), Just (lo', hi')) -> lo <= hi' && lo' <= hi
    _ -> True

  width = toInteger caps.capSimdWidth
  threads = toInteger k.kernelThreads
  simdGroups = (threads + width - 1) `div` width
  dims = Map.fromList [(p.paramName, r) | p <- k.kernelParams, Dim r <- [p.paramRole]]
  scalars = Set.fromList [p.paramName | p <- k.kernelParams, Scalar <- [p.paramRole]]
  tgCounts = Map.fromList [(a.tgName, toInteger a.tgCount) | a <- k.kernelTgArrays]
  readOnly = Set.fromList [b.bufName | b <- k.kernelBuffers, b.bufAccess == ReadOnly]
  apartBuffers = Set.fromList [b.bufName | b <- k.kernelBuffers, b.bufApart]
  extents = Map.fromList [(b.bufName, b.bufExtent) | b <- k.kernelBuffers]
  apartTables = Set.fromList [t.tableName | t <- k.kernelTables, t.tableApart]
  distinctTables = Set.fromList [t.tableName | t <- k.kernelTables, t.tableDistinct]
  writableTables = Set.fromList [t.tableName | t <- k.kernelTables, t.tableAccess /= ReadOnly]
  tableSizes = Map.fromList [(t.tableName, (t.tableEntries, t.tableExtent)) | t <- k.kernelTables]

  ------------------------------------------------------------------------
  -- Pairs

  -- A pair whose indices never meet needs no look at the threads.
  races a b = any solvable (meeting a b) && any picked (cases a b)

  {- A system, and if it has solutions, with each minimum and maximum in it
  one of its operands, in each thread: those only take solutions away. -}
  picked (seeds, context) =
    let system = relevant seeds context
        here = nub [v | c <- system, v <- Map.keys (coefficients (body c)), Map.member (onOne v) choices]
        alternatives v = (if onOne v == v then id else map (map conOnTwo)) (Map.findWithDefault [[]] (onOne v) choices)
     in solvable system && any (\o -> solvable (relevant (seeds <> concat o) context)) (mapM alternatives here)
   where
    body = \case
      EqualsZero x -> x
      NonNegative x -> x

  meeting a b = [relevant m (common a b) | m <- meet a (onTwo b)]

  common a b =
    let b2 = onTwo b
     in a.aCons <> b2.aCons <> position One <> position Two <> dimensions <> (if Map.null a.aProducts then inMemory a else [])

  {- The two indices equal: their affine parts, and for each dimension
  that multiplies something, g times a variable for the dimension times
  d / g, where d is the difference of its multipliers and g their common
  factor, by d's sign: 0 for d = 0, at least the dimension for d ≥ g, at
  most its opposite for d ≤ -g. -}
  meet a b2
    | shared a.aMem = [[]]
    | otherwise = meetAt a b2
  -- Through a table of distinct entries, at one entry too.
  meetAt a b2 =
    let ns = nub (Map.keys a.aProducts <> Map.keys b2.aProducts)
        multiplier x n = Map.findWithDefault (constant 0) n x.aProducts
        difference n = minus (multiplier a n) (multiplier b2 n)
        -- The difference's common factor g, so that d = g·d': the
        -- product is g times the dimension times d'.
        factor n = let d = difference n in max 1 (foldr gcd (constantPart d) (Map.elems (coefficients d)))
        same = equal (foldl' plus a.aIndex [times (factor n) (term (VProduct n)) | n <- ns]) b2.aIndex
        signs n =
          let g = factor n
              d = difference n
              q = term (VProduct n)
              dn = term (VDim n)
           in [ [EqualsZero d, EqualsZero q]
              , [atMost (constant g) d, atMost dn q]
              , [atMost d (constant (negate g)), atMost q (times (-1) dn)]
              ]
        entry = [equal a.aEntry b2.aEntry | Via _ <- [a.aMem]]
     in [same : entry <> concat choice | choice <- mapM signs ns]

  cases a b =
    let a1 = a
        b2 = onTwo b
        (varsA, varsB) = (variablesOf a, variablesOf b)
        used = Set.fromList (varsA <> varsB)
        distinctOrSame (q, r, x) =
          distinct q <> distinct r <> [[equal x (rename onTwoVar x)]]
        facts = mapM distinctOrSame [t | t@(q, r, _) <- splits, Set.member q used || Set.member r used]
        base0 = [(m <> concat f, c) | c <- [common a b], m <- meet a1 b2, f <- facts]
        tripped = nub (tripsOf a <> tripsOf b)
        shared' launch =
          [ equal (term (VTrips One l)) (term (VTrips Two l))
          | l <- tripped
          , maybe False (if launch then (.loopLaunch) else (.loopKnown)) (Map.lookup l loops)
          ]
        -- The uniform loops around both whose passes' barriers vary,
        -- outermost first: the two threads in one pass of each, or in one
        -- of the outer ones and two of the next.
        around = takeWhile varies (map fst (takeWhile (uncurry (==)) (zip (reverse a.aLoops) (reverse b.aLoops))))
        varies l = maybe False (\lp -> lp.loopUniform && not (uncurry (&&) lp.loopFixed)) (Map.lookup l loops)
        passes = go [] around
         where
          go done = \case
            [] -> [(within False done <> [equal a1.aEpoch b2.aEpoch], Just (within True done <> [equal a1.aSimdEpoch b2.aSimdEpoch]))]
            l : rest -> [c | earlier <- [True, False], Just c <- [apart done l earlier]] <> go (done <> [l]) rest
        -- In one pass of each loop done: the same iterations, and the same
        -- numbers counted in it, of a SIMD group's barriers if asked.
        within simd done =
          let owners = Nothing : map Just done
              -- What's the same in both threads in such a pass: their
              -- threadgroup's, their SIMD group's if asked, the passes', and
              -- what's computed from those alone.
              equalAt s = \case
                VGroup _ -> True
                VSimd _ -> s
                VIter _ l -> l `elem` done
                VUniform _ _ -> True
                VDim _ -> True
                VDimUnit _ -> True
                VParam _ -> True
                VLaunch _ -> True
                _ -> False
              computed s m = maybe False (all (equalAt s)) (Map.lookup m basis)
           in concat
                [ [equal (term (VIter One l)) (term (VIter Two l)) | not simd, l <- done]
                , [equal (term (VAux One m)) (term (VAux Two m)) | m <- derivedIn, computed simd m, not simd || not (computed False m)]
                , [equal (term (VEpoch One e)) (term (VEpoch Two e)) | e <- counted, Just (owner, s) <- [Map.lookup e epochs], s == simd, owner `elem` owners]
                , [equal (term (VTrips One l)) (term (VTrips Two l)) | not simd, l <- tripped, Just lp <- [Map.lookup l loops], lp.loopUniform, lp.loopParent `elem` owners]
                ]
        counted = nub [e | VEpoch _ e <- varsA <> varsB]
        derivedIn = nub [m | VAux _ m <- varsA <> varsB, Map.member m basis]
        {- In one pass of each loop done and in two of l, the first thread's
        the earlier or the second's: apart by at least the barriers after the
        earlier access in its pass, before the later one in its, and of each
        pass between. Nothing where that can't be none. -}
        apart done l earlier = do
          lp <- Map.lookup l loops
          let (first, second) = if earlier then (a, b) else (b, a)
              (one, two) = (term (VIter One l), term (VIter Two l))
              distance = if earlier then minus two one else minus one two
              gap = Map.findWithDefault mempty l first.aUntil <> Map.findWithDefault mempty l second.aSince
              between simd
                | pick simd lp.loopFixed = Just [if simd then equal a1.aSimdEpoch b2.aSimdEpoch else equal a1.aEpoch b2.aEpoch]
                | fewestOf simd gap > 0 = Nothing
                | fewestOf simd lp.loopLeast > 0 = Just [atMost distance (constant 1)]
                | otherwise = Just []
          order <- between False
          pure (within False done <> [atMost (constant 1) distance] <> order, (within True done <>) <$> between True)
        lanes
          | a.aWide || b.aWide =
              if a.aStmt == b.aStmt then [c | v <- a.aDigits <> iters a, c <- distinct v] else [[]]
          | otherwise = distinct (VLane One)
        iters x = [VIter One l | l <- x.aLoops]
        distinct v = let w = onTwoVar v in [[atMost (plus (term v) (constant 1)) (term w)], [atMost (plus (term w) (constant 1)) (term v)]]
        sameGroup = [equal (term (VGroup One)) (term (VGroup Two))] <> [equal (term (VUniform One u)) (term (VUniform Two u)) | u <- nub (uniformsOf a <> uniformsOf b)]
        sameSimd = [equal (term (VSimd One)) (term (VSimd Two))]
     in case a.aMem of
          Tg _ ->
            concat
              [ [(base <> c, context) | c <- distinct (VSimd One)]
                  <> [(base <> sameSimd <> inSimd <> c, context) | Just inSimd <- [simdOrder], c <- lanes]
              | (seeds, context) <- base0
              , (order, simdOrder) <- passes
              , let base = seeds <> sameGroup <> shared' False <> order
              ]
          -- Two threadgroups touch a buffer or a table declared apart at
          -- disjoint elements, as the launch's data keeps.
          mem ->
            concat
              [ (if declaredApart mem then [] else [(base <> c, context) | c <- distinct (VGroup One)])
                  <> [(base <> sameGroup <> c, context) | c <- distinct (VSimd One)]
                  <> [(base <> sameGroup <> sameSimd <> c, context) | c <- lanes]
              | (seeds, context) <- base0
              , let base = seeds <> shared' True
              ]
  declaredApart = \case
    Dev n -> Set.member n apartBuffers
    Via n -> Set.member n apartTables
    Tg _ -> False

  position s =
    [ atMost (constant 0) (term (VLane s))
    , atMost (term (VLane s)) (constant (width - 1))
    , atMost (constant 0) (term (VSimd s))
    , atMost (term (VSimd s)) (constant (simdGroups - 1))
    , atMost (plus (times width (term (VSimd s))) (term (VLane s))) (constant (threads - 1))
    , atMost (constant 0) (term (VGroup s))
    ]
      <> [atMost (term (VGroup s)) (minus g (constant 1)) | Just g <- [launchTerm k.kernelGroups]]

  dimensions =
    concat
      [ [atMost (constant (toInteger r.dimMin)) (term (VDim n)), atMost (term (VDim n)) (constant (toInteger r.dimMax))]
          <> [equal (term (VDim n)) (times (toInteger r.dimMultipleOf) (term (VDimUnit n))) | r.dimMultipleOf > 1]
      | (n, r) <- Map.toList dims
      ]
      <> [atMost (constant 0) (term (VParam n)) | n <- Set.toList scalars]

  inMemory a = case a.aMem of
    Tg n -> [atMost (constant 0) i] <> [atMost i (constant (c - 1)) | Just c <- [Map.lookup n tgCounts]]
    Dev b -> [atMost (constant 0) i] <> [atMost i (minus e (constant 1)) | Just x <- [Map.lookup b extents], Just e <- [launchTerm x]]
    Via t ->
      [atMost (constant 0) i, atMost (constant 0) a.aEntry]
        <> concat
          [ [atMost i (minus x (constant 1)) | Just x <- [launchTerm extent]]
              <> [atMost a.aEntry (minus c (constant 1)) | Just c <- [launchTerm entries]]
          | Just (entries, extent) <- [Map.lookup t tableSizes]
          ]
   where
    i = a.aIndex

  -- A launch size over the dimensions: affine, or nothing known.
  launchTerm = \case
    ELit (LU32 n) -> Just (constant (toInteger n))
    EParam n | Map.member n dims -> Just (term (VDim n))
    EBin Add x y -> plus <$> launchTerm x <*> launchTerm y
    EBin Sub x y -> minus <$> launchTerm x <*> launchTerm y
    EBin Mul x y -> case (launchTerm x, launchTerm y) of
      (Just a, Just b)
        | Map.null (coefficients a) -> Just (times (constantPart a) b)
        | Map.null (coefficients b) -> Just (times (constantPart b) a)
      _ -> Nothing
    _ -> Nothing

  finding a b =
    Finding Hazards $ case a.aMem of
      Tg n -> "Threads may race on the threadgroup array " <> n <> ": " <> both <> " can touch one element between the same barriers."
      Dev n -> "Threads may race on the buffer " <> n <> ": " <> both <> " can touch one element in two threads of the launch."
      Via n -> "Threads may race through the table " <> n <> ": " <> both <> " can touch one element in two threads of the launch."
   where
    both = if a.aStmt == b.aStmt && a.aWhat == b.aWhat then a.aWhat <> " in two threads" else a.aWhat <> " and " <> b.aWhat

  ------------------------------------------------------------------------
  -- Walking the body

  -- Each statement with the fewest barriers of those after it.
  block :: Ctx -> [Stmt] -> W Ctx
  block ctx stmts = go ctx (zip stmts (drop 1 (scanr (\s after -> fewest [s] <> after) mempty stmts)))
   where
    go c = \case
      [] -> pure c
      (s, after) : rest -> do
        c' <- stmt c {cUntil = Map.map (<> after) ctx.cUntil} s
        go c' {cUntil = ctx.cUntil} rest

  stmt :: Ctx -> Stmt -> W Ctx
  stmt outer s = do
    n <- gets (.wsStmt)
    modify' (\st -> st {wsStmt = n + 1})
    let ctx = outer {cAhead = fewest [s]}
    case s of
      SLet v _ e -> do
        loads ctx n e
        t <- expr ctx e
        pure ctx {cDefs = Map.insert v t ctx.cDefs}
      SVar _ _ e -> loads ctx n e >> pure ctx
      SAssign _ e -> loads ctx n e >> pure ctx
      SStore b i e -> do
        mapM_ (loads ctx n) [i, e]
        t <- expr ctx i
        element ctx n (Dev b) True ("the store to " <> b <> "[" <> renderExpr i <> "]") 1 t
        pure ctx
      SStoreVia b e i v -> do
        mapM_ (loads ctx n) [e, i, v]
        te <- entryOf ctx e
        t <- expr ctx i
        elementVia ctx n b True ("the store through " <> b <> "[" <> renderExpr e <> "][" <> renderExpr i <> "]") 1 te t
        pure ctx
      STgStore a i e -> do
        mapM_ (loads ctx n) [i, e]
        t <- expr ctx i
        element ctx n (Tg a) True ("the store to " <> a <> "[" <> renderExpr i <> "]") 1 t
        pure ctx
      SLoadPack ns _ b i -> do
        loads ctx n i
        t <- expr ctx i
        element ctx n (Dev b) False ("the pack loaded from " <> b <> " at " <> renderExpr i) (toInteger (length ns)) t
        pure ctx
      SLoadPackVia ns _ b e i -> do
        mapM_ (loads ctx n) [e, i]
        when (Set.member b writableTables) $ do
          te <- entryOf ctx e
          t <- expr ctx i
          elementVia ctx n b False ("the pack loaded through " <> b <> "[" <> renderExpr e <> "] at " <> renderExpr i) (toInteger (length ns)) te t
        pure ctx
      SStorePack b i es -> do
        mapM_ (loads ctx n) (i : es)
        t <- expr ctx i
        element ctx n (Dev b) True ("the pack stored to " <> b <> " at " <> renderExpr i) (toInteger (length es)) t
        pure ctx
      STgStorePack a i es -> do
        mapM_ (loads ctx n) (i : es)
        t <- expr ctx i
        element ctx n (Tg a) True ("the pack stored to " <> a <> " at " <> renderExpr i) (toInteger (length es)) t
        pure ctx
      SIf _ c th el -> do
        loads ctx n c
        yes <- condition ctx c
        no <- negation ctx c
        -- Where the sides' barriers may differ, the number after the
        -- branch: at least each inner access's plus the barriers after it
        -- on its side, as after a loop.
        let side = negate (n + 1)
            varies simd = count simd th /= count simd el || isNothing (count simd th)
            joined simd = if varies simd then Just . term <$> freshEpoch (listToMaybe ctx.cLoops) simd else pure Nothing
        xb <- joined False
        xs <- joined True
        let inner
              | isJust xb || isJust xs = ctx {cUntil = Map.insert side mempty ctx.cUntil, cExits = Map.insert side (xb, xs) ctx.cExits}
              | otherwise = ctx
        after1 <- block inner {cGuards = ctx.cGuards <> fromMaybe [] yes} th
        after2 <- block inner {cGuards = ctx.cGuards <> fromMaybe [] no} el
        let least = lesser (fewest th) (fewest el)
        (epoch, ec) <- joinEpoch ctx False xb ctx.cEpoch after1.cEpoch after2.cEpoch least
        (simd, sc) <- joinEpoch ctx True xs ctx.cSimdEpoch after1.cSimdEpoch after2.cSimdEpoch least
        pure
          ctx
            { cDefs = Map.union after1.cDefs after2.cDefs
            , cGuards = ctx.cGuards <> ec <> sc
            , cEpoch = epoch
            , cSimdEpoch = simd
            , cSince = Map.unionWith lesser after1.cSince after2.cSince
            }
      SFor region v start end step body -> do
        mapM_ (loads ctx n) [start, end, step]
        l <- gets (Map.size . (.wsLoops))
        (s0, sc) <- flat =<< expr ctx start
        (e0, ec) <- flat =<< expr ctx end
        (st0, stc) <- flat =<< expr ctx step
        let iter = term (VIter One l)
            trips = term (VTrips One l)
            bounded = sc <> ec <> stc
            -- Where the loop holds barriers, its iterations count in the
            -- epochs: those its bounds give.
            tied = if count True body == Just 0 then [] else tie (ctx.cGuards <> bounded) start end step s0 e0 st0 trips
            limits = [atMost (constant 0) iter, atMost (plus iter (constant 1)) trips, atMost (constant 0) trips] <> tied <> bounded
            known = all (onlyLaunch False) [s0, e0, st0]
            perPass simd = count simd body
            fixed = (isJust (perPass False), isJust (perPass True))
            parent = listToMaybe ctx.cLoops
            total = fewest [s]
        modify' (\st -> st {wsLoops = Map.insert l (Loop (region == RUniform) (region == RUniform && known) (region == RUniform && all (onlyLaunch True) [s0, e0, st0]) parent fixed (fewest body)) st.wsLoops})
        (value, vc) <- case asConstant st0 of
          Just c | c > 0 -> pure (plus s0 (times c iter), [atMost (plus s0 (times c iter)) (minus e0 (constant 1))])
          _ -> do
            x <- freshVar
            pure (term x, [atMost s0 (term x), atMost (term x) (minus e0 (constant 1))])
        (inside, ic) <- epochAt (Just l) False ctx.cEpoch (perPass False) iter 0
        (insideSimd, isc) <- epochAt (Just l) True ctx.cSimdEpoch (perPass True) iter 0
        (afterEpoch, ac) <- epochAt parent False ctx.cEpoch (perPass False) trips (fewestOf False total)
        (afterSimd, asc) <- epochAt parent True ctx.cSimdEpoch (perPass True) trips (fewestOf True total)
        let exit simd x = if pick simd fixed then Nothing else Just x
            ctx' =
              ctx
                { cDefs = Map.insert v (linear value vc) ctx.cDefs
                , cGuards = ctx.cGuards <> limits <> vc <> ic <> isc
                , cEpoch = inside
                , cSimdEpoch = insideSimd
                , cLoops = l : ctx.cLoops
                , cSince = Map.insert l mempty ctx.cSince
                , cUntil = Map.insert l mempty ctx.cUntil
                , cExits = Map.insert l (exit False afterEpoch, exit True afterSimd) ctx.cExits
                }
        _ <- block ctx' body
        pure
          ctx
            { cGuards = ctx.cGuards <> [atMost (constant 0) trips] <> tied <> bounded <> ac <> asc
            , cEpoch = afterEpoch
            , cSimdEpoch = afterSimd
            , cSince = Map.map (<> total) ctx.cSince
            }
      SBarrier -> pure ctx {cEpoch = plus ctx.cEpoch (constant 1), cSimdEpoch = plus ctx.cSimdEpoch (constant 1), cSince = Map.map (<> Fewest 1 1) ctx.cSince}
      SSimdBarrier -> pure ctx {cSimdEpoch = plus ctx.cSimdEpoch (constant 1), cSince = Map.map (<> Fewest 0 1) ctx.cSince}
      SSimd _ _ _ e -> loads ctx n e >> pure ctx
      SFragVar _ _ -> pure ctx
      SFragMma {} -> pure ctx
      SFragConvert _ _ -> pure ctx
      SFragLoad _ m i st tr -> do
        mapM_ (loads ctx n) ([i, st] <> [e | FVia _ e <- [m]])
        fragment ctx n m False ("the fragment loaded from " <> memName m <> " at " <> renderExpr i) i st tr
        pure ctx
      SFragStore m i st _ -> do
        mapM_ (loads ctx n) ([i, st] <> [e | FVia _ e <- [m]])
        fragment ctx n m True ("the fragment stored to " <> memName m <> " at " <> renderExpr i) i st False
        pure ctx
      SFragScaleRows _ a i -> do
        loads ctx n i
        Term base bp bc <- expr ctx i
        r <- freshVar
        record
          ctx
          Touch
            { aMem = Tg a
            , aStore = False
            , aWhat = "the row scales read from " <> a <> " at " <> renderExpr i
            , aIndex = plus base (term r)
            , aEntry = constant 0
            , aProducts = bp
            , aCons = bc <> digit r 8
            , aEpoch = ctx.cEpoch
            , aSimdEpoch = ctx.cSimdEpoch
            , aWide = True
            , aStmt = n
            , aDigits = [r]
            , aLoops = ctx.cLoops
            , aSince = ctx.cSince
            , aUntil = Map.map (ctx.cAhead <>) ctx.cUntil
            }
        pure ctx

  memName = \case
    FDevice b -> b
    FThreadgroup a -> a
    FVia b e -> b <> "[" <> renderExpr e <> "]"
  memOf = \case
    FDevice b -> Dev b
    FThreadgroup a -> Tg a
    FVia b _ -> Via b

  -- An entry of a table: its affine form, with what defines it.
  entryOf ctx e = expr ctx e >>= flat

  -- 'element' through a table, at an entry.
  elementVia ctx n b store what width' entry = elementAt ctx n (Via b) entry store what width'

  -- An access to the elements of a lane's pack, or to one element.
  element ctx n m = elementAt ctx n m (constant 0, [])
  elementAt ctx n m (entry, ec) store what width' (Term i ip ic) = do
    p <- freshVar
    record
      ctx
      Touch
        { aMem = m
        , aStore = store
        , aWhat = what
        , aIndex = plus (times width' i) (term p)
        , aEntry = entry
        , aProducts = Map.map (times width') ip
        , aCons = ec <> ic <> digit p width'
        , aEpoch = ctx.cEpoch
        , aSimdEpoch = ctx.cSimdEpoch
        , aWide = False
        , aStmt = n
        , aDigits = []
        , aLoops = ctx.cLoops
        , aSince = ctx.cSince
        , aUntil = Map.map (ctx.cAhead <>) ctx.cUntil
        }

  {- A fragment's 8×8 elements: (r, c) at index + r·stride + c, or
  index + c·stride + r transposed. A stride that isn't a constant gets a
  row at a time, its multiple then affine. -}
  fragment ctx n m store what i st tr = do
    Term base bp bc <- expr ctx i
    (stride, sc) <- flat =<< expr ctx st
    (entry, ec) <- case m of
      FVia _ e -> entryOf ctx e
      _ -> pure (constant 0, [])
    r <- freshVar
    c <- freshVar
    let (along, across) = if tr then (c, r) else (r, c)
        one rowTerm extra =
          record
            ctx
            Touch
              { aMem = memOf m
              , aStore = store
              , aWhat = what
              , aIndex = plus base (plus rowTerm (term across))
              , aEntry = entry
              , aProducts = bp
              , aCons = bc <> sc <> ec <> digit r 8 <> digit c 8 <> extra
              , aEpoch = ctx.cEpoch
              , aSimdEpoch = ctx.cSimdEpoch
              , aWide = True
              , aStmt = n
              , aDigits = [r, c]
              , aLoops = ctx.cLoops
              , aSince = ctx.cSince
              , aUntil = Map.map (ctx.cAhead <>) ctx.cUntil
              }
    case asConstant stride of
      Just s -> one (times s (term along)) []
      Nothing -> forM_ [0 .. 7] $ \j -> one (times j stride) [equal (term along) (constant j)]

  digit v n = [atMost (constant 0) (term v), atMost (term v) (constant (n - 1))]

  -- An access, and after each loop around whose passes' barriers vary, at
  -- least those of its pass after it.
  record ctx a = modify' (\st -> st {wsAccesses = a {aCons = ctx.cGuards <> exits <> a.aCons} : st.wsAccesses})
   where
    exits =
      concat
        [ [atMost (plus a.aEpoch (constant (fewestOf False u))) x | Just x <- [xb]]
            <> [atMost (plus a.aSimdEpoch (constant (fewestOf True u))) x | Just x <- [xs]]
        | (l, (xb, xs)) <- Map.toList ctx.cExits
        , let u = Map.findWithDefault mempty l a.aUntil
        ]

  -- The loads inside an expression of statement n, each an access.
  loads :: Ctx -> Int -> Expr -> W ()
  loads ctx n = \case
    ELoad b i -> do
      loads ctx n i
      t <- expr ctx i
      element ctx n (Dev b) False ("the load of " <> b <> "[" <> renderExpr i <> "]") 1 t
    ETgLoad a i -> do
      loads ctx n i
      t <- expr ctx i
      element ctx n (Tg a) False ("the load of " <> a <> "[" <> renderExpr i <> "]") 1 t
    ELoadVia b e i -> do
      mapM_ (loads ctx n) [e, i]
      when (Set.member b writableTables) $ do
        te <- entryOf ctx e
        t <- expr ctx i
        elementVia ctx n b False ("the load through " <> b <> "[" <> renderExpr e <> "][" <> renderExpr i <> "]") 1 te t
    EUn _ e -> loads ctx n e
    EBin _ x y -> mapM_ (loads ctx n) [x, y]
    ESelect c x y -> mapM_ (loads ctx n) [c, x, y]
    EFma x y z -> mapM_ (loads ctx n) [x, y, z]
    EConvert _ e -> loads ctx n e
    ENarrowInt _ _ e -> loads ctx n e
    _ -> pure ()

  ------------------------------------------------------------------------
  -- Values

  expr :: Ctx -> Expr -> W Term
  expr ctx = \case
    ELit l -> case l of
      LU32 x -> pure (linear (constant (toInteger x)) [])
      LI32 x -> pure (linear (constant (toInteger x)) [])
      LU8 x -> pure (linear (constant (toInteger x)) [])
      LI8 x -> pure (linear (constant (toInteger x)) [])
      _ -> derived []
    EVar v -> maybe (local ctx v) pure (Map.lookup v ctx.cDefs)
    EParam p
      | Map.member p dims -> pure (linear (term (VDim p)) [])
      | Set.member p scalars -> pure (linear (term (VParam p)) [])
      | otherwise -> unknown
    EBuiltin b -> pure . (`linear` []) $ case b of
      ThreadInGroup -> plus (times width (term (VSimd One))) (term (VLane One))
      SimdLane -> term (VLane One)
      SimdGroup -> term (VSimd One)
      GroupInGrid -> term (VGroup One)
    EBin op x y -> do
      Term a ap ac <- expr ctx x
      Term b bp bc <- expr ctx y
      let cs = ac <> bc
          sumOf p q = Map.filter (/= constant 0) (Map.unionWith plus p q)
      case op of
        Add -> pure (Term (plus a b) (sumOf ap bp) cs)
        -- Without wrapping around: the difference is a u32.
        Sub -> pure (Term (minus a b) (sumOf ap (Map.map (times (-1)) bp)) (cs <> [atMost b a | Map.null ap, Map.null bp]))
        Mul
          | Map.null ap, Just c <- asConstant a -> pure (Term (times c b) (Map.map (times c) bp) cs)
          | Map.null bp, Just c <- asConstant b -> pure (Term (times c a) (Map.map (times c) ap) cs)
          | Map.null ap, Map.null bp, Just (n, c) <- dimension a -> pure (Term (constant 0) (Map.singleton n (times c b)) cs)
          | Map.null ap, Map.null bp, Just (n, c) <- dimension b -> pure (Term (constant 0) (Map.singleton n (times c a)) cs)
          | otherwise -> derived [Term a ap ac, Term b bp bc]
        Div
          | Map.null ap
          , Map.null bp
          , Just d <- asConstant b
          , d > 0 -> do
              (q, _, qc) <- divided (levelOf (Term a ap ac)) a d
              pure (linear (term q) (cs <> qc))
        Rem
          | Map.null ap
          , Map.null bp
          , Just d <- asConstant b
          , d > 0 -> do
              (_, r, qc) <- divided (levelOf (Term a ap ac)) a d
              pure (linear (term r) (cs <> qc))
        -- By a divisor that isn't a constant: an unknown, the remainder
        -- below the divisor; a quotient and a remainder of the same are
        -- remembered together.
        _
          | op `elem` [Div, Rem]
          , Map.null ap
          , Map.null bp -> do
              u <- freshOf [Term a ap ac, Term b bp bc]
              modify' (\st -> st {wsDivisions = Division a b (op == Div) u : st.wsDivisions})
              pure (linear (term u) (cs <> [atMost (constant 0) (term u)] <> [atMost (plus (term u) (constant 1)) b | op == Rem]))
        -- A minimum or a maximum: either operand, the one the order
        -- picks.
        _
          | op `elem` [Min, Max]
          , Map.null ap
          , Map.null bp -> do
              t@(Term m _ _) <- derived [Term a ap ac, Term b bp bc]
              let (lo, hi) = if op == Min then (a, b) else (b, a)
              forM_ (Map.keys (coefficients m)) $ \v ->
                modify' (\st -> st {wsChoices = Map.insert v [[equal m a, atMost lo hi], [equal m b, atMost hi lo]] st.wsChoices})
              pure t
        _ -> derived [Term a ap ac, Term b bp bc]
    -- A buffer the kernel never writes holds one value an element.
    ELoad b i | Set.member b readOnly -> expr ctx i >>= derived . pure
    -- A table the kernel never writes holds one value an element.
    ELoadVia b e i | not (Set.member b writableTables) -> do
      te <- expr ctx e
      ti <- expr ctx i
      derived [te, ti]
    ESelect c x y -> mapM (expr ctx) [c, x, y] >>= derived
    EFma x y z -> mapM (expr ctx) [x, y, z] >>= derived
    EUn _ e -> expr ctx e >>= derived . pure
    EConvert _ e -> expr ctx e >>= derived . pure
    ENarrowInt _ _ e -> expr ctx e >>= derived . pure
    _ -> unknown

  -- A value's affine form: itself, or an unknown if it multiplies a dimension.
  flat t@(Term l ps cs)
    | Map.null ps = pure (l, cs)
    | otherwise = (\(Term u _ uc) -> (u, uc)) <$> derived [t]

  -- A multiple of a dimension and nothing else.
  dimension l = case Map.toList (coefficients l) of
    [(VDim n, c)] | constantPart l == 0 -> Just (n, c)
    _ -> Nothing

  -- The quotient q and remainder r of x by d: x = d·q + r, 0 ≤ r < d, x a
  -- u32; one pair for each value and divisor.
  divided level x d =
    gets (Map.lookup (x, d) . (.wsQuotients)) >>= \case
      Just (q, r) -> pure (q, r, definition x d q r)
      Nothing -> do
        q <- freshFrom level (Map.keys (coefficients x))
        r <- freshFrom level (Map.keys (coefficients x))
        modify' (\st -> st {wsQuotients = Map.insert (x, d) (q, r) st.wsQuotients})
        pure (q, r, definition x d q r)
  definition x d q r = [equal x (plus (times d (term q)) (term r)), atMost (constant 0) x, atMost (constant 0) (term r), atMost (term r) (constant (d - 1))]

  {- A local statements assign: where it's the same in every thread of the
  threadgroup, the same in two at one pass of each loop around. -}
  local ctx v
    | Set.member v uniformLocals = do
        x <- freshFrom (if null ctx.cLoops then Group else Thread) (VGroup One : [VIter One l | l <- ctx.cLoops])
        pure (linear (term x) [atMost (constant 0) (term x)])
    | otherwise = unknown

  -- An unknown u32 nothing is known of.
  unknown = do
    x <- freshAt Thread
    pure (linear (term x) [atMost (constant 0) (term x)])

  {- An unknown u32 computed from values: as uniform as they are, and the
  same in two threads where they are. -}
  derived from = do
    x <- freshOf from
    pure (linear (term x) [atMost (constant 0) (term x)])
  freshOf from = freshFrom (maximum (Launch : map levelOf from)) (nub (concatMap valueVars from))
  valueVars (Term l ps _) = Map.keys (coefficients l) <> concatMap (Map.keys . coefficients) (Map.elems ps) <> map VDim (Map.keys ps)
  freshFrom level from = do
    x <- freshAt level
    case x of
      VAux _ n -> modify' (\st -> st {wsInputs = Map.insert n from st.wsInputs})
      _ -> pure ()
    pure x

  freshAt level = do
    n <- gets (.wsNext)
    modify' (\st -> st {wsNext = n + 1})
    pure $ case level of
      Launch -> VLaunch n
      Group -> VUniform One n
      Thread -> VAux One n

  levelOf (Term l ps _) = maximum (Launch : map varLevel (Map.keys (coefficients l) <> concatMap (Map.keys . coefficients) (Map.elems ps)) <> [Launch | not (Map.null ps)])

  freshVar = freshAt Thread

  -- A condition as constraints on the threads that take the branch, if it is one.
  condition :: Ctx -> Expr -> W (Maybe [Con])
  condition ctx = \case
    EBin And x y -> do
      a <- condition ctx x
      b <- condition ctx y
      pure ((<>) <$> a <*> b)
    EBin op x y | op `elem` [Eq, Lt, Le, Gt, Ge] -> Just <$> compareTerms ctx op x y
    EUn Not e -> negation ctx e
    ELit (LBool True) -> pure (Just [])
    _ -> pure Nothing

  negation :: Ctx -> Expr -> W (Maybe [Con])
  negation ctx = \case
    EBin Lt x y -> Just <$> compareTerms ctx Ge x y
    EBin Le x y -> Just <$> compareTerms ctx Gt x y
    EBin Gt x y -> Just <$> compareTerms ctx Le x y
    EBin Ge x y -> Just <$> compareTerms ctx Lt x y
    EBin Ne x y -> Just <$> compareTerms ctx Eq x y
    EUn Not e -> condition ctx e
    ELit (LBool False) -> pure (Just [])
    _ -> pure Nothing

  compareTerms ctx op x y = do
    (a, ac) <- flat =<< expr ctx x
    (b, bc) <- flat =<< expr ctx y
    pure . ((ac <> bc) <>) $ case op of
      Eq -> [equal a b]
      Lt -> [atMost (plus a (constant 1)) b]
      Le -> [atMost a b]
      Gt -> [atMost (plus b (constant 1)) a]
      _ -> [atMost b a]

  ------------------------------------------------------------------------
  -- Epochs

  {- The barriers before a pass, or after the loop: the count before it
  plus those of the passes before, where every pass holds the same number;
  else a number counted in the pass, or where the loop is, no smaller than
  the count before it plus the fewest those passes hold. -}
  epochAt owner simd before perPass passes least = case perPass of
    Just b -> pure (plus before (times b passes), [])
    Nothing -> (\x -> (term x, [atMost (plus before (constant least)) (term x)])) <$> freshEpoch owner simd

  -- After a branch: the count both sides give, or a number counted where
  -- the branch is, no smaller than the count before it plus the fewest
  -- either side holds.
  joinEpoch ctx simd ready before a b least
    | a == b = pure (a, [])
    | otherwise = do
        x <- maybe (term <$> freshEpoch (listToMaybe ctx.cLoops) simd) pure ready
        pure (x, [atMost (plus before (constant (fewestOf simd least))) x])

  freshEpoch owner simd = do
    n <- gets (.wsNext)
    modify' (\st -> st {wsNext = n + 1, wsEpochs = Map.insert n (owner, simd) st.wsEpochs})
    pure (VEpoch One n)

  -- Barriers in one pass over a body, if the count is a constant.
  count :: Bool -> [Stmt] -> Maybe Integer
  count simd = fmap sum . traverse one
   where
    one = \case
      SBarrier -> Just 1
      SSimdBarrier -> Just (if simd then 1 else 0)
      SIf _ _ th el -> do
        a <- count simd th
        b <- count simd el
        if a == b then Just a else Nothing
      SFor _ _ start end step body -> do
        b <- count simd body
        if b == 0 then Just 0 else (b *) <$> constantTrips start end step
      _ -> Just 0

  -- The fewest barriers statements hold.
  fewest :: [Stmt] -> Fewest
  fewest = foldMap $ \case
    SBarrier -> Fewest 1 1
    SSimdBarrier -> Fewest 0 1
    SIf _ _ th el -> lesser (fewest th) (fewest el)
    SFor _ _ start end step body -> maybe mempty (\t -> let Fewest a b = fewest body in Fewest (t * a) (t * b)) (constantTrips start end step)
    _ -> mempty

  {- A loop's iterations as its bounds give them: the number, for constant
  bounds; else, for a constant step c, c·trips ≥ end − start, and
  c·trips ≤ end − start + c − 1 where end − start is never below 1 − c. -}
  tie guards start end step s0 e0 st0 trips = case constantTrips start end step of
    Just t -> [equal trips (constant t)]
    Nothing -> case asConstant st0 of
      Just c
        | c > 0 ->
            let span' = minus e0 s0
             in atMost span' (times c trips) : [atMost (times c trips) (plus span' (constant (c - 1))) | maybe False (>= 1 - c) (lowest (bounds guards) span')]
      _ -> []

  -- Whether a bound is the same in every thread: of the dimensions and
  -- parameters, and of the threadgroup unless the whole launch is asked.
  onlyLaunch launch l =
    all
      ( \case
          VDim _ -> True
          VDimUnit _ -> True
          VParam _ -> True
          VLaunch _ -> True
          VGroup _ -> not launch
          VUniform _ _ -> not launch
          _ -> False
      )
      (Map.keys (coefficients l))

{- | The least and the greatest value of each variable the constraints
give, from a few rounds of propagating them through each other.
-}
bounds :: [Con] -> Map Var (Maybe Integer, Maybe Integer)
bounds cs = iterate step Map.empty !! 4
 where
  forms = concatMap (\case NonNegative x -> [x]; EqualsZero x -> [x, times (-1) x]) cs
  step b = foldl' tighten' b forms
  -- From k·v + rest ≥ 0: k·v ≥ -max rest.
  tighten' b x = foldl' (\b' (v, k) -> maybe b' (\m -> narrow v k m b') (greatest b (minus x (times k (term v))))) b (Map.toList (coefficients x))
  narrow v k m b'
    | k > 0 = Map.insertWith both v (Just (negate (m `div` k)), Nothing) b'
    | otherwise = Map.insertWith both v (Nothing, Just (m `div` negate k)) b'
  both (lo, hi) (lo', hi') = (maxM lo lo', minM hi hi')
  maxM x y = maybe y (\a -> Just (maybe a (max a) y)) x
  minM x y = maybe y (\a -> Just (maybe a (min a) y)) x

-- | The greatest value of an affine form within bounds, if they bound it.
greatest :: Map Var (Maybe Integer, Maybe Integer) -> Lin -> Maybe Integer
greatest b x = (constantPart x +) . sum <$> traverse one (Map.toList (coefficients x))
 where
  one (v, k) = do
    (lo, hi) <- Map.lookup v b
    if k > 0 then (k *) <$> hi else (k *) <$> lo

-- | The least and greatest values of an affine form within bounds.
range :: Map Var (Maybe Integer, Maybe Integer) -> Lin -> Maybe (Integer, Integer)
range b x = (,) <$> lowest b x <*> greatest b x

-- | The least value of an affine form within bounds, if they bound it.
lowest :: Map Var (Maybe Integer, Maybe Integer) -> Lin -> Maybe Integer
lowest b x = negate <$> greatest b (times (-1) x)

-- | The loops whose iterations an access's constraints count.
tripsOf :: Touch -> [Int]
tripsOf a = nub [l | x <- a.aEpoch : a.aSimdEpoch : a.aIndex : a.aEntry : Map.elems a.aProducts <> map body a.aCons, VTrips _ l <- Map.keys (coefficients x)]
 where
  body = \case
    EqualsZero x -> x
    NonNegative x -> x

-- | How uniform a value is: the same in the launch, in a threadgroup, or neither.
data Level = Launch | Group | Thread
  deriving stock (Eq, Ord, Show)

varLevel :: Var -> Level
varLevel = \case
  VDim _ -> Launch
  VDimUnit _ -> Launch
  VParam _ -> Launch
  VLaunch _ -> Launch
  VGroup _ -> Group
  VUniform _ _ -> Group
  _ -> Thread

{- | The constraints a system needs: those given, and those of the rest
that share a variable with them, or with one of those, and so on. The rest
can only add solutions back, never take one the system has: dropping them
loses no race.
-}
relevant :: [Con] -> [Con] -> [Con]
relevant seeds rest = seeds <> grow (vars seeds) rest
 where
  vars = Set.fromList . concatMap (Map.keys . coefficients . body)
  grow known cs =
    let (touching, others) = partition (any (`Set.member` known) . Map.keys . coefficients . body) cs
     in if null touching then [] else touching <> grow (Set.union known (vars touching)) others
  body = \case
    EqualsZero x -> x
    NonNegative x -> x

-- | The variables of an access's index and constraints.
variablesOf :: Touch -> [Var]
variablesOf a = nub [v | x <- a.aEpoch : a.aSimdEpoch : a.aIndex : a.aEntry : Map.elems a.aProducts <> map body a.aCons, v <- Map.keys (coefficients x)]
 where
  body = \case
    EqualsZero x -> x
    NonNegative x -> x

-- | The unknowns an access shares with every thread of its threadgroup.
uniformsOf :: Touch -> [Int]
uniformsOf a = nub [u | x <- a.aEpoch : a.aSimdEpoch : a.aIndex : a.aEntry : Map.elems a.aProducts <> map body a.aCons, VUniform _ u <- Map.keys (coefficients x)]
 where
  body = \case
    EqualsZero x -> x
    NonNegative x -> x

asConstant :: Lin -> Maybe Integer
asConstant l = if Map.null (coefficients l) then Just (constantPart l) else Nothing

-- | The iterations of a loop of constant bounds.
constantTrips :: Expr -> Expr -> Expr -> Maybe Integer
constantTrips start end step = do
  (s0, e0, st0) <- (,,) <$> constantOf start <*> constantOf end <*> constantOf step
  if st0 > 0 then Just (max 0 ((e0 - s0 + st0 - 1) `div` st0)) else Nothing

constantOf :: Expr -> Maybe Integer
constantOf = \case
  ELit (LU32 n) -> Just (toInteger n)
  EBin op x y -> do
    a <- constantOf x
    b <- constantOf y
    case op of
      Add -> Just (a + b)
      Sub -> Just (a - b)
      Mul -> Just (a * b)
      Div | b > 0 -> Just (a `div` b)
      Rem | b > 0 -> Just (a `mod` b)
      Min -> Just (min a b)
      Max -> Just (max a b)
      _ -> Nothing
  _ -> Nothing

-- | A touch with the variables of the pair's second thread.
onTwo :: Touch -> Touch
onTwo a =
  a
    { aIndex = rename onTwoVar a.aIndex
    , aEntry = rename onTwoVar a.aEntry
    , aProducts = Map.map (rename onTwoVar) a.aProducts
    , aCons = map conOnTwo a.aCons
    , aEpoch = rename onTwoVar a.aEpoch
    , aSimdEpoch = rename onTwoVar a.aSimdEpoch
    , aDigits = map onTwoVar a.aDigits
    }

conOnTwo :: Con -> Con
conOnTwo = \case
  EqualsZero x -> EqualsZero (rename onTwoVar x)
  NonNegative x -> NonNegative (rename onTwoVar x)

-- | A variable of the pair's second thread as the first's.
onOne :: Var -> Var
onOne = \case
  VGroup _ -> VGroup One
  VSimd _ -> VSimd One
  VLane _ -> VLane One
  VIter _ l -> VIter One l
  VTrips _ l -> VTrips One l
  VAux _ n -> VAux One n
  VUniform _ n -> VUniform One n
  VEpoch _ n -> VEpoch One n
  v -> v

onTwoVar :: Var -> Var
onTwoVar = \case
  VGroup _ -> VGroup Two
  VSimd _ -> VSimd Two
  VLane _ -> VLane Two
  VIter _ l -> VIter Two l
  VTrips _ l -> VTrips Two l
  VAux _ n -> VAux Two n
  VUniform _ n -> VUniform Two n
  VEpoch _ n -> VEpoch Two n
  v -> v
