-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | What a program means: its step evaluated in f64, every operation as
exact as a double computes it, in the order of the program; and how far
from that a program's kernels, computing in binary32, can be.

Rounding happens only where a program stores into state of a narrower type:
a KV cache holds each stored element rounded to its element type (bf16 or
f16), to nearest, ties to even, directly from its exact value. Everything
else stays a double, so a program's kernels are compared with what their
arithmetic approximates, not with another approximation of it.

A step's rows store into caches and attend over them in row order, as if
they ran one at a time: a row's attention reads what the rows up to it
stored, and the cache as the step found it for the rest. A row whose
attention reaches past its own position reads as the step's row that
stored the position it reaches, if one did: the stores of the rows between
are its too. A ring without
room for a step's rows of one sequence past the positions a row reads (its
length less the window, or with none the positions' bound, and one) still
gives each row the keys it reads: a plan stages such a step's stores, and
lands them after the attention ("FuryKernel.Program.Plan"). The cache the
step leaves holds each element's last store.

= Bounds

Every element of every value carries a bound: how far from its exact value
a kernel's binary32 value of it can be, derived through the program from
the bounds of what it is computed from, for kernels whose arithmetic keeps
to the contract below. A kernel's store into a cache is right iff it lies
between the roundings to the cache's element type of the exact value less
and plus its bound, or both are NaN. Bounds are computed in f64 and
widened by 2^-20 of themselves, for the f64 evaluation's own rounding.

What the bounds assume of the kernels:

* They read inputs rounded to binary32 (or to bf16, an input of it), and
  weights, tables and caches as bound, and compute in binary32. Addition,
  subtraction, multiplication, division, square root and fused
  multiply-add are correctly rounded, but a subnormal operand may be
  taken as a zero of its sign and a subnormal result given as one, as
  'FuryKernel.IR.BinOp' allows.
* They add a sum's terms and a dot product's products in any order and
  association, fusing multiply-adds or not, and multiply a product's
  factors in any order. Constants (an epsilon, a factor, a scale, GELU's)
  are rounded once to binary32.
* exp and log are within 4 ulp, tanh within 5, sin and cos within 4,
  rsqrt within 2: the Metal Shading Language's precise functions are
  within these, by its specification's table of single-precision accuracy.
* RMSNorm takes the sum of squares times or over the width, plus epsilon,
  its rsqrt (or one over its square root), and each element's product with
  that, its gamma and the factor.
* RoPE takes each frequency @θ^(-2i/d)@ within 4 ulp, the angle as the
  product of the position and the frequency, and the angle's cosine and
  sine; the pairs it doesn't turn stay as they are. Cosines and sines from
  tables, rounded once from their exact values and combined by the
  angle-addition formulas, are within this.
* GELU is @0.5 · a · (1 + tanh(√(2/π) · (a + 0.044715 · a³)))@, a softcap
  @cap · tanh(x / cap)@ or with the reciprocal.
* A route's weights are exp of each logit less the largest, over their
  sum, times the expert's scale.
* Attention's scores are the dot product of the query and the key times the
  scale, or of the query times the scale and the key. Its weights are exp
  of the score less a running maximum (or exp2, both scaled by @log₂ e@),
  its sums rescaled by exp of the maximum's change when it moves, at most
  once for each later key. Its output is the weighted sum of the values
  over the sum of the weights, split or not, the splits combined once.

Routes and arg-maxes choose. Where the bounds of a row's logits leave its
top k or their order undecided, or its largest element (a near tie), a
kernel can choose otherwise, and the ids, the weights and everything
computed from them are unbounded. So is a value whose bound reaches
binary32's overflow, which a kernel's value can then be. A store of an
unbounded element passes whatever it is.

Bounds are worst cases, over a long sum far wider than the error a
kernel's sum makes, so a kernel that rounds the wrong way (toward zero,
say) mostly passes them: rounding modes are left to each target's bitwise
checks (the CPU target's @checks.cpu@, a GPU target's conformance).

The evaluator is a reference for tests, written for clarity: it takes
small programs and shapes. A cache is bound sparsely, by the slots of a
ring or the blocks of a paged pool that the step reads and stores, never
the cache its source declares, which can be larger than memory; reading or
storing one not bound is 'Left'. It checks every binding against what its
source declares, so that over a checked program it returns what is wrong
as 'Left', never throws.
-}
module FuryKernel.Program.Eval (
  Bindings (..),
  Evaluation (..),
  Kept (..),
  evaluate,
  evaluateStored,
  roundBF16,
  geluTanh,
) where

import Control.Monad (foldM, forM, unless, when)
import Data.Array.Unboxed (Array, UArray, bounds, listArray, (!), (//))
import Data.Bits (countLeadingZeros)
import Data.Foldable (for_)
import Data.List (sortBy, transpose, zip4, zip6)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (isNothing, listToMaybe)
import Data.Ord (comparing)
import Data.Text (Text)
import Data.Text qualified as T
import Data.Word (Word32, Word64)
import FuryKernel.IR (Name)
import FuryKernel.Numerics (narrowF16Double, widenF16)
import FuryKernel.Program.Check (CheckedProgram, checkedProgram)
import FuryKernel.Program.Graph
import FuryKernel.Type (ScalarType (..))
import GHC.Float (double2Float, float2Double)
import Numeric.Natural (Natural)

-- | What a step is given, by source name.
data Bindings = Bindings
  { bindRows :: Int
  -- ^ The step's rows.
  , bindInputs :: Map Name [[Double]]
  -- ^ Activations, a list per row.
  , bindRowInts :: Map Name [[Word32]]
  , bindWeights :: Map Name [Double]
  -- ^ Row-major, outputs first, widened from their storage type.
  , bindTables :: Map Name [[Double]]
  -- ^ A weight per entry.
  , bindCaches :: Map Name (Map Int [Double])
  {- ^ As stored, widened: per cache, the slots of a ring or the blocks of
  a paged pool it binds, each its elements.
  -}
  }

data Evaluation = Evaluation
  { evalValues :: Map ValueId [[Double]]
  -- ^ Every value, a list per row; ids as doubles.
  , evalBounds :: Map ValueId [[Double]]
  {- ^ Every value's bounds, as its elements: how far from each a kernel's
  binary32 value can be, infinite where unbounded.
  -}
  , evalOutputs :: Map Name [[Double]]
  , evalCaches :: Map Name (Map Int [Double])
  -- ^ The caches' bound slots or blocks after the step's stores.
  }

type Arr = UArray Int Double

-- | An element of a cache: its slot or block, and its index in that.
type Element = (Int, Int)

-- | A cache's bound slots or blocks.
type Units = Map Int Arr

data Env = Env
  { envValues :: Map ValueId [Arr]
  , envBounds :: Map ValueId [Arr]
  , envCaches :: Map Name Units
  , envStores :: Map Name (Units, Map Element [(Int, Int, Approx)])
  {- ^ Per cache the step stored into: the cache as the step found it,
  and at each element the step's stores, (row, position, value) in row
  order, each value with how far from it what the kernels stored can be.
  -}
  }

-- | What a step's kernels stored, by cache name.
data Kept = Kept
  { keptCaches :: Map Name (Map Int [Double])
  {- ^ The caches as the step left them: at least the slots or blocks the
  step stores into.
  -}
  , keptRows :: Map Name [[Double]]
  {- ^ The values the kernels staged for a cache's store, each row's as
  the store's value has its elements: every row's, those a later row's
  store overwrote included.
  -}
  }

evaluate :: CheckedProgram -> Bindings -> Either Text Evaluation
evaluate = evaluateWith Nothing

{- | 'evaluate', given what a step's kernels stored: each store into a
cache keeps the element the kernels stored (staged, if they staged it,
else as they left the cache), after checking that the element's bound
allows it, so that what attends over the caches reads the keys and values
the kernels read. A cache the kernels staged must hold each element's last
staged value. Without the staged rows, a store a later row's overwrote
keeps the evaluator's own rounding.
-}
evaluateStored :: Kept -> CheckedProgram -> Bindings -> Either Text Evaluation
evaluateStored kept = evaluateWith (Just kept)

evaluateWith :: Maybe Kept -> CheckedProgram -> Bindings -> Either Text Evaluation
evaluateWith kept checked b = do
  caches <- forM (Map.toList b.bindCaches) $ \(n, units) -> do
    for_ [l | Source {sourceName = n', sourceKind = SCache l _} <- p.programSources, n' == n] $ \l ->
      for_ (Map.toList units) $ \(u, xs) -> do
        unless (u >= 0 && toInteger u < toInteger (unitCount l)) $
          Left (n <> " binds " <> unitName l <> " " <> showT u <> ", of the " <> showT (unitCount l) <> " it has.")
        sized (n <> "'s " <> unitName l <> " " <> showT u) (length xs) (unitSize l)
    pure (n, Map.map arr units)
  env <- foldM node (Env Map.empty Map.empty (Map.fromList caches) Map.empty) p.programNodes
  let rows = map elemsOf
  outs <- forM p.programOutputs $ \o -> case Map.lookup o.outputValue env.envValues of
    Just v -> Right (o.outputName, rows v)
    Nothing -> Left ("The output " <> o.outputName <> " was never made.")
  pure
    Evaluation
      { evalValues = Map.map rows env.envValues
      , evalBounds = Map.map rows env.envBounds
      , evalOutputs = Map.fromList outs
      , evalCaches = Map.map (Map.map elemsOf) env.envCaches
      }
 where
  p = checkedProgram checked
  m = b.bindRows
  sources = Map.fromList [(s.sourceId, s) | s <- p.programSources]
  values = Map.fromList [(v.valueId, v) | v <- p.programValues]

  sourceOf :: SourceId -> Either Text Source
  sourceOf s = maybe (Left "A source of no name.") Right (Map.lookup s sources)

  weightOf :: SourceId -> Either Text Arr
  weightOf s = do
    src <- sourceOf s
    xs <- maybe (Left ("No weight " <> src.sourceName <> ".")) Right (Map.lookup src.sourceName b.bindWeights)
    case src.sourceKind of
      SWeight dims _ _ -> sized src.sourceName (length xs) (fromIntegral (product dims))
      _ -> pure ()
    pure (arr xs)

  intsOf :: SourceId -> Either Text [[Word32]]
  intsOf s = do
    src <- sourceOf s
    ints <- maybe (Left ("No row integers " <> src.sourceName <> ".")) Right (Map.lookup src.sourceName b.bindRowInts)
    when (length ints /= m) $ Left (src.sourceName <> " has " <> showT (length ints) <> " rows, not " <> showT m <> ".")
    for_ [k | SRowInts k _ <- [src.sourceKind]] $ \k -> for_ ints $ \row -> sized src.sourceName (length row) (fromIntegral k)
    case [(r, i, bound) | SRowInts _ bound <- [src.sourceKind], (r, row) <- zip [0 :: Int ..] ints, i <- row, fromIntegral i >= bound] of
      (r, i, bound) : _ -> Left (src.sourceName <> ": row " <> showT r <> " holds " <> showT i <> ", which isn't below its bound of " <> showT bound <> ".")
      [] -> pure ints

  int1 :: SourceId -> Either Text [Int]
  int1 s = map (fromIntegral . headOr 0) <$> intsOf s

  intRows :: SourceId -> Either Text [[Int]]
  intRows s = map (map fromIntegral) <$> intsOf s

  valueOf :: Env -> ValueId -> Either Text [Arr]
  valueOf env v = maybe (Left "A value read before it is made.") Right (Map.lookup v env.envValues)

  boundsOf :: Env -> ValueId -> Either Text [Arr]
  boundsOf env v = maybe (Left "A value read before it is made.") Right (Map.lookup v env.envBounds)

  -- A value's rows, each element with its bound.
  approxOf :: Env -> ValueId -> Either Text [[Approx]]
  approxOf env v = zipWith (\x e -> zipWith Approx (elemsOf x) (elemsOf e)) <$> valueOf env v <*> boundsOf env v

  shapeOf :: ValueId -> Shape
  shapeOf v = maybe (Shape 0 0) (.valueShape) (Map.lookup v values)

  node :: Env -> Node -> Either Text Env
  node env n = case (n.nodeOp, n.nodeOutputs) of
    (Input s, [o]) -> do
      src <- sourceOf s
      rs <- maybe (Left ("No input " <> src.sourceName <> ".")) Right (Map.lookup src.sourceName b.bindInputs)
      unless (length rs == m) $ Left (src.sourceName <> " has " <> showT (length rs) <> " rows, not " <> showT m <> ".")
      for_ rs $ \row -> sized src.sourceName (length row) (size (shapeOf o))
      -- As kernels read it: rounded to binary32, or to bf16 for an input
      -- of it.
      let readAs = case src.sourceKind of
            SInput _ TBF16 -> roundBF16
            _ -> float2Double . double2Float
      put o (map arr rs) [arr [(settle (Approx x (abs (x - readAs x)))).aBound | x <- row] | row <- rs]
    (Linear w _ x, [o]) -> do
      wa <- weightOf w
      xs <- valueOf env x
      as <- approxOf env x
      let k = size (shapeOf x)
          rowsOut = size (shapeOf o)
      put
        o
        [listArray (0, rowsOut - 1) [dot wa (j * k) xr k | j <- [0 .. rowsOut - 1]] | xr <- xs]
        [boundsArr [dotA (zip [exactA (wa ! (j * k + i)) | i <- [0 .. k - 1]] ar) | j <- [0 .. rowsOut - 1]] | ar <- as]
    (LinearRouted t ids x, [o]) -> do
      src <- sourceOf t
      entries <- maybe (Left ("No table " <> src.sourceName <> ".")) Right (Map.lookup src.sourceName b.bindTables)
      case src.sourceKind of
        STable count dims _ _ -> do
          unless (length entries == fromIntegral count) $
            Left (src.sourceName <> " has " <> showT (length entries) <> " entries bound, not " <> showT count <> ".")
          for_ entries $ \e -> sized src.sourceName (length e) (fromIntegral (product dims))
        _ -> pure ()
      idss <- valueOf env ids
      idBounds <- boundsOf env ids
      xs <- valueOf env x
      as <- approxOf env x
      let outW = size' (shapeOf o).shapeSegment
          k = case src.sourceKind of
            STable _ [_, c] _ _ -> fromIntegral c
            _ -> 0
          shared = size (shapeOf x) == k
      rowsOut <- forM (zip4 idss idBounds xs as) $ \(idr, idb, xr, ar) ->
        fmap unzipConcat . forM (zip3 [0 ..] (elemsOf idr) (elemsOf idb)) $ \(rank, e, eb) -> do
          let entry = round e :: Int
          when (entry < 0 || entry >= length entries) $ Left (n.nodeName <> ": expert " <> showT entry <> " is not in the table.")
          let wa = arr (entries !! entry)
              base = if shared then 0 else rank * k
              slice = take k (drop base ar)
          pure
            ( [dot wa (j * k) (sliceArr xr base k) k | j <- [0 .. outW - 1]]
            , -- A kernel's expert can be another where its id is unbounded.
              [if eb > 0 then 1 / 0 else (dotA (zip [exactA (wa ! (j * k + i)) | i <- [0 .. k - 1]] slice)).aBound | j <- [0 .. outW - 1]]
            )
      put o (map (arr . fst) rowsOut) (map (arr . snd) rowsOut)
    (RmsNorm eps g f x, [o]) -> do
      gamma <- traverse weightOf g
      xs <- valueOf env x
      as <- approxOf env x
      let seg = size' (shapeOf x).shapeSegment
      put
        o
        [arr (concatMap (norm gamma seg) (chunks seg (elemsOf xr))) | xr <- xs]
        [arr (concatMap (normBounds eps gamma f seg) (chunks seg ar)) | ar <- as]
     where
      norm gamma seg xs =
        let ms = sum (map (\v -> v * v) xs) / fromIntegral seg
            s = (ms + eps) ** (-0.5)
         in [v * s * maybe 1 (! j) gamma * f | (j, v) <- zip [0 ..] xs]
    (Add a c, [o]) -> do
      as <- valueOf env a
      cs <- valueOf env c
      aas <- approxOf env a
      cas <- approxOf env c
      put
        o
        [arr (zipWith (+) (elemsOf x) (elemsOf y)) | (x, y) <- zip as cs]
        [boundsArr (zipWith (\u v -> sumA [u, v]) x y) | (x, y) <- zip aas cas]
    (Concat a c, [o]) -> do
      as <- valueOf env a
      cs <- valueOf env c
      ab <- boundsOf env a
      cb <- boundsOf env c
      put o [arr (elemsOf x <> elemsOf y) | (x, y) <- zip as cs] [arr (elemsOf x <> elemsOf y) | (x, y) <- zip ab cb]
    (ScaleBy w x, [o]) -> do
      wa <- weightOf w
      xs <- valueOf env x
      as <- approxOf env x
      put o [arr (map (* (wa ! 0)) (elemsOf xr)) | xr <- xs] [boundsArr [productA [u, exactA (wa ! 0)] | u <- ar] | ar <- as]
    (GatedGelu form x, [o]) -> do
      xs <- valueOf env x
      as <- approxOf env x
      let seg = size' (shapeOf x).shapeSegment
          gated s = let (a, c) = splitAt (seg `div` 2) s in zipWith (\u v -> gelu form u * v) a c
          gatedA s = let (a, c) = splitAt (seg `div` 2) s in zipWith (\u v -> productA [geluBounds form u, v]) a c
      put o [arr (concatMap gated (chunks seg (elemsOf xr))) | xr <- xs] [boundsArr (concatMap gatedA (chunks seg ar)) | ar <- as]
    (GeluMul form a c, [o]) -> do
      as <- valueOf env a
      cs <- valueOf env c
      aas <- approxOf env a
      cas <- approxOf env c
      put
        o
        [arr (zipWith (\u v -> gelu form u * v) (elemsOf x) (elemsOf y)) | (x, y) <- zip as cs]
        [boundsArr (zipWith (\u v -> productA [geluBounds form u, v]) x y) | (x, y) <- zip aas cas]
    (Rope r pos x, [o]) -> do
      ps <- intRows pos
      xs <- valueOf env x
      as <- approxOf env x
      let d = size' (shapeOf x).shapeSegment
          part = d `div` size' r.ropeAxes
          rotateHead positions h = concat (zipWith (rotate r part . fromIntegral) positions (chunks part h))
          rotateBounds positions h = concat (zipWith (rotationBounds r part . fromIntegral) positions (chunks part h))
      put
        o
        [arr (concatMap (rotateHead positions) (chunks d (elemsOf xr))) | (positions, xr) <- zip ps xs]
        [arr (concatMap (rotateBounds positions) (chunks d ar)) | (positions, ar) <- zip ps as]
    (Softcap c x, [o]) -> do
      xs <- valueOf env x
      as <- approxOf env x
      put
        o
        [arr (map (\v -> c * tanh (v / c)) (elemsOf xr)) | xr <- xs]
        [boundsArr [productA [constA c, monotoneA 5 tanh (productA [u, constA (1 / c)])] | u <- ar] | ar <- as]
    (Route r x, [ids, ws]) -> do
      scale <- traverse weightOf r.routeExpertScale
      xs <- valueOf env x
      as <- approxOf env x
      let k = fromIntegral r.routeK
          routes = map (routeRow r.routeWeights scale k . elemsOf) xs
          -- Decided, the ids are exact and the weights within the
          -- softmax's bounds; else a kernel can route otherwise.
          routeBounds ar
            | routeDecided k ar = (map (const 0) (take k ar), routeWeightBounds r.routeWeights scale k ar)
            | otherwise = (map (const (1 / 0)) (take k ar), map (const (1 / 0)) (take k ar))
          bs = map routeBounds as
      env' <- put ids [arr (map (fromIntegral . fst) rt) | rt <- routes] (map (arr . fst) bs)
      putIn env' ws [arr (map snd rt) | rt <- routes] (map (arr . snd) bs)
    (CombineRanks w x, [o]) -> do
      wss <- valueOf env w
      xs <- valueOf env x
      was <- approxOf env w
      as <- approxOf env x
      let seg = size' (shapeOf x).shapeSegment
      put
        o
        [arr (foldl1 (zipWith (+)) [map (* wr) s | (wr, s) <- zip (elemsOf wr') (chunks seg (elemsOf xr))]) | (wr', xr) <- zip wss xs]
        [boundsArr [dotA (zip wa column) | column <- transpose (chunks seg ar)] | (wa, ar) <- zip was as]
    (KvStore (StoreAt c pos slot) x, []) -> do
      src <- sourceOf c
      (layout, held) <- case src.sourceKind of
        SCache l t -> Right (l, t)
        _ -> Left (src.sourceName <> " is not a cache.")
      let name = src.sourceName
      when (Map.member name env.envStores) $
        Left (n.nodeName <> " stores into " <> name <> " again: a program stores into a cache once.")
      ps <- int1 pos
      seqs <- intsOf slot
      as <- approxOf env x
      cacheUnits <- maybe (Left ("No cache " <> name <> ".")) Right (Map.lookup name env.envCaches)
      let d = fromIntegral (cacheHeadDim layout)
          heads = fromIntegral (cacheKvHeads layout)
          staged = kept >>= Map.lookup name . (.keptRows)
      -- Each row's stores, rows in order: (row, position, element, the
      -- element's value with its bound, the value the step keeps).
      own <- fmap concat . forM (zip4 [0 :: Int ..] ps seqs as) $ \(r, position, sq, ar) -> do
        row' <- case staged of
          Nothing -> pure Nothing
          Just rs -> case drop r rs of
            sr : _ | length sr == heads * d -> pure (Just (arr sr))
            _ -> Left ("The kernels staged no row " <> showT r <> " for " <> name <> ".")
        let elements = listArray (0, heads * d - 1) ar :: Array Int Approx
        forM [(h, i) | h <- [0 .. heads - 1], i <- [0 .. d - 1]] $ \(h, i) -> do
          (u, at) <- cacheIndex layout sq position h
          let element = elements ! (h * d + i)
          value <- case row' of
            Nothing -> pure (ownRounding held element)
            Just sr -> do
              let k = sr ! (h * d + i)
              unless (mayStore held element k) $
                Left ("The kernels staged " <> showT k <> " for " <> name <> " at " <> elementText layout (u, at + i) <> ", where " <> allowed held element <> ".")
              pure (exactA k)
          pure (r, position, (u, at + i), element, value)
      checkWrites name layout cacheUnits [a | (_, _, a, _, _) <- own]
      -- What the step leaves: each element's last store, the kernels' if
      -- they left the cache.
      let lasts = Map.fromList [(a, (r, element, v)) | (r, _, a, element, v) <- own]
      finals <- case kept >>= Map.lookup name . (.keptCaches) of
        Nothing -> pure (Map.map (\(r, _, v) -> (r, v)) lasts)
        Just ks -> do
          let keptUnits = Map.map arr ks
          flip Map.traverseWithKey lasts $ \a (r, element, v) -> do
            k <- maybe (Left ("The kernels left nothing in " <> name <> " at " <> elementText layout a <> ", which the step stores into.")) Right (unitAt keptUnits a)
            case staged of
              Just _ ->
                unless (k == v.aExact || isNaN k && isNaN v.aExact) $
                  Left ("The kernels left " <> showT k <> " in " <> name <> " at " <> elementText layout a <> ", not the " <> showT v.aExact <> " they staged for it.")
              Nothing ->
                unless (mayStore held element k) $
                  Left ("The kernels stored " <> showT k <> " in " <> name <> " at " <> elementText layout a <> ", where " <> allowed held element <> ".")
            pure (r, exactA k)
      let used = [(a, [(r, position, if fmap fst (Map.lookup a finals) == Just r then maybe v snd (Map.lookup a finals) else v)]) | (r, position, a, _, v) <- own]
      pure
        env
          { envCaches = Map.insert name (writeUnits cacheUnits [(a, v.aExact) | (a, (_, v)) <- Map.toList finals]) env.envCaches
          , envStores = Map.insert name (cacheUnits, Map.fromListWith (flip (<>)) used) env.envStores
          }
    (Attend a q (AttendTo k v pos slot), [o]) -> do
      kSrc <- sourceOf k
      vSrc <- sourceOf v
      layout <- case kSrc.sourceKind of
        SCache l _ -> Right l
        _ -> Left (kSrc.sourceName <> " is not a cache.")
      kRead <- reader kSrc.sourceName
      vRead <- reader vSrc.sourceName
      ps <- int1 pos
      reaches <- maybe (Right ps) (int1 . (.reachSource)) a.attnReach
      for_ a.attnReach $ \r ->
        case [(i, p', rc) | (i, p', rc) <- zip3 [0 :: Int ..] ps reaches, rc > p' + fromIntegral r.reachAhead] of
          (i, p', rc) : _ -> Left (n.nodeName <> ": row " <> showT i <> " reaches " <> showT rc <> ", more than " <> showT r.reachAhead <> " past its position, " <> showT p' <> ".")
          [] -> pure ()
      seqs <- intsOf slot
      qs <- valueOf env q
      qbs <- boundsOf env q
      let d = fromIntegral a.attnHeadDim
          heads = fromIntegral a.attnHeads
          group = heads `div` fromIntegral a.attnKvHeads
      rowsOut <- forM (zip6 [0 :: Int ..] ps reaches seqs qs qbs) $ \(r0, position, reach, sq, qr, qb) -> do
        let first = maybe 0 (\w -> max 0 (position - fromIntegral w + 1)) a.attnWindow
            ts = [first .. max position reach]
            -- A row reading past its own position reads as the row of the
            -- step that stored its reach, if one did: the stores of the
            -- rows between are its too.
            r = headOr r0 [j | (j, p', s') <- zip3 [0 ..] ps seqs, j >= r0, p' == reach, s' == sq]
        fmap unzipConcat . forM [0 .. heads - 1] $ \h -> do
          let kvh = h `div` group
              query = sliceArr qr (h * d) d
              queryA = zipWith Approx (elemsOf query) (elemsOf (sliceArr qb (h * d) d))
          keys <- forM ts $ \t -> do
            (u, at) <- cacheIndex layout sq t kvh
            ks <- forM [0 .. d - 1] $ \i -> kRead r0 r t (u, at + i)
            vs <- forM [0 .. d - 1] $ \i -> vRead r0 r t (u, at + i)
            pure (ks, vs)
          let exacts = arr . map (.aExact)
              arrays = [(exacts ks, exacts vs) | (ks, vs) <- keys]
              scores = [a.attnScale * sum [query ! i * ks ! i | i <- [0 .. d - 1]] | (ks, _) <- arrays]
              top = maximum scores
              es = map (\s -> exp (s - top)) scores
              total = sum es
              -- A kernel's score: the scale on the dot product, or on the
              -- query.
              scoreA ks s =
                Approx s $
                  max
                    (productA [dotA (zip queryA ks), constA a.attnScale]).aBound
                    (dotA (zip [productA [e, constA a.attnScale] | e <- queryA] ks)).aBound
          pure
            ( [sum [e / total * vs ! i | (e, (_, vs)) <- zip es arrays] | i <- [0 .. d - 1]]
            , attendBounds (zipWith scoreA (map fst keys) scores) (transpose (map snd keys))
            )
      put o (map (arr . fst) rowsOut) (map (arr . snd) rowsOut)
    (Gather t tok f, [o]) -> do
      wa <- weightOf t
      toks <- intRows tok
      let width = size (shapeOf o)
      put
        o
        [listArray (0, width - 1) [sum [wa ! (token * width + j) | token <- tokens] * f | j <- [0 .. width - 1]] | tokens <- toks]
        [boundsArr [gatherA [wa ! (token * width + j) | token <- tokens] f | j <- [0 .. width - 1]] | tokens <- toks]
    (Standardize biasSrc scaleSrc f x, [o]) -> do
      bias <- weightOf biasSrc
      scale <- weightOf scaleSrc
      xs <- valueOf env x
      as <- approxOf env x
      put
        o
        [arr [(f * v - bias ! j) * scale ! j | (j, v) <- zip [0 ..] (elemsOf xr)] | xr <- xs]
        [boundsArr [productA [sumA [productA [constA f, u], negA (exactA (bias ! j))], exactA (scale ! j)] | (j, u) <- zip [0 ..] ar] | ar <- as]
    (ArgMax x, [o]) -> do
      xs <- valueOf env x
      as <- approxOf env x
      put o [arr [fromIntegral (argMax (elemsOf xr))] | xr <- xs] [arr [if argMaxDecided ar then 0 else 1 / 0] | ar <- as]
    _ -> Left (n.nodeName <> ": the wrong number of values.")
   where
    put = putIn env
    putIn e o vs bs = Right e {envValues = Map.insert o vs e.envValues, envBounds = Map.insert o bs e.envBounds}

    {- An element of a cache as row @r@ reads it for position @t@, reading
    to the row @r'@ of its reach (@r@ itself if it reads to its own
    position): a row's store of position @t@ after @r@, up to @r'@, if one
    stored it, as a lowering that stages the step's keys and values gives
    it; else the last store of a row up to @r@, if the step stored the
    element, else the cache as the step found it. A store of another
    position there by a row up to @r@ means the step stored more of a
    sequence than its cache keeps. Each comes with how far from it what
    the kernels stored can be. -}
    reader :: Name -> Either Text (Int -> Int -> Int -> Element -> Either Text Approx)
    reader name = case (Map.lookup name env.envStores, Map.lookup name env.envCaches) of
      (Just (before, stores), _) -> Right $ \r r' t at ->
        let upTo k = reverse (takeWhile (\(j, _, _) -> j <= k) (Map.findWithDefault [] at stores))
         in case [value | (j, t', value) <- upTo r', j > r, t' == t] of
              value : _ -> Right value
              [] -> case upTo r of
                (_, t', value) : _
                  | t' == t -> Right value
                  | otherwise -> Left (name <> ": row " <> showT r <> " reads position " <> showT t <> " where a row stored " <> showT t' <> ".")
                [] -> found name before at
      (Nothing, Just units) -> Right $ \_ _ _ at -> found name units at
      (Nothing, Nothing) -> Left ("No cache " <> name <> ".")
    found name units at = maybe (Left (unbound name at)) (Right . exactA) (unitAt units at)

  -- Every store within a bound slot or block of the cache.
  checkWrites name layout units ws = case [a | a <- ws, isNothing (unitAt units a)] of
    [] -> Right ()
    a : _ -> Left ("A store to " <> name <> " at " <> elementText layout a <> ", which isn't bound.")

  unbound name at = name <> ": " <> maybe "slot or block" (`elementText` at) (Map.lookup name cacheLayouts) <> ", isn't bound."
  cacheLayouts = Map.fromList [(s.sourceName, l) | s <- p.programSources, SCache l _ <- [s.sourceKind]]
  elementText layout (u, i) = unitName layout <> " " <> showT u <> ", element " <> showT i

  -- A binding of as many elements as its source declares.
  sized :: Name -> Int -> Int -> Either Text ()
  sized name got want = unless (got == want) $ Left (name <> " is bound to " <> showT got <> " elements, not the " <> showT want <> " it declares.")

  -- Where head h of a position starts: its slot or block, and its index in that.
  cacheIndex :: CacheLayout -> [Word32] -> Int -> Int -> Either Text Element
  cacheIndex layout sq position h = case layout of
    Ring _ _ len d -> do
      let s = fromIntegral (headOr 0 sq)
      pure (s, (h * size' len + position `mod` size' len) * size' d)
    Paged _ blockSize _ d -> do
      let bs = size' blockSize
          slotInTable = position `div` bs
      when (slotInTable >= length sq) $ Left ("Position " <> showT position <> " is past the block table.")
      let block = fromIntegral (sq !! slotInTable)
      pure (block, (h * bs + position `mod` bs) * size' d)

-- | A ring's slots, or a paged pool's blocks.
unitCount :: CacheLayout -> Natural
unitCount = \case
  Ring slots _ _ _ -> slots
  Paged blocks _ _ _ -> blocks

-- | The elements of a slot of a ring, or of a block of a paged pool.
unitSize :: CacheLayout -> Int
unitSize = \case
  Ring _ heads len d -> fromIntegral (heads * len * d)
  Paged _ blockSize heads d -> fromIntegral (blockSize * heads * d)

unitName :: CacheLayout -> Text
unitName = \case
  Ring {} -> "slot"
  Paged {} -> "block"

-- | An element of a bound slot or block.
unitAt :: Units -> Element -> Maybe Double
unitAt units (u, i) = do
  a <- Map.lookup u units
  let (lo, hi) = bounds a
  if i >= lo && i <= hi then Just (a ! i) else Nothing

-- | Stores into bound slots or blocks.
writeUnits :: Units -> [(Element, Double)] -> Units
writeUnits units ws = foldl' (\m (u, xs) -> Map.adjust (// xs) u m) units (Map.toList (Map.fromListWith (flip (<>)) [(u, [(i, v)]) | ((u, i), v) <- ws]))

-- | The exact rotation of one head of width @d@.
rotate :: RopeSpec -> Int -> Double -> [Double] -> [Double]
rotate r d position xs = case r.ropePairs of
  RopeHalves -> map new [0 .. d - 1]
 where
  half = d `div` 2
  a = listArray (0, d - 1) xs :: Arr
  angle i
    | i < fromIntegral r.ropeRotated = position * r.ropeTheta ** negate (fromIntegral (2 * i) / fromIntegral d)
    | otherwise = 0
  new j
    | j < half = let t = angle j in a ! j * cos t - a ! (j + half) * sin t
    | otherwise = let i = j - half; t = angle i in a ! j * cos t + a ! i * sin t

-- | The top @k@ of a row's logits, ties to the lower index, with weights.
routeRow :: RouteWeights -> Maybe Arr -> Int -> [Double] -> [(Int, Double)]
routeRow SoftmaxOfTopK scale k xs = zip top [e / total * maybe 1 (! i) scale | (i, e) <- zip top es]
 where
  order = sortBy (\(i, x) (j, y) -> comparing negate x y <> compare i j) (zip [0 ..] xs)
  top = map fst (take k order)
  best = maybe 0 snd (listToMaybe order)
  es = [exp (x - best) | (_, x) <- take k order]
  total = sum es

-- | GELU, in the tanh approximation.
geluTanh :: Double -> Double
geluTanh x = 0.5 * x * (1 + tanh (sqrt (2 / pi) * (x + 0.044715 * x * x * x)))

-- | The GELU of a form.
gelu :: Gelu -> Double -> Double
gelu GeluTanh = geluTanh

{- | Rounds to the nearest bfloat16, ties to even, directly from the double:
eight significant bits, binary32's exponent range, gradual underflow, and
overflow to infinity.
-}
roundBF16 :: Double -> Double
roundBF16 x
  | isNaN x || isInfinite x || x == 0 = x
  | abs rounded >= 2 ^^ (128 :: Int) = if x > 0 then 1 / 0 else -(1 / 0)
  | otherwise = rounded
 where
  -- x = mantissa · 2^exponent, so its leading bit is at exponent plus the
  -- mantissa's bit length less one.
  (mantissa, exponent') = decodeFloat x
  leading = exponent' + (63 - countLeadingZeros (fromInteger (abs mantissa) :: Word64))
  e = max (-126) leading
  ulp = 2 ^^ (e - 7) :: Rational
  -- round on a Rational rounds half to even.
  rounded = fromRational (fromInteger (round (toRational x / ulp)) * ulp)

argMax :: [Double] -> Int
argMax xs = fst (foldl1 (\(i, x) (j, y) -> if y > x then (j, y) else (i, x)) (zip [0 ..] xs))

------------------------------------------------------------------------------
-- Bounds

-- | An element: its exact value, and how far from it a kernel's binary32 value can be.
data Approx = Approx {aExact :: Double, aBound :: Double}

-- | binary32's unit roundoff, and the most a value flushed to zero loses.
unit, flush :: Double
unit = 2 ^^ (-24 :: Int)
flush = 2 ^^ (-126 :: Int)

{- | The relative error of @n@ roundings, @γ_n = n u / (1 − n u)@; infinite
where they could lose everything.
-}
gammaN :: Int -> Double
gammaN n
  | n <= 0 = 0
  | nu >= 0.5 = 1 / 0
  | otherwise = nu / (1 - nu)
 where
  nu = fromIntegral n * unit

exactA :: Double -> Approx
exactA x = Approx x 0

-- | A constant, rounded once to binary32.
constA :: Double -> Approx
constA c = settle (Approx c (abs (c - float2Double (double2Float c))))

negA :: Approx -> Approx
negA a = Approx (negate a.aExact) a.aBound

-- | The largest magnitude a kernel's value can have.
mag :: Approx -> Double
mag a = abs a.aExact + a.aBound

{- | A bound made safe: unbounded if not a number, or if a kernel's value
can overflow binary32; at least the value's magnitude if a kernel can flush
it to zero.
-}
settle :: Approx -> Approx
settle a@(Approx x e)
  | isNaN x = a
  | isNaN e || mag a >= overflow = Approx x (1 / 0)
  | abs x - e < flush = Approx x (max e (abs x))
  | otherwise = a
 where
  -- The least magnitude binary32 rounds to infinity.
  overflow = 2 ^^ (128 :: Int) - 2 ^^ (103 :: Int)

-- | A sum, its terms added in any order.
sumA :: [Approx] -> Approx
sumA ts =
  settle . Approx (sum (map (.aExact) ts)) $
    sum (map (.aBound) ts) + gammaN (n - 1) * sum (map mag ts) + 2 * fromIntegral n * flush
 where
  n = length ts

-- | A dot product, its products added in any order, fused or not.
dotA :: [(Approx, Approx)] -> Approx
dotA ps =
  settle . Approx (sum [a.aExact * c.aExact | (a, c) <- ps]) $
    sum (map spread ps) + gammaN n * sum [mag a * mag c | (a, c) <- ps] + 4 * fromIntegral n * flush
 where
  n = length ps
  -- (|a| + e_a)(|c| + e_c) − |a c|, without the subtraction's cancellation.
  spread (a, c) = abs a.aExact * c.aBound + a.aBound * abs c.aExact + a.aBound * c.aBound

{- | A product, its factors multiplied in any order: a result flushed to
zero loses at most binary32's least normal, times the factors after it.
-}
productA :: [Approx] -> Approx
productA fs =
  settle . Approx (product (map (.aExact) fs)) $
    spread + gammaN (count - 1) * (whole + spread) + 2 * fromIntegral count * flush * product [max 1 (mag f) | f <- fs]
 where
  count = length fs
  -- Π |x|, and Π (|x| + e) − Π |x| without the subtraction's cancellation.
  (whole, spread) = foldl' (\(w, e) f -> (w * abs f.aExact, e * mag f + w * f.aBound)) (1, 0) fs

-- | A quotient, correctly rounded; unbounded if the divisor's bound reaches zero.
quotientA :: Approx -> Approx -> Approx
quotientA a c
  | isNaN c.aExact || isNaN c.aBound || abs c.aExact <= c.aBound = Approx q (1 / 0)
  | otherwise = settle (Approx q ((abs a.aExact * c.aBound + a.aBound * abs c.aExact) / (abs c.aExact * low) + unit * mag a / low + flush))
 where
  q = a.aExact / c.aExact
  low = abs c.aExact - c.aBound

{- | A function within @k@ ulp, monotone over the bound: as far from the
exact value as its values at the bound's ends, and its own error.
-}
monotoneA :: Double -> (Double -> Double) -> Approx -> Approx
monotoneA k f a
  | isInfinite a.aBound = Approx y (1 / 0)
  | otherwise = settle (Approx y (worst [abs (lo - y), abs (hi - y)] + 2 * k * unit * worst [abs lo, abs hi] + flush))
 where
  y = f a.aExact
  lo = f (a.aExact - a.aBound)
  hi = f (a.aExact + a.aBound)

-- | The largest, NaN if any is.
worst :: [Double] -> Double
worst xs
  | any isNaN xs = 0 / 0
  | otherwise = maximum (0 : xs)

-- | The bounds of the GELU of a form.
geluBounds :: Gelu -> Approx -> Approx
geluBounds GeluTanh = geluA

-- | GELU in the contract's form.
geluA :: Approx -> Approx
geluA a =
  productA
    [ constA 0.5
    , a
    , sumA [exactA 1, monotoneA 5 tanh (productA [constA (sqrt (2 / pi)), sumA [a, productA [constA 0.044715, a, a, a]]])]
    ]

-- | The bounds of a segment's RMSNorm.
normBounds :: Double -> Maybe Arr -> Double -> Int -> [Approx] -> [Double]
normBounds eps gamma f seg xs = [(productA ([x, r, constA f] <> [exactA (g ! j) | Just g <- [gamma]])).aBound | (j, x) <- zip [0 ..] xs]
 where
  meanSquare = productA [dotA [(x, x) | x <- xs], constA (1 / fromIntegral seg)]
  r = monotoneA 2 (\v -> 1 / sqrt v) (sumA [meanSquare, constA eps])

{- | The cosine and sine of a rotary embedding's angle for the pair @i@ of
a part @d@ wide: the frequency within 4 ulp, the angle its product with
the position, cosine and sine within 4 ulp of the angle's.
-}
ropeTrig :: Double -> Int -> Int -> Double -> (Approx, Approx)
ropeTrig theta d i position = (trig cos, trig sin)
 where
  freq = theta ** negate (fromIntegral (2 * i) / fromIntegral d)
  angle = productA [constA position, settle (Approx freq (8 * unit * freq + flush))]
  trig f
    | isInfinite angle.aBound = Approx (f angle.aExact) (1 / 0)
    | otherwise =
        let y = f angle.aExact
            e = angle.aBound
         in settle (Approx y (min 2 e + 8 * unit * min 1 (abs y + e) + flush))

-- | The bounds of one head's rotation, of width @d@.
rotationBounds :: RopeSpec -> Int -> Double -> [Approx] -> [Double]
rotationBounds r d position xs = case r.ropePairs of
  RopeHalves -> map new [0 .. d - 1]
 where
  half = d `div` 2
  a = listArray (0, d - 1) xs :: Array Int Approx
  turned i = i < fromIntegral r.ropeRotated
  new j
    | j < half && turned j = let (c, s) = ropeTrig r.ropeTheta d j position in (dotA [(a ! j, c), (negA (a ! (j + half)), s)]).aBound
    | j >= half && turned (j - half) = let (c, s) = ropeTrig r.ropeTheta d (j - half) position in (dotA [(a ! j, c), (a ! (j - half), s)]).aBound
    | otherwise = (a ! j).aBound

-- | The logits as a route takes them: the largest first, ties to the lower index.
ranked :: [Approx] -> [(Int, Approx)]
ranked = sortBy (\(i, x) (j, y) -> comparing negate x.aExact y.aExact <> compare i j) . zip [0 ..]

{- | Whether the first comes before the second whatever their binary32
values within their bounds: larger, or tied at the lower index.
-}
certainlyBefore :: (Int, Approx) -> (Int, Approx) -> Bool
certainlyBefore (i, a) (j, c) = lo > hi || (lo == hi && i < j)
 where
  lo = a.aExact - a.aBound
  hi = c.aExact + c.aBound

{- | Whether a route's top @k@ and their order are the same for every
binary32 value of the logits within their bounds: each of the top before
the next, the last before every other.
-}
routeDecided :: Int -> [Approx] -> Bool
routeDecided k xs = and (zipWith certainlyBefore top (drop 1 top)) && and [certainlyBefore l t | l <- take 1 (reverse top), t <- rest]
 where
  (top, rest) = splitAt k (ranked xs)

-- | The bounds of a decided route's weights.
routeWeightBounds :: RouteWeights -> Maybe Arr -> Int -> [Approx] -> [Double]
routeWeightBounds SoftmaxOfTopK scale k xs = case top of
  [] -> []
  (_, best) : _ ->
    let es = [monotoneA 4 exp (sumA [x, negA best]) | (_, x) <- top]
        total = sumA es
     in [(productA (quotientA e total : [exactA (s ! i) | Just s <- [scale]])).aBound | ((i, _), e) <- zip top es]
 where
  top = take k (ranked xs)

argMaxDecided :: [Approx] -> Bool
argMaxDecided xs = case ranked xs of
  best : rest -> all (certainlyBefore best) rest
  [] -> True

-- | A sum of rows' elements times a factor: summed, then scaled, or scaled, then summed.
gatherA :: [Double] -> Double -> Approx
gatherA ws f = Approx byDot.aExact (max byDot.aBound bySum.aBound)
 where
  byDot = dotA [(exactA w, constA f) | w <- ws]
  bySum = productA [sumA (map exactA ws), constA f]

{- | The bounds of attention's output elements, from each key's score and,
element by element, the keys' values. Each weight passes through at most
@T + 1@ exps and products, @T + 1@ additions and two divisions, so its
logarithm is within @ρ@ of the exact one: its score's bound, the rounding
of the subtraction of the maximum (and of the scaling, for exp2), and
@12 (T + 1)@ of binary32's unit roundoff. With @p@ the exact softmax,
@g = e^ρ − 1@ and @ḡ = Σ p g@, the output is within
@(Σ p (g + ḡ) |v| + Σ p e^ρ e_v) / (1 − ḡ)@ of the exact before the
division rounds it, and weights flushed to zero lose a few of binary32's
least normal, times the largest value.
-}
attendBounds :: [Approx] -> [[Approx]] -> [Double]
attendBounds scores = map element
 where
  t = fromIntegral (length scores) :: Double
  top = maximum (-(1 / 0) : map (.aExact) scores)
  ws = [exp (s.aExact - top) | s <- scores]
  ps = map (/ sum ws) ws
  highest = maximum (0 : map mag scores)
  rhos = [s.aBound + 2 * unit * (mag s + highest) + 12 * (t + 1) * unit + flush | s <- scores]
  gs = map (\rho -> exp rho - 1) rhos
  gbar = sum (zipWith (*) ps gs)
  element vs
    | isNaN gbar || gbar >= 1 = 1 / 0
    | otherwise =
        let o = sum (zipWith (\p' v -> p' * v.aExact) ps vs)
            spread =
              (sum [p' * (g + gbar) * abs v.aExact | (p', g, v) <- zip3 ps gs vs] + sum [p' * exp rho * v.aBound | (p', rho, v) <- zip3 ps rhos vs]) / (1 - gbar)
                + 4 * t * flush * maximum (0 : map mag vs)
         in (settle (Approx o (spread + unit * (abs o + spread) + flush))).aBound

-- | The bound widened for the f64 evaluation's own rounding.
widened :: Double -> Double
widened e = e * (1 + 2 ^^ (-20 :: Int))

-- | The least and the greatest value of the cache's type a kernel's value of the element rounds to.
storable :: CacheElem -> Approx -> (Double, Double)
storable t a = (roundTo t (a.aExact - e), roundTo t (a.aExact + e))
 where
  e = widened a.aBound

-- | Rounds to the cache's element type, to nearest, ties to even, directly from the double.
roundTo :: CacheElem -> Double -> Double
roundTo = \case
  CacheBF16 -> roundBF16
  CacheF16 -> float2Double . widenF16 . narrowF16Double

{- | Whether a kernel can have stored @k@ for the element: unbounded,
anything; NaN where the exact value is; otherwise between the roundings
of the exact value less and plus its bound.
-}
mayStore :: CacheElem -> Approx -> Double -> Bool
mayStore t a k
  | isInfinite a.aBound = True
  | isNaN a.aExact = isNaN k
  | otherwise = let (lo, hi) = storable t a in lo <= k && k <= hi

-- | What the element's bound allows a store of it.
allowed :: CacheElem -> Approx -> Text
allowed t a
  | isNaN a.aExact = "the exact value is NaN"
  | lo == hi = "the exact " <> showT a.aExact <> ", within " <> showT a.aBound <> ", rounds to " <> showT lo
  | otherwise = "the exact " <> showT a.aExact <> ", within " <> showT a.aBound <> ", rounds to between " <> showT lo <> " and " <> showT hi
 where
  (lo, hi) = storable t a

{- | The evaluator's rounding of the element, with how far from it a
kernel's store can be.
-}
ownRounding :: CacheElem -> Approx -> Approx
ownRounding t a
  | isNaN a.aExact || lo == hi = Approx own 0
  | isInfinite a.aBound || isInfinite lo || isInfinite hi = Approx own (1 / 0)
  | otherwise = Approx own (max (hi - own) (own - lo))
 where
  own = roundTo t a.aExact
  (lo, hi) = storable t a

boundsArr :: [Approx] -> Arr
boundsArr = arr . map (.aBound)

unzipConcat :: [([a], [b])] -> ([a], [b])
unzipConcat ps = (concatMap fst ps, concatMap snd ps)

dot :: Arr -> Int -> Arr -> Int -> Double
dot w base x k = sum [w ! (base + i) * x ! i | i <- [0 .. k - 1]]

arr :: [Double] -> Arr
arr xs = listArray (0, length xs - 1) xs

elemsOf :: Arr -> [Double]
elemsOf a = let (lo, hi) = bounds a in [a ! i | i <- [lo .. hi]]

sliceArr :: Arr -> Int -> Int -> Arr
sliceArr a base k = listArray (0, k - 1) [a ! (base + i) | i <- [0 .. k - 1]]

size :: Shape -> Int
size = fromIntegral . (.shapeWidth)

size' :: (Integral a) => a -> Int
size' = fromIntegral

chunks :: Int -> [a] -> [[a]]
chunks k xs
  | k <= 0 = [xs]
  | null xs = []
  | otherwise = let (a, b) = splitAt k xs in a : chunks k b

headOr :: a -> [a] -> a
headOr d = \case
  x : _ -> x
  [] -> d

showT :: (Show a) => a -> Text
showT = T.pack . show
