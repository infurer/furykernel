-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | What a program means: its step evaluated in f64, every operation as
exact as a double computes it, in the order of the program.

Rounding happens only where a program stores into state of a narrower type:
a KV cache of bf16 holds each stored element rounded to nearest, ties to
even, directly from its exact value. Everything else stays a double, so a
program's kernels are compared with what their arithmetic approximates, not
with another approximation of it.

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
  evaluate,
  roundBF16,
  geluTanh,
) where

import Control.Monad (foldM, forM, unless, when)
import Data.Array.Unboxed (UArray, bounds, listArray, (!), (//))
import Data.Bits (countLeadingZeros)
import Data.Foldable (for_)
import Data.List (sortBy)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (isNothing, listToMaybe)
import Data.Ord (comparing)
import Data.Text (Text)
import Data.Text qualified as T
import Data.Word (Word32, Word64)
import FuryKernel.IR (Name)
import FuryKernel.Program.Check (CheckedProgram, checkedProgram)
import FuryKernel.Program.Graph
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
  , envCaches :: Map Name Units
  }

evaluate :: CheckedProgram -> Bindings -> Either Text Evaluation
evaluate checked b = do
  caches <- forM (Map.toList b.bindCaches) $ \(n, units) -> do
    for_ [l | Source {sourceName = n', sourceKind = SCache l _} <- p.programSources, n' == n] $ \l ->
      for_ (Map.toList units) $ \(u, xs) -> do
        unless (u >= 0 && toInteger u < toInteger (unitCount l)) $
          Left (n <> " binds " <> unitName l <> " " <> showT u <> ", of the " <> showT (unitCount l) <> " it has.")
        sized (n <> "'s " <> unitName l <> " " <> showT u) (length xs) (unitSize l)
    pure (n, Map.map arr units)
  env <- foldM node (Env Map.empty (Map.fromList caches)) p.programNodes
  let rows = map elemsOf
  outs <- forM p.programOutputs $ \o -> case Map.lookup o.outputValue env.envValues of
    Just v -> Right (o.outputName, rows v)
    Nothing -> Left ("The output " <> o.outputName <> " was never made.")
  pure
    Evaluation
      { evalValues = Map.map rows env.envValues
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

  valueOf :: Env -> ValueId -> Either Text [Arr]
  valueOf env v = maybe (Left "A value read before it is made.") Right (Map.lookup v env.envValues)

  shapeOf :: ValueId -> Shape
  shapeOf v = maybe (Shape 0 0) (.valueShape) (Map.lookup v values)

  node :: Env -> Node -> Either Text Env
  node env n = case (n.nodeOp, n.nodeOutputs) of
    (Input s, [o]) -> do
      src <- sourceOf s
      rs <- maybe (Left ("No input " <> src.sourceName <> ".")) Right (Map.lookup src.sourceName b.bindInputs)
      unless (length rs == m) $ Left (src.sourceName <> " has " <> showT (length rs) <> " rows, not " <> showT m <> ".")
      for_ rs $ \row -> sized src.sourceName (length row) (size (shapeOf o))
      put o (map arr rs)
    (Linear w _ x, [o]) -> do
      wa <- weightOf w
      xs <- valueOf env x
      let k = size (shapeOf x)
          rowsOut = size (shapeOf o)
      put o [listArray (0, rowsOut - 1) [dot wa (j * k) xr k | j <- [0 .. rowsOut - 1]] | xr <- xs]
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
      xs <- valueOf env x
      let outW = size' (shapeOf o).shapeSegment
          k = case src.sourceKind of
            STable _ [_, c] _ _ -> fromIntegral c
            _ -> 0
          shared = size (shapeOf x) == k
      rowsOut <- forM (zip idss xs) $ \(idr, xr) ->
        fmap concat . forM (zip [0 ..] (elemsOf idr)) $ \(rank, e) -> do
          let entry = round e :: Int
          when (entry < 0 || entry >= length entries) $ Left (n.nodeName <> ": expert " <> showT entry <> " is not in the table.")
          let wa = arr (entries !! entry)
              base = if shared then 0 else rank * k
          pure [dot wa (j * k) (sliceArr xr base k) k | j <- [0 .. outW - 1]]
      put o (map arr rowsOut)
    (RmsNorm eps g f x, [o]) -> do
      gamma <- traverse weightOf g
      xs <- valueOf env x
      let seg = size' (shapeOf x).shapeSegment
      put o [arr (concatMap (norm gamma seg) (chunks seg (elemsOf xr))) | xr <- xs]
     where
      norm gamma seg xs =
        let ms = sum (map (\v -> v * v) xs) / fromIntegral seg
            s = (ms + eps) ** (-0.5)
         in [v * s * maybe 1 (! j) gamma * f | (j, v) <- zip [0 ..] xs]
    (Add a c, [o]) -> do
      as <- valueOf env a
      cs <- valueOf env c
      put o [arr (zipWith (+) (elemsOf x) (elemsOf y)) | (x, y) <- zip as cs]
    (ScaleBy w x, [o]) -> do
      wa <- weightOf w
      xs <- valueOf env x
      put o [arr (map (* (wa ! 0)) (elemsOf xr)) | xr <- xs]
    (GatedGelu form x, [o]) -> do
      xs <- valueOf env x
      let seg = size' (shapeOf x).shapeSegment
          gated s = let (a, c) = splitAt (seg `div` 2) s in zipWith (\u v -> gelu form u * v) a c
      put o [arr (concatMap gated (chunks seg (elemsOf xr))) | xr <- xs]
    (GeluMul form a c, [o]) -> do
      as <- valueOf env a
      cs <- valueOf env c
      put o [arr (zipWith (\u v -> gelu form u * v) (elemsOf x) (elemsOf y)) | (x, y) <- zip as cs]
    (Rope r pos x, [o]) -> do
      ps <- int1 pos
      xs <- valueOf env x
      let d = size' (shapeOf x).shapeSegment
      put o [arr (concatMap (rotate r d (fromIntegral position)) (chunks d (elemsOf xr))) | (position, xr) <- zip ps xs]
    (Softcap c x, [o]) -> do
      xs <- valueOf env x
      put o [arr (map (\v -> c * tanh (v / c)) (elemsOf xr)) | xr <- xs]
    (Route r x, [ids, ws]) -> do
      scale <- traverse weightOf r.routeExpertScale
      xs <- valueOf env x
      let routes = map (routeRow r.routeWeights scale (fromIntegral r.routeK) . elemsOf) xs
      env' <- put ids [arr (map (fromIntegral . fst) rt) | rt <- routes]
      putIn env' ws [arr (map snd rt) | rt <- routes]
    (CombineRanks w x, [o]) -> do
      wss <- valueOf env w
      xs <- valueOf env x
      let seg = size' (shapeOf x).shapeSegment
      put o [arr (foldl1 (zipWith (+)) [map (* wr) s | (wr, s) <- zip (elemsOf wr') (chunks seg (elemsOf xr))]) | (wr', xr) <- zip wss xs]
    (KvStore (StoreAt c pos slot) x, []) -> do
      src <- sourceOf c
      layout <- case src.sourceKind of
        SCache l _ -> Right l
        _ -> Left (src.sourceName <> " is not a cache.")
      ps <- int1 pos
      seqs <- intsOf slot
      xs <- valueOf env x
      cacheUnits <- maybe (Left ("No cache " <> src.sourceName <> ".")) Right (Map.lookup src.sourceName env.envCaches)
      let d = fromIntegral (cacheHeadDim layout)
          heads = fromIntegral (cacheKvHeads layout)
      writes <- fmap concat . forM (zip3 ps seqs xs) $ \(position, sq, xr) ->
        forM [(h, i) | h <- [0 .. heads - 1], i <- [0 .. d - 1]] $ \(h, i) -> do
          (u, at) <- cacheIndex layout sq position h
          pure ((u, at + i), roundBF16 (xr ! (h * d + i)))
      checkWrites src.sourceName layout cacheUnits (map fst writes)
      pure env {envCaches = Map.insert src.sourceName (writeUnits cacheUnits writes) env.envCaches}
    (Attend a q (AttendTo k v pos slot), [o]) -> do
      kSrc <- sourceOf k
      vSrc <- sourceOf v
      layout <- case kSrc.sourceKind of
        SCache l _ -> Right l
        _ -> Left (kSrc.sourceName <> " is not a cache.")
      kUnits <- maybe (Left ("No cache " <> kSrc.sourceName <> ".")) Right (Map.lookup kSrc.sourceName env.envCaches)
      vUnits <- maybe (Left ("No cache " <> vSrc.sourceName <> ".")) Right (Map.lookup vSrc.sourceName env.envCaches)
      ps <- int1 pos
      seqs <- intsOf slot
      qs <- valueOf env q
      let d = fromIntegral a.attnHeadDim
          heads = fromIntegral a.attnHeads
          group = heads `div` fromIntegral a.attnKvHeads
      rowsOut <- forM (zip3 ps seqs qs) $ \(position, sq, qr) -> do
        let first = maybe 0 (\w -> max 0 (position - fromIntegral w + 1)) a.attnWindow
            ts = [first .. position]
        fmap concat . forM [0 .. heads - 1] $ \h -> do
          let kvh = h `div` group
              query = sliceArr qr (h * d) d
          keys <- forM ts $ \t -> do
            (u, at) <- cacheIndex layout sq t kvh
            ks <- forM [0 .. d - 1] $ \i -> found kSrc.sourceName kUnits (u, at + i)
            vs <- forM [0 .. d - 1] $ \i -> found vSrc.sourceName vUnits (u, at + i)
            pure (arr ks, arr vs)
          let scores = [a.attnScale * sum [query ! i * ks ! i | i <- [0 .. d - 1]] | (ks, _) <- keys]
              top = maximum scores
              es = map (\s -> exp (s - top)) scores
              total = sum es
          pure [sum [e / total * vs ! i | (e, (_, vs)) <- zip es keys] | i <- [0 .. d - 1]]
      put o (map arr rowsOut)
    (Gather t tok f, [o]) -> do
      wa <- weightOf t
      toks <- int1 tok
      let width = size (shapeOf o)
      put o [listArray (0, width - 1) [wa ! (token * width + j) * f | j <- [0 .. width - 1]] | token <- toks]
    (ArgMax x, [o]) -> do
      xs <- valueOf env x
      put o [arr [fromIntegral (argMax (elemsOf xr))] | xr <- xs]
    _ -> Left (n.nodeName <> ": the wrong number of values.")
   where
    put = putIn env
    putIn e o vs = Right e {envValues = Map.insert o vs e.envValues}
    found name units at = maybe (Left (unbound name at)) Right (unitAt units at)

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
