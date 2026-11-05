-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | Affine forms over integer variables, and whether a system of affine
equalities and inequalities has a solution in the integers.

'solvable' is the Omega test: it eliminates the equalities exactly, then
the inequalities' variables one at a time, exactly where every lower or
every upper bound on the variable has a coefficient of one, else through
the real and the dark shadows and, between them, the splinters. It answers
'True' when a system takes more steps or grows larger than it is willing
to work through, so 'False' always means that there is no integer
solution.
-}
module FuryKernel.Check.Affine (
  -- * Affine forms
  Affine,
  term,
  constant,
  plus,
  minus,
  times,
  constantPart,
  coefficients,
  rename,

  -- * Systems
  Constraint (..),
  equal,
  atMost,
  solvable,
  splitBy,
) where

import Control.Monad.Trans.State.Strict (State, evalState, get, put)
import Data.IntMap.Strict (IntMap)
import Data.IntMap.Strict qualified as IntMap
import Data.List (minimumBy)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Ord (comparing)
import Data.Set qualified as Set

-- | @Σ a_v · v + c@, every coefficient nonzero.
data Affine v = Affine (Map v Integer) Integer
  deriving stock (Eq, Ord, Show)

term :: v -> Affine v
term v = Affine (Map.singleton v 1) 0

constant :: Integer -> Affine v
constant = Affine Map.empty

plus :: (Ord v) => Affine v -> Affine v -> Affine v
plus (Affine a c) (Affine b d) = Affine (Map.filter (/= 0) (Map.unionWith (+) a b)) (c + d)

minus :: (Ord v) => Affine v -> Affine v -> Affine v
minus a b = plus a (times (-1) b)

times :: Integer -> Affine v -> Affine v
times 0 _ = constant 0
times k (Affine a c) = Affine (Map.map (* k) a) (k * c)

constantPart :: Affine v -> Integer
constantPart (Affine _ c) = c

coefficients :: Affine v -> Map v Integer
coefficients (Affine a _) = a

rename :: (Ord w) => (v -> w) -> Affine v -> Affine w
rename f (Affine a c) = Affine (Map.filter (/= 0) (Map.mapKeysWith (+) f a)) c

-- | An affine form that is zero, or one that is at least zero.
data Constraint v
  = EqualsZero (Affine v)
  | NonNegative (Affine v)
  deriving stock (Eq, Ord, Show)

equal :: (Ord v) => Affine v -> Affine v -> Constraint v
equal a b = EqualsZero (minus a b)

-- | @atMost a b@: @a ≤ b@.
atMost :: (Ord v) => Affine v -> Affine v -> Constraint v
atMost a b = NonNegative (minus b a)

------------------------------------------------------------------------------
-- The Omega test

-- | @Σ a_i · x_i + c@ over numbered variables, every coefficient nonzero.
data Row = Row (IntMap Integer) Integer
  deriving stock (Eq, Ord)

-- | Each variable's least and greatest value, where the rows imply one.
type Bounds = IntMap (Maybe Integer, Maybe Integer)

{- | Whether the constraints may have a solution in the integers: 'False'
only when they have none.

The digit split takes its bounds from the constraints given, propagated
once: they hold at every solution of the constraints, and those are what
every system the search derives from them is looking for.
-}
solvable :: (Ord v) => [Constraint v] -> Bool
solvable cs = evalState (equalities (boundsOf eqs ins) next eqs ins) effort
 where
  (next, _, eqs, ins) = numbered cs

-- | The constraints' variables numbered, their equalities and inequalities as rows.
numbered :: (Ord v) => [Constraint v] -> (Int, Map v Int, [Row], [Row])
numbered cs = (Map.size numbering, numbering, [row a | EqualsZero a <- cs], [row a | NonNegative a <- cs])
 where
  numbering = Map.fromList (zip (Set.toList (Set.fromList (concatMap names cs))) [0 ..])
  names = \case
    EqualsZero a -> Map.keys (coefficients a)
    NonNegative a -> Map.keys (coefficients a)
  row (Affine a c) = Row (IntMap.fromList [(numbering Map.! v, k) | (v, k) <- Map.toList a]) c

-- | What equalities and inequalities imply of each variable.
boundsOf :: [Row] -> [Row] -> Bounds
boundsOf eqs ins = propagate (ins <> concat [[r, scaleRow (-1) r] | r <- eqs])

{- | The first constraints, their equalities split by their digits by the
bounds that both sets imply: in a system that holds both sets, the split
equalities have the integer solutions the first constraints' own have.
-}
splitBy :: (Ord v) => [Constraint v] -> [Constraint v] -> [Constraint v]
splitBy cs context = [c | c@(NonNegative _) <- cs] <> map (EqualsZero . affine) (digits (boundsOf eqs ins) own)
 where
  (_, numbering, eqs, ins) = numbered (cs <> context)
  own = take (length [() | EqualsZero _ <- cs]) eqs
  names = IntMap.fromList [(i, v) | (v, i) <- Map.toList numbering]
  affine (Row a c) = Affine (Map.fromList [(names IntMap.! i, k) | (i, k) <- IntMap.toList a]) c

{- | Equalities split by their digits: where every coefficient of some
terms is a multiple of @m@ and the rest of the row (its constant included)
lies strictly between @-m@ and @m@ by the bounds given, the rest is zero,
and so are those terms. Applied over again to the parts, it takes a
row-major index apart, row from column, digit by digit.
-}
digits :: Bounds -> [Row] -> [Row]
digits known = concatMap split
 where
  split r@(Row a c) =
    case [ (Row high 0, Row low c)
         | m <- reverse (Set.toList (Set.fromList (map abs (IntMap.elems a))))
         , m > 1
         , let (high, low) = IntMap.partition ((== 0) . (`mod` m)) a
         , not (IntMap.null high)
         , not (IntMap.null low) || c /= 0
         , Just (lo, hi) <- [rangeOf (Row low c)]
         , negate m < lo
         , hi < m
         ] of
      (high, low) : _ -> split low <> split (reduce high)
      [] -> [r]
  -- The row over the gcd of its coefficients.
  reduce (Row a c) = let g = foldr gcd 0 (IntMap.elems a) in if g > 1 && c `mod` g == 0 then Row (IntMap.map (`div` g) a) (c `div` g) else Row a c
  rangeOf (Row a c) = do
    parts <- traverse (\(v, k) -> IntMap.lookup v known >>= \(lo, hi) -> (\l h -> if k > 0 then (k * l, k * h) else (k * h, k * l)) <$> lo <*> hi) (IntMap.toList a)
    pure (c + sum (map fst parts), c + sum (map snd parts))

{- | The steps the test takes before it stops and answers that a solution
may exist, and the rows it holds at most.
-}
effort, budget :: Int
effort = 20000
budget = 4096

-- | A search for a solution: the steps it has left.
type Search = State Int

-- | One step, or 'True' once the steps are spent.
step :: Search Bool -> Search Bool
step k = do
  left <- get
  if left <= 0 then pure True else put (left - 1) >> k

orElse, andThen :: Search Bool -> Search Bool -> Search Bool
orElse a b = a >>= \x -> if x then pure True else b
andThen a b = a >>= \x -> if x then b else pure False

anyOf :: (a -> Search Bool) -> [a] -> Search Bool
anyOf f = foldr (orElse . f) (pure False)

{- | Eliminates the equalities one at a time, then hands the inequalities
on; @next@ numbers a new variable.
-}
equalities :: Bounds -> Int -> [Row] -> [Row] -> Search Bool
equalities known next eqs ins
  | length ins > budget = pure True
  | otherwise = step $ case eqs of
      [] -> inequalities known next ins
      e : rest
        | [_] <- split' -> one e rest
        | otherwise -> equalities known next (split' <> rest) ins
       where
        split' = digits known [e]
 where
  one eq rest = case eq of
    Row a c
      | IntMap.null a -> if c == 0 then equalities known next rest ins else pure False
      | c `mod` g /= 0 -> pure False
      | otherwise -> case [(x, k) | (x, k) <- IntMap.toList a', abs k == 1] of
          (x, k) : _ ->
            -- x = -k · (the rest of the row), k being ±1.
            let value = scaleRow (negate k) (Row (IntMap.delete x a') c')
             in equalities known next (map (substitute x value) rest) (map (substitute x value) ins)
          [] ->
            -- Pugh's step: with m = |a_k| + 1 and a new variable σ, an
            -- equation the row implies gives x_k a coefficient of ±1;
            -- substituting it shrinks the row's coefficients.
            let (x, k) = minimumBy (comparing (abs . snd)) (IntMap.toList a')
                m = abs k + 1
                value =
                  scaleRow (signum k) $
                    Row (IntMap.insert next (negate m) (IntMap.map (`modHat` m) (IntMap.delete x a'))) (c' `modHat` m)
                sub = substitute x value
             in equalities known (next + 1) (sub (Row a' c') : map sub rest) (map sub ins)
     where
      g = foldr gcd 0 (IntMap.elems a)
      a' = IntMap.map (`div` g) a
      c' = c `div` g

-- | @a@ minus the multiple of @m@ nearest to it, ties upwards.
modHat :: Integer -> Integer -> Integer
modHat a m = a - m * ((2 * a + m) `div` (2 * m))

scaleRow :: Integer -> Row -> Row
scaleRow k (Row a c) = Row (IntMap.filter (/= 0) (IntMap.map (* k) a)) (k * c)

addRows :: Row -> Row -> Row
addRows (Row a c) (Row b d) = Row (IntMap.filter (/= 0) (IntMap.unionWith (+) a b)) (c + d)

-- | Variable @x@ replaced by the affine form @value@.
substitute :: Int -> Row -> Row -> Row
substitute x value r@(Row a c) = case IntMap.lookup x a of
  Nothing -> r
  Just k -> addRows (Row (IntMap.delete x a) c) (scaleRow k value)

{- | Inequalities @row ≥ 0@: normalized, then a variable gone, until none
is left.
-}
inequalities :: Bounds -> Int -> [Row] -> Search Bool
inequalities known next rows0
  | length rows0 > budget = pure True
  | otherwise = step $ case normalize rows0 of
      Nothing -> pure False
      Just (eqs, rows)
        | not (null eqs) -> equalities known next eqs rows
        | null rows -> pure True
        | otherwise -> eliminate known next rows

{- | Each row divided by the greatest common divisor of its coefficients,
its constant rounded down, the rows without variables dropped, and of rows
alike but for their constants the tightest kept; then two rows that pin a
form to one value become an equality. 'Nothing' if a row or a pair has no
solution.
-}
normalize :: [Row] -> Maybe ([Row], [Row])
normalize rows = do
  tight <- traverse one rows
  let best = Map.fromListWith min [(a, c) | Just (a, c) <- tight]
      opposite = IntMap.map negate
      pinned = [(a, c) | (a, c) <- Map.toList best, a < opposite a, Just c' <- [Map.lookup (opposite a) best], c + c' == 0]
      pinnedKeys = Set.fromList (concat [[a, opposite a] | (a, _) <- pinned])
  if or [c + c' < 0 | (a, c) <- Map.toList best, Just c' <- [Map.lookup (opposite a) best]]
    then Nothing
    else
      Just
        ( [Row a c | (a, c) <- pinned]
        , [Row a c | (a, c) <- Map.toList best, not (Set.member a pinnedKeys)]
        )
 where
  one (Row a c)
    | IntMap.null a = if c >= 0 then Just Nothing else Nothing
    | otherwise =
        let g = foldr gcd 0 (IntMap.elems a)
         in Just (Just (IntMap.map (`div` g) a, c `div` g))

{- | Removes one variable, by the first that applies of: dropping one
bounded on one side only, which can always be chosen far enough out; an
exact elimination that adds no rows; trying both values of one with two;
an exact elimination that at most doubles the rows; trying each value of
one with fewer than 16; and an elimination, exact or by the real and the
dark shadows and the splinters between them.
-}
eliminate :: Bounds -> Int -> [Row] -> Search Bool
eliminate known next rows
  | (y : _) <- [y | y <- vars, null (lowers y) || null (uppers y)] =
      inequalities known next [r | r <- rows, not (mentions y r)]
  | (y : _) <- [y | y <- vars, exact y, growth y <= 0] = fourierMotzkin y
  | any (\(lo, hi) -> maybe False (\l -> maybe False (< l) hi) lo) (IntMap.elems here) = pure False
  | Just (y, values) <- smallest 2 = try y values
  | (y : _) <- [y | y <- vars, exact y, growth y <= length rows] = fourierMotzkin y
  | Just (y, values) <- smallest 16 = try y values
  | (y : _) <- [y | y <- vars, exact y] = fourierMotzkin y
  | Just (y, values) <- smallest 64 = try y values
  | otherwise = fourierMotzkin (minimumBy (comparing growth) vars)
 where
  vars = Set.toList (Set.fromList [y | Row a _ <- rows, y <- IntMap.keys a])
  coef y (Row a _) = IntMap.findWithDefault 0 y a
  mentions y r = coef y r /= 0
  lowers y = [r | r <- rows, coef y r > 0]
  uppers y = [r | r <- rows, coef y r < 0]
  exact y = all ((== 1) . coef y) (lowers y) || all ((== -1) . coef y) (uppers y)
  growth y = length (lowers y) * length (uppers y) - length (lowers y) - length (uppers y)
  try y = anyOf (\v -> inequalities known next (map (substitute y (Row IntMap.empty v)) rows))
  -- The variable with the fewest values that the rows allow, if they are
  -- fewer than @n@.
  smallest n = case [(hi - lo, y, [lo .. hi]) | (y, (Just lo, Just hi)) <- IntMap.toList here, hi - lo < n] of
    [] -> Nothing
    cands -> let (_, y, vs) = minimumBy (comparing (\(w, _, _) -> w)) cands in Just (y, vs)
  -- The bounds these rows imply, which can be tighter than the system's.
  here = propagate rows
  fourierMotzkin x
    | exact x = inequalities known next (real <> others)
    | otherwise =
        inequalities known next (real <> others)
          `andThen` ( inequalities known next (dark <> others)
                        `orElse` anyOf (\e -> equalities known next [e] rows) splinters
                    )
   where
    others = [r | r <- rows, not (mentions x r)]
    pairs = [(l, u) | l <- lowers x, u <- uppers x]
    -- From a·x + p ≥ 0 and -b·x + q ≥ 0: b·p + a·q ≥ 0 over the reals,
    -- and b·p + a·q ≥ (a - 1)(b - 1) where an integer x surely fits.
    real = [combine 0 l u | (l, u) <- pairs]
    dark = [combine ((coef x l - 1) * (negate (coef x u) - 1)) l u | (l, u) <- pairs]
    combine slack l@(Row la lc) u@(Row ua uc) =
      addRows
        (scaleRow (negate (coef x u)) (Row (IntMap.delete x la) lc))
        (addRows (scaleRow (coef x l) (Row (IntMap.delete x ua) uc)) (Row IntMap.empty (negate slack)))
    -- Pugh's splinters: with m the largest coefficient of x in an upper
    -- bound, a solution the dark shadow misses has a·x = -p + i for a
    -- lower bound a·x + p ≥ 0 and some 0 ≤ i ≤ (a·m - a - m) / m.
    splinters =
      let m = maximum [negate (coef x u) | u <- uppers x]
       in [ Row la (lc - i)
          | l@(Row la lc) <- lowers x
          , let a = coef x l
          , i <- [0 .. (a * m - a - m) `div` m]
          ]

{- | Each variable's least and greatest value that the rows imply, from a
few rounds of bounding each through the others: from @k·v + rest ≥ 0@,
@k·v ≥ -max rest@.
-}
propagate :: [Row] -> Bounds
propagate rows = iterate pass IntMap.empty !! 3
 where
  pass b = foldr (tightenWith b) b rows
  tightenWith b (Row a c) b' = IntMap.foldrWithKey (\v k acc -> maybe acc (\m -> narrow v k m acc) (greatest b (Row (IntMap.delete v a) c))) b' a
  narrow v k m acc
    | k > 0 = IntMap.insertWith both v (Just (negate (m `div` k)), Nothing) acc
    | otherwise = IntMap.insertWith both v (Nothing, Just (m `div` negate k)) acc
  both (lo, hi) (lo', hi') = (pick max lo lo', pick min hi hi')
  pick f x y = case (x, y) of
    (Just p, Just q) -> Just (f p q)
    (Just p, Nothing) -> Just p
    (Nothing, q) -> q
  greatest b (Row a c) = (c +) . sum <$> traverse (\(v, k) -> IntMap.lookup v b >>= \(lo, hi) -> if k > 0 then (k *) <$> hi else (k *) <$> lo) (IntMap.toList a)
