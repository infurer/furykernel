-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | Launch sizes over the legal values of the dimensions.

A dimension takes the values of its range that are multiples of its
divisor and that a u32 holds: from the least of them to the greatest, a
divisor apart. A launch size (the grid or a buffer's extent) is an expression over the
dimensions and u32 literals in @+ - * / %@, which the host computes before
a launch.

'checkLaunchSizes' refuses a dimension without legal values, and a launch
size that, for some legal values of its dimensions, leaves u32 at some
step or divides by zero. Interval arithmetic bounds every step over a box
of legal values; a box it leaves open is split in two, and the halves in
turn, until every part is settled or a single point, where intervals are
exact. So a launch size is refused with the values that break it, or
because 'effort' boxes could not show that none do. A launch size the gate
accepts has one value at every legal point, whether computed in u32,
which wraps, in u64 or in the integers.

The same search finds a launch size's least value ('lowest') and shows one
launch size no larger than another ('noLarger'), where their polynomial
normal forms don't show it first.
-}
module FuryKernel.Check.Launch (
  -- * Legal values
  Values (..),
  legalValues,
  Box,
  kernelBox,
  pointText,
  intersect,
  covers,

  -- * Launch sizes
  Size,
  launchSize,
  sizeText,
  Failure (..),
  breachText,
  Verdict (..),
  effort,
  safety,
  lowest,
  noLarger,
  checkLaunchSizes,
) where

import Data.Either (rights)
import Data.List (sortOn)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Ord (Down (..))
import Data.Set (Set)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as T
import FuryKernel.Finding (Finding (..), Rule (..))
import FuryKernel.IR (DimRange (..), Expr (..), Literal (..), Name, Param (..), ParamRole (..))
import FuryKernel.IR qualified as IR

------------------------------------------------------------------------------
-- Legal values

-- | A dimension's legal values: from the least to the greatest, a step apart.
data Values = Values
  { valuesLeast :: Integer
  , valuesStep :: Integer
  , valuesGreatest :: Integer
  }
  deriving stock (Eq, Show)

-- | The legal values of a range, if it has any.
legalValues :: DimRange -> Maybe Values
legalValues r
  | m == 0 || least > greatest = Nothing
  | otherwise = Just (Values least m greatest)
 where
  m = toInteger r.dimMultipleOf
  least = (toInteger r.dimMin + m - 1) `div` m * m
  greatest = min (toInteger r.dimMax) maxU32 `div` m * m

count :: Values -> Integer
count v = (v.valuesGreatest - v.valuesLeast) `div` v.valuesStep + 1

-- | Some dimensions' legal values, or a part of them: a box of points.
type Box = Map Name Values

-- | The legal values of every dimension of a kernel, if each has some.
kernelBox :: IR.Kernel -> Maybe Box
kernelBox k = Map.fromList <$> sequence [(p.paramName,) <$> legalValues r | p <- k.kernelParams, Dim r <- [p.paramRole]]

-- | @m = 1, n = 9@.
pointText :: Map Name Integer -> Text
pointText pt = T.intercalate ", " [n <> " = " <> showT v | (n, v) <- Map.toList pt]

leastPoint, greatestPoint :: Box -> Map Name Integer
leastPoint = Map.map (.valuesLeast)
greatestPoint = Map.map (.valuesGreatest)

{- | The box in two, the lower half first, along the dimension with the most
values; nothing for a point.
-}
split :: Box -> Maybe (Box, Box)
split box = case sortOn (Down . count . snd) [(n, v) | (n, v) <- Map.toList box, count v > 1] of
  [] -> Nothing
  (n, v) : _ ->
    let mid = v.valuesLeast + (count v - 1) `div` 2 * v.valuesStep
     in Just (Map.insert n v {valuesGreatest = mid} box, Map.insert n v {valuesLeast = mid + v.valuesStep} box)

------------------------------------------------------------------------------
-- Launch sizes

-- | A launch size: u32 literals and dimensions in @+ - * / %@.
data Size
  = Lit Integer
  | Var Name
  | Op Arith Size Size
  deriving stock (Eq, Show)

data Arith = Plus | Minus | Times | Over | Modulo
  deriving stock (Eq, Show)

-- | An expression as a launch size over these dimensions, if it is one.
launchSize :: Set Name -> Expr -> Maybe Size
launchSize dims = \case
  ELit (LU32 n) -> Just (Lit (toInteger n))
  EParam n | n `Set.member` dims -> Just (Var n)
  EBin op a b -> Op <$> arith op <*> launchSize dims a <*> launchSize dims b
  _ -> Nothing
 where
  arith = \case
    IR.Add -> Just Plus
    IR.Sub -> Just Minus
    IR.Mul -> Just Times
    IR.Div -> Just Over
    IR.Rem -> Just Modulo
    _ -> Nothing

-- | @(n + 63) / 64 * 8@.
sizeText :: Size -> Text
sizeText = go (0 :: Int)
 where
  go p = \case
    Lit n -> showT n
    Var n -> n
    Op o a b ->
      let q = if o == Plus || o == Minus then 6 else 7
          t = go q a <> " " <> symbol o <> " " <> go (q + 1) b
       in if q < p then "(" <> t <> ")" else t
  symbol = \case
    Plus -> "+"
    Minus -> "-"
    Times -> "*"
    Over -> "/"
    Modulo -> "%"

dimsOf :: Size -> [Name]
dimsOf = \case
  Lit _ -> []
  Var n -> [n]
  Op _ a b -> dimsOf a <> dimsOf b

-- | The box's dimensions that these sizes use.
over :: [Size] -> Box -> Box
over sizes box = Map.restrictKeys box (Set.fromList (concatMap dimsOf sizes))

maxU32 :: Integer
maxU32 = 4294967295

{- | Bounds on a launch size over a box, if they keep every step of it in
u32 and every divisor above 0. A difference is at least 0 where intervals
say so or the operands' normal forms do.
-}
bounds :: Box -> Size -> Maybe (Integer, Integer)
bounds box = go
 where
  go = \case
    Lit n -> Just (n, n)
    Var n -> (\v -> (v.valuesLeast, v.valuesGreatest)) <$> Map.lookup n box
    Op o a b -> do
      (a1, a2) <- go a
      (b1, b2) <- go b
      (lo, hi) <- case o of
        Plus -> Just (a1 + b1, a2 + b2)
        Minus
          | a1 - b2 < 0 && atLeast a b -> Just (0, a2 - b1)
          | otherwise -> Just (a1 - b2, a2 - b1)
        Times -> Just (a1 * b1, a2 * b2)
        Over | b1 > 0 -> Just (a1 `div` b2, a2 `div` b1)
        Modulo
          | b1 > 0 && a2 < b1 -> Just (a1, a2)
          | b1 > 0 && b1 == b2 && a1 `div` b1 == a2 `div` b1 -> Just (a1 `mod` b1, a2 `mod` b1)
          | b1 > 0 -> Just (0, min a2 (b2 - 1))
        _ -> Nothing
      if lo >= 0 && hi <= maxU32 then Just (lo, hi) else Nothing

-- | What breaks a launch size at a point.
data Failure
  = -- | A step leaves u32: the step, and its value.
    Leaves Size Integer
  | -- | A divisor is 0.
    ByZero Size
  | -- | One launch size is larger than the other: their values.
    Larger Integer Integer
  | -- | None of the boxes takes the point.
    Outside
  deriving stock (Eq, Show)

-- | A launch size at a point, exactly, step by step as the host computes it.
valueAt :: Map Name Integer -> Size -> Either Failure Integer
valueAt pt = go
 where
  go s = case s of
    Lit n -> Right n
    Var n -> Right (Map.findWithDefault 0 n pt)
    Op o a b -> do
      x <- go a
      y <- go b
      v <- case o of
        Plus -> Right (x + y)
        Minus -> Right (x - y)
        Times -> Right (x * y)
        Over
          | y == 0 -> Left (ByZero b)
          | otherwise -> Right (x `div` y)
        Modulo
          | y == 0 -> Left (ByZero b)
          | otherwise -> Right (x `mod` y)
      if v < 0 || v > maxU32 then Left (Leaves s v) else Right v

-- | What a search over a box found.
data Verdict
  = Holds
  | -- | The point where it breaks, the least the search came to.
    Breaks (Map Name Integer) Failure
  | -- | 'effort' boxes did not settle it.
    Open
  deriving stock (Eq, Show)

-- | The most boxes a search examines.
effort :: Int
effort = 4096

{- | Whether a launch size stays in u32 at every point of a box, dividing by
no zero.
-}
safety :: Box -> Size -> Verdict
safety box s = go 0 [over [s] box]
 where
  go :: Int -> [Box] -> Verdict
  go _ [] = Holds
  go n (b : bs)
    | n >= effort = Open
    | Just _ <- bounds b s = go (n + 1) bs
    | Just (l, h) <- split b = go (n + 1) (l : h : bs)
    | otherwise = case valueAt (leastPoint b) s of
        Left f -> Breaks (leastPoint b) f
        Right _ -> go (n + 1) bs

{- | A launch size's least value over a box, if it stays in u32 there; where
'effort' boxes don't find it, the least that interval arithmetic leaves
possible.
-}
lowest :: Box -> Size -> Maybe Integer
lowest box s = case valueAt (leastPoint start) s of
  Left _ -> Nothing
  Right v -> go v 0 [start]
 where
  start = over [s] box
  go :: Integer -> Int -> [Box] -> Maybe Integer
  go best _ [] = Just best
  go best n bs@(b : rest)
    | n >= effort = Just (minimum (best : map (maybe 0 fst . (`bounds` s)) bs))
    | Just (lo, _) <- bounds b s, lo >= best = go best (n + 1) rest
    | Just (l, h) <- split b = go (minimum (best : corners b)) (n + 1) (l : h : rest)
    | otherwise = case valueAt (leastPoint b) s of
        Right v -> go (min best v) (n + 1) rest
        Left _ -> Nothing
  corners b = rights [valueAt (leastPoint b) s, valueAt (greatestPoint b) s]

{- | Whether launch size @a@ is no larger than @b@ at every point of a box
where both stay in u32: by their polynomial normal forms, else by interval
arithmetic over parts of the box.
-}
noLarger :: Box -> Size -> Size -> Verdict
noLarger box a b
  | atLeast b a = Holds
  | otherwise = go 0 [over [a, b] box]
 where
  go :: Int -> [Box] -> Verdict
  go _ [] = Holds
  go n (x : xs)
    | n >= effort = Open
    | Just (_, ahi) <- bounds x a, Just (blo, _) <- bounds x b, ahi <= blo = go (n + 1) xs
    | Just (l, h) <- split x = go (n + 1) (l : h : xs)
    | otherwise = case (valueAt pt a, valueAt pt b) of
        (Left f, _) -> Breaks pt f
        (_, Left f) -> Breaks pt f
        (Right va, Right vb)
          | va > vb -> Breaks pt (Larger va vb)
          | otherwise -> go (n + 1) xs
   where
    pt = leastPoint x

-- | The points two boxes share, if they share any.
intersect :: Box -> Box -> Maybe Box
intersect a b = sequence (Map.intersectionWith both a b)
 where
  -- Legal values are multiples of their step, from the least up.
  both v w =
    let step = lcm v.valuesStep w.valuesStep
        least = (max v.valuesLeast w.valuesLeast + step - 1) `div` step * step
        greatest = min v.valuesGreatest w.valuesGreatest `div` step * step
     in if least > greatest then Nothing else Just (Values least step greatest)

-- | Whether every point of a box lies in another, which takes its dimensions.
within :: Box -> Box -> Bool
within inner outer = and [maybe False (inside v) (Map.lookup n outer) | (n, v) <- Map.toList inner]
 where
  inside v o =
    v.valuesLeast >= o.valuesLeast
      && v.valuesGreatest <= o.valuesGreatest
      && (v.valuesLeast - o.valuesLeast) `mod` o.valuesStep == 0
      && (v.valuesLeast == v.valuesGreatest || v.valuesStep `mod` o.valuesStep == 0)

-- | Whether every point of a box lies in one of some boxes.
covers :: [Box] -> Box -> Verdict
covers boxes = go 0 . pure
 where
  go :: Int -> [Box] -> Verdict
  go _ [] = Holds
  go n (b : bs)
    | n >= effort = Open
    | any (b `within`) boxes = go (n + 1) bs
    | Just (l, h) <- split b = go (n + 1) (l : h : bs)
    | otherwise = Breaks (leastPoint b) Outside

------------------------------------------------------------------------------
-- Polynomial normal forms

{- | A polynomial in dimensions, quotients and remainders: each monomial's
atoms with their powers, and its coefficient, none 0.
-}
type Poly = Map (Map Atom Int) Integer

data Atom
  = ADim Name
  | AQuot Poly Poly
  | ARem Poly Poly
  deriving stock (Eq, Ord)

{- | The polynomial of a launch size. Where no step of it leaves u32, its
atoms are at least 0, and so is a polynomial with positive coefficients.
-}
normal :: Size -> Poly
normal = \case
  Lit n -> Map.filter (/= 0) (Map.singleton Map.empty n)
  Var n -> atom (ADim n)
  Op o x y -> case o of
    Plus -> add (normal x) (normal y)
    Minus -> add (normal x) (scale (-1) (normal y))
    Times -> Map.filter (/= 0) (Map.fromListWith (+) [(Map.unionWith (+) m1 m2, c1 * c2) | (m1, c1) <- Map.toList (normal x), (m2, c2) <- Map.toList (normal y)])
    Over -> atom (AQuot (normal x) (normal y))
    Modulo -> atom (ARem (normal x) (normal y))
 where
  atom t = Map.singleton (Map.singleton t 1) 1

{- | Whether @a@ is at least @b@ by their normal forms, wherever no step of
either leaves u32: their difference has positive coefficients.
-}
atLeast :: Size -> Size -> Bool
atLeast a b = all (> 0) (add (normal a) (scale (-1) (normal b)))

add :: Poly -> Poly -> Poly
add p q = Map.filter (/= 0) (Map.unionWith (+) p q)

scale :: Integer -> Poly -> Poly
scale k = Map.filter (/= 0) . Map.map (* k)

------------------------------------------------------------------------------
-- The gate's check

{- | Every dimension's range with legal values, and every launch size in
u32 at every step, dividing by no zero, for every legal value of its
dimensions. A launch size of more than dimensions and u32 literals in
@+ - * / %@ is the typing check's.
-}
checkLaunchSizes :: IR.Kernel -> [Finding]
checkLaunchSizes k =
  [ Finding LaunchSizes ("The range of " <> p.paramName <> ", " <> rangeText r <> ", " <> why)
  | p <- k.kernelParams
  , Dim r <- [p.paramRole]
  , Just why <- [emptiness r]
  ]
    <> [ Finding LaunchSizes (what (sizeText s) <> m)
       | (what, e) <- sizes
       , Just s <- [launchSize (Set.fromList (map fst dims)) e]
       , all (`Map.member` box) (dimsOf s)
       , Just m <- [problem s (safety box s)]
       ]
 where
  dims = [(p.paramName, r) | p <- k.kernelParams, Dim r <- [p.paramRole]]
  box = Map.fromList [(n, v) | (n, r) <- dims, Just v <- [legalValues r]]
  emptiness r
    | r.dimMultipleOf == 0 = Just "has no divisor."
    | Nothing <- legalValues r = Just "holds no value."
    | toInteger r.dimMax > maxU32 = Just ("reaches past what a u32 holds, " <> showT maxU32 <> ".")
    | otherwise = Nothing
  rangeText r = showT r.dimMin <> ".." <> showT r.dimMax <> " multiple of " <> showT r.dimMultipleOf
  sizes =
    [(("The grid size " <>), k.kernelGroups)]
      <> [(\t -> "The extent of " <> b.bufName <> ", " <> t <> ",", b.bufExtent) | b <- k.kernelBuffers]
  problem s = \case
    Holds -> Nothing
    Breaks pt f -> Just (breachText pt f)
    Open -> Just (" could not be shown to stay in u32, dividing by no 0, for every legal value of " <> andList (Set.toList (Set.fromList (dimsOf s))) <> ": " <> showT effort <> " boxes of them leave it open.")

-- | What breaks where, for messages: @ leaves u32 at n = 9: 8 - n is -1.@
breachText :: Map Name Integer -> Failure -> Text
breachText pt = \case
  Leaves s v -> " leaves u32" <> at <> ": " <> sizeText s <> " is " <> showT v <> "."
  ByZero (Lit _) -> " divides by 0" <> at <> "."
  ByZero d -> " divides by 0" <> at <> ": " <> sizeText d <> " is 0."
  Larger x y -> " is larger" <> at <> ": " <> showT x <> " against " <> showT y <> "."
  Outside -> " is outside every box" <> at <> "."
 where
  at = if Map.null pt then "" else " at " <> pointText pt

-- | @k, m and n@.
andList :: [Text] -> Text
andList = \case
  [] -> ""
  [x] -> x
  xs -> T.intercalate ", " (init xs) <> " and " <> last xs

showT :: (Show a) => a -> Text
showT = T.pack . show
