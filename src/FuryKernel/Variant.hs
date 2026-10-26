-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | Variant spaces: one kernel family, many ways to compute it.

A family declares its axes, the choices that change how its kernels
compute but not what (rows per SIMD group, threads, pack width, split-K),
and its shape classes, the problem sizes it bakes into kernels. For each
class, 'enumerate' visits every point of the axes and prunes, in order:

1. legality: the family builds the kernel, and the gate accepts it;
2. divisibility: the family's rule, e.g. a class's dimensions by the pack
   width or by the rows of a threadgroup;
3. parallelism: at least twice as many threadgroups as the GPU has cores;
4. duplicates: points whose kernels have the same IR, names aside.

What remains are the class's variants, named @family_class_x_axiskey@, as
in @gemv_n4096_k2816_m1_x_r2_t128_p8_s1@, and @family_class@ for a family
without axes. A committed selection picks one per class
("FuryKernel.Selection").

A name joined with @_x_@ splits back at the first @_x_@ whose right side is
valid ('splitJoin'), as 'variantOfName' splits a variant's name at the
first whose right side is an axis key. 'checkSpace' refuses a space whose
names wouldn't read back, or whose classes would be misread: a family
holding @_x_@, axis prefixes or dimension names other than lowercase
letters, a prefix or a class twice, an axis without values, and batch
dimensions that aren't every class's.

Some axes decide the order of a reduction: the variants of classes that
differ only in a batch dimension must agree on those, so that a row's
result doesn't depend on the batch it ran in.
-}
module FuryKernel.Variant (
  -- * Axes and points
  Axis (..),
  Point (..),
  pointValue,
  points,
  axisKey,
  parseAxisKey,
  parseAxisKeyAs,

  -- * Shape classes
  Shape (..),
  shapeKey,
  shapeDim,

  -- * Spaces
  Profile (..),
  Space (..),
  Variant (..),
  Outcome (..),
  variantName,
  variantOfName,
  splitJoin,
  checkSpace,
  consider,
  enumerate,
  variants,
  launchGroups,
) where

import Data.Char (isAsciiLower, isDigit)
import Data.Either (isRight)
import Data.List (find)
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe, isNothing, listToMaybe)
import Data.Text (Text)
import Data.Text qualified as T
import FuryKernel.Build (Kernel)
import FuryKernel.Check (Checked, check, checkedIR)
import FuryKernel.Check.Launch (kernelBox, launchSize, lowest)
import FuryKernel.Finding (Finding)
import FuryKernel.IR (Name)
import FuryKernel.IR qualified as IR
import FuryKernel.Target (Target)
import Numeric.Natural (Natural)

------------------------------------------------------------------------------
-- Axes and points

-- | One choice of a family's, and the values tuning tries.
data Axis = Axis
  { axisName :: Text
  -- ^ For messages, as @split_k@.
  , axisPrefix :: Text
  -- ^ Lowercase letters, in variant names, as @s@ in @s4@.
  , axisValues :: [Natural]
  , axisOrder :: Bool
  -- ^ Whether the axis decides the order of a reduction.
  }
  deriving stock (Eq, Show)

-- | A value for every axis, by prefix, in the axes' order.
newtype Point = Point [(Text, Natural)]
  deriving stock (Eq, Ord, Show)

pointValue :: Text -> Point -> Maybe Natural
pointValue prefix (Point vs) = lookup prefix vs

-- | Every point, the first axis varying slowest.
points :: [Axis] -> [Point]
points axes = map Point (mapM (\a -> [(a.axisPrefix, v) | v <- a.axisValues]) axes)

-- | @r2_t128_p8_s1@.
axisKey :: Point -> Text
axisKey (Point vs) = T.intercalate "_" [prefix <> T.pack (show v) | (prefix, v) <- vs]

{- | A point from its axis key: every axis, in order, with one of its
values, written as 'axisKey' writes it. A family without axes has the empty
key.
-}
parseAxisKey :: [Axis] -> Text -> Either Text Point
parseAxisKey = parseAxisKeyAs "The axis key"

-- | 'parseAxisKey', its messages naming the key as @what@ does.
parseAxisKeyAs :: Text -> [Axis] -> Text -> Either Text Point
parseAxisKeyAs what axes key
  | length parts /= length axes =
      Left (what <> " " <> key <> " has " <> showT (length parts) <> " parts for " <> showT (length axes) <> " axes.")
  | otherwise = Point <$> traverse part (zip axes parts)
 where
  parts = if T.null key then [] else T.splitOn "_" key
  part (a, p) = case T.stripPrefix a.axisPrefix p of
    Just digits | not (T.null digits) && T.all isDigit digits -> value a (read (T.unpack digits)) digits
    _ -> Left (what <> " " <> key <> " has " <> p <> " where " <> a.axisName <> " (" <> a.axisPrefix <> ") belongs.")
  value a v digits
    | showT v /= digits = Left (what <> " " <> key <> " writes " <> a.axisName <> " " <> showT v <> " as " <> digits <> ".")
    | v `elem` a.axisValues = Right (a.axisPrefix, v)
    | otherwise = Left (what <> " " <> key <> " gives " <> a.axisName <> " " <> digits <> ", which is not one of its values.")

------------------------------------------------------------------------------
-- Shape classes

-- | The dimensions a family bakes into a kernel, by name, as @[("n", 4096)]@.
newtype Shape = Shape [(Text, Natural)]
  deriving stock (Eq, Ord, Show)

-- | @n4096_k2816_m1@.
shapeKey :: Shape -> Text
shapeKey (Shape ds) = T.intercalate "_" [d <> T.pack (show v) | (d, v) <- ds]

shapeDim :: Text -> Shape -> Maybe Natural
shapeDim d (Shape ds) = lookup d ds

------------------------------------------------------------------------------
-- Spaces

-- | What tuning measured of a GPU that pruning uses: its cores.
newtype Profile = Profile {profileCores :: Natural}
  deriving stock (Eq, Show)

-- | A family's variants for target @t@.
data Space t = Space
  { spaceFamily :: Name
  , spaceAxes :: [Axis]
  , spaceShapes :: [Shape]
  , spaceBatch :: [Text]
  -- ^ The shape dimensions that are batch sizes.
  , spaceDivides :: Shape -> Point -> [Text]
  -- ^ Why a point doesn't divide a shape; nothing when it does.
  , spaceKernel :: Name -> Shape -> Point -> Either [Finding] (Kernel t)
  -- ^ The kernel with the given name, or why the point is illegal.
  }

data Variant t = Variant
  { variantShape :: Shape
  , variantPoint :: Point
  , variantKernel :: Checked t
  , variantGroups :: Natural
  -- ^ Threadgroups of the smallest launch.
  }

-- | What enumeration made of a point.
data Outcome t
  = Kept (Variant t)
  | Illegal [Finding]
  | Indivisible [Text]
  | -- | The threadgroups of the smallest launch.
    TooFewGroups Natural
  | -- | The kept point with the same kernel.
    SameAs Point

{- | @family_class_x_axiskey@, the name of the point's kernel for a class:
@family_class@ for a family without axes, and @family_x_axiskey@ for a
class without dimensions.
-}
variantName :: Space t -> Shape -> Point -> Name
variantName space shape point = space.spaceFamily <> classPart <> keyPart
 where
  classPart = if T.null (shapeKey shape) then "" else "_" <> shapeKey shape
  keyPart = if T.null (axisKey point) then "" else "_x_" <> axisKey point

-- | 'variantName''s inverse: the class and point whose kernel a name names.
variantOfName :: Space t -> Name -> Maybe (Shape, Point)
variantOfName space name = do
  point <- case space.spaceAxes of
    [] -> Just (Point [])
    axes -> do
      (_, key) <- splitJoin (isRight . parseAxisKey axes) name
      either (const Nothing) Just (parseAxisKey axes key)
  shape <- find (\s -> variantName space s point == name) space.spaceShapes
  pure (shape, point)

{- | A name joined with @_x_@, split back at the first @_x_@ whose right side
@valid@ takes: a left side that holds @_x_@ or ends in @_x@ splits back,
as long as what follows its own @_x_@ isn't valid.
-}
splitJoin :: (Text -> Bool) -> Text -> Maybe (Text, Text)
splitJoin valid name =
  listToMaybe
    [ (T.take i name, right)
    | i <- [0 .. T.length name - 3]
    , T.take 3 (T.drop i name) == "_x_"
    , let right = T.drop (i + 3) name
    , valid right
    ]

{- | What would make the space's names ambiguous or its classes misread:
nothing for a space whose variants' names split back.
-}
checkSpace :: Space t -> [Text]
checkSpace space =
  ["The family " <> fam <> " holds _x_, which joins a variant's name to its axis key." | "_x_" `T.isInfixOf` fam]
    <> ["The axis " <> a.axisName <> "'s prefix " <> showT a.axisPrefix <> " is not lowercase letters." | a <- space.spaceAxes, not (letters a.axisPrefix)]
    <> ["The axes " <> T.intercalate " and " names <> " share the prefix " <> p <> "." | (p, names) <- repeated [(a.axisPrefix, a.axisName) | a <- space.spaceAxes]]
    <> ["The axis " <> a.axisName <> " has no values." | a <- space.spaceAxes, null a.axisValues]
    <> [ "The class " <> shapeKey s <> "'s dimension " <> showT d <> " is not lowercase letters."
       | s@(Shape ds) <- space.spaceShapes
       , (d, _) <- ds
       , not (letters d)
       ]
    <> [ "The class " <> shapeKey s <> " gives " <> d <> " " <> showT (length vs) <> " times."
       | s@(Shape ds) <- space.spaceShapes
       , (d, vs) <- repeated ds
       ]
    <> ["The class " <> k <> " is given " <> showT (length ks) <> " times." | (k, ks) <- repeated [(shapeKey s, ()) | s <- space.spaceShapes]]
    <> [ "The batch dimension " <> b <> " is not a dimension of the class " <> shapeKey s <> "."
       | b <- space.spaceBatch
       , s <- space.spaceShapes
       , isNothing (shapeDim b s)
       ]
 where
  fam = space.spaceFamily
  letters t = not (T.null t) && T.all isAsciiLower t
  -- The keys given more than once, in order, with what each was given with.
  repeated :: (Ord k) => [(k, v)] -> [(k, [v])]
  repeated kvs = [(k, vs) | (k, vs@(_ : _ : _)) <- Map.toList (Map.fromListWith (flip (<>)) [(k, [v]) | (k, v) <- kvs])]

-- | A point for a class, pruned by everything but duplicates.
consider :: (Target t) => Profile -> Space t -> Shape -> Point -> Outcome t
consider profile space shape point =
  case space.spaceKernel (variantName space shape point) shape point >>= check of
    Left findings -> Illegal findings
    Right checked
      | reasons@(_ : _) <- space.spaceDivides shape point -> Indivisible reasons
      | groups < 2 * profile.profileCores -> TooFewGroups groups
      | otherwise -> Kept (Variant shape point checked groups)
     where
      groups = launchGroups (checkedIR checked)

-- | Every point of a class and what pruning made of it.
enumerate :: (Target t) => Profile -> Space t -> Shape -> [(Point, Outcome t)]
enumerate profile space shape = go Map.empty (points space.spaceAxes)
 where
  go _ [] = []
  go seen (p : ps) = case consider profile space shape p of
    Kept v
      | Just first <- Map.lookup (bodyText v) seen -> (p, SameAs first) : go seen ps
      | otherwise -> (p, Kept v) : go (Map.insert (bodyText v) p seen) ps
    outcome -> (p, outcome) : go seen ps
  -- The kernel's IR under the family's name, as text, so that only the body counts.
  bodyText :: Variant t -> Text
  bodyText v = IR.renderKernel (checkedIR v.variantKernel) {IR.kernelName = space.spaceFamily}

-- | The kept variants of a class, in enumeration order.
variants :: (Target t) => Profile -> Space t -> Shape -> [Variant t]
variants profile space shape = [v | (_, Kept v) <- enumerate profile space shape]

{- | Threadgroups in the grid's smallest launch: its least value over the
legal values of the dimensions, from each one's least up, or where the
search runs out of effort the least that interval arithmetic leaves
possible ("FuryKernel.Check.Launch"); 0 for a grid the gate refuses.
-}
launchGroups :: IR.Kernel -> Natural
launchGroups k = fromMaybe 0 $ do
  box <- kernelBox k
  grid <- launchSize (Map.keysSet box) k.kernelGroups
  fromInteger <$> lowest box grid

showT :: (Show a) => a -> Text
showT = T.pack . show
