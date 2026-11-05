-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE AllowAmbiguousTypes #-}

{- | Selections: which variant of a family each shape class runs on a
target, as tuning chose and a repository commits.

The format, version 1:

> {
>   "format": "furykernel-selection",
>   "version": 1,
>   "family": "gemv",
>   "target": "AppleGPU9",
>   "profile": {
>     "cores": 40
>   },
>   "classes": {
>     "n128_k2816_m1": {
>       "variant": "r1_t128_p8_s4",
>       "ir_hash": "0x5b1e46d0c3a9f2e7"
>     }
>   }
> }

The profile is the GPU's as tuning measured it ('Profile'). Each class's
key is its 'shapeKey', its pick the 'axisKey' of its variant and the IR
hash ('FuryKernel.Abi.irHash') of the variant's kernel when it was picked;
classes are written in key order, each once. A selection is valid for a
space and a GPU profile when the space passes
'FuryKernel.Variant.checkSpace', the selection was made for that profile,
names the space's family and target, picks a variant for every class of
the space and for no other, every
pick survives enumeration's pruning but for duplicates and is the kernel
it was picked as, and the classes that differ only in batch dimensions
agree on every axis that decides a reduction's order.
-}
module FuryKernel.Selection (
  Selection (..),
  Pick (..),
  selectionFormat,
  renderSelection,
  parseSelection,
  validateSelection,
) where

import Control.Monad (unless)
import Data.Either (partitionEithers)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text qualified as T
import Data.Word (Word64)
import FuryKernel.Abi (hex64, irHash)
import FuryKernel.Finding (renderFinding)
import FuryKernel.IR (Name)
import FuryKernel.Json (Json (..), parseJson, renderJson)
import FuryKernel.Target (Target, TargetCaps (..), targetCaps)
import FuryKernel.Variant

data Selection = Selection
  { selectionFamily :: Name
  , selectionTarget :: Text
  , selectionProfile :: Profile
  -- ^ The GPU's as tuning measured it, which pruning depends on.
  , selectionClasses :: Map Text Pick
  -- ^ Each class's pick, by its shape key.
  }
  deriving stock (Eq, Show)

-- | A class's pick: its variant's axis key, and its kernel's IR hash then.
data Pick = Pick
  { pickAxes :: Text
  , pickIrHash :: Word64
  }
  deriving stock (Eq, Show)

-- | The value of the @format@ field.
selectionFormat :: Text
selectionFormat = "furykernel-selection"

renderSelection :: Selection -> Text
renderSelection s =
  renderJson $
    JObject
      [ ("format", JString selectionFormat)
      , ("version", JInt 1)
      , ("family", JString s.selectionFamily)
      , ("target", JString s.selectionTarget)
      , ("profile", JObject [("cores", JInt (toInteger s.selectionProfile.profileCores))])
      , ("classes", JObject [(c, pick p) | (c, p) <- Map.toList s.selectionClasses])
      ]
 where
  pick p = JObject [("variant", JString p.pickAxes), ("ir_hash", JString (hex64 p.pickIrHash))]

-- | A selection from its JSON: the format only, not what it selects.
parseSelection :: Text -> Either [Text] Selection
parseSelection text = case parseJson text of
  Left e -> Left [e]
  Right (JObject fields) -> do
    -- The format and its version first: another version may have other fields.
    format <- field fields "format" str
    unless (format == selectionFormat) $ Left ["The format is " <> format <> ", not " <> selectionFormat <> "."]
    version <- field fields "version" int
    unless (version == 1) $ Left ["Version " <> T.pack (show version) <> " of the format; this reads version 1."]
    let known = ["format", "version", "family", "target", "profile", "classes"]
        unknown = [k | (k, _) <- fields, k `notElem` known]
    unless (null unknown) $ Left ["An unknown field: " <> k <> "." | k <- unknown]
    family <- field fields "family" str
    target <- field fields "target" str
    profile <- field fields "profile" $ \case
      JObject [("cores", JInt n)] | n >= 0 -> Just (Profile (fromInteger n))
      _ -> Nothing
    classes <- field fields "classes" $ \case
      JObject cs -> case partitionEithers [maybe (Left c) (Right . (c,)) (asPick v) | (c, v) <- cs] of
        ([], picks) -> Just (Map.fromList picks)
        _ -> Nothing
      _ -> Nothing
    pure (Selection family target profile classes)
  Right _ -> Left ["A selection is a JSON object."]
 where
  field fields name decode = case lookup name fields of
    Nothing -> Left ["The field " <> name <> " is missing."]
    Just j -> maybe (Left ["The field " <> name <> " has the wrong type."]) Right (decode j)
  str = asString
  int = \case
    JInt n -> Just n
    _ -> Nothing
  asString = \case
    JString t -> Just t
    _ -> Nothing
  asPick = \case
    JObject [("variant", JString v), ("ir_hash", JString h)] -> Pick v <$> hash h
    _ -> Nothing
  -- 0x and 16 lowercase hex digits, as 'hex64' writes them.
  hash h = case T.stripPrefix "0x" h of
    Just digits | T.length digits == 16, T.all (`elem` ("0123456789abcdef" :: String)) digits -> Just (T.foldl' (\n d -> n * 16 + fromIntegral (hexDigit d)) 0 digits)
    _ -> Nothing
  hexDigit d = length (takeWhile (/= d) "0123456789abcdef")

{- | The selected variant of every class, in the space's class order, or
everything that is wrong with the selection.
-}
validateSelection :: forall t. (Target t) => Profile -> Space t -> Selection -> Either [Text] [Variant t]
validateSelection profile space sel
  | null problems = Right [v | (_, Right v) <- picked]
  | otherwise = Left problems
 where
  caps = targetCaps @t
  problems =
    checkSpace space
      <> [ "The selection was made for a GPU of " <> T.pack (show sel.selectionProfile.profileCores) <> " cores, not " <> T.pack (show profile.profileCores) <> "."
         | sel.selectionProfile /= profile
         ]
      <> ["The selection is for the family " <> sel.selectionFamily <> ", not " <> space.spaceFamily <> "." | sel.selectionFamily /= space.spaceFamily]
      <> ["The selection is for the target " <> sel.selectionTarget <> ", not " <> caps.capName <> "." | sel.selectionTarget /= caps.capName]
      <> ["No variant for the class " <> shapeKey s <> "." | s <- space.spaceShapes, shapeKey s `Map.notMember` sel.selectionClasses]
      <> ["The class " <> c <> " is not one of the family's." | c <- Map.keys sel.selectionClasses, c `notElem` map shapeKey space.spaceShapes]
      <> [e | (_, Left e) <- picked]
      <> orderProblems
  picks = sel.selectionClasses
  picked =
    [ (shape, pick shape key)
    | shape <- space.spaceShapes
    , Just key <- [Map.lookup (shapeKey shape) picks]
    ]
  pick shape chosen = do
    point <- parseAxisKeyAs ("The class " <> shapeKey shape <> "'s axis key") space.spaceAxes chosen.pickAxes
    let name = variantName space shape point
    case consider profile space shape point of
      Kept v
        | irHash v.variantKernel /= chosen.pickIrHash ->
            Left (name <> " was picked as the kernel of IR hash " <> hex64 chosen.pickIrHash <> ", and is now " <> hex64 (irHash v.variantKernel) <> ".")
        | otherwise -> Right v
      Illegal fs -> Left (name <> " is illegal: " <> T.intercalate " " (map renderFinding fs))
      Indivisible reasons -> Left (name <> " does not divide its class: " <> T.intercalate " " reasons)
      TooFewGroups n ->
        Left (name <> " launches " <> T.pack (show n) <> " threadgroups, fewer than twice the " <> T.pack (show profile.profileCores) <> " cores.")
      SameAs p -> Left (name <> " is the same kernel as " <> axisKey p <> ".")
  -- Classes that differ only in batch dimensions, with the order axes of
  -- their picks.
  orderProblems =
    [ "The classes " <> shapeKey a <> " and " <> shapeKey b <> " differ only in batch size, and order reductions differently: " <> axisKey pa <> " against " <> axisKey pb <> "."
    | (i, (a, Right va)) <- zip [0 :: Int ..] picked
    , (j, (b, Right vb)) <- zip [0 :: Int ..] picked
    , i < j
    , unbatched a == unbatched b
    , let pa = orderOf va.variantPoint
          pb = orderOf vb.variantPoint
    , pa /= pb
    ]
  unbatched (Shape ds) = [d | d@(name, _) <- ds, name `notElem` space.spaceBatch]
  orderAxes = [a.axisPrefix | a <- space.spaceAxes, a.axisOrder]
  orderOf (Point vs) = Point [p | p@(prefix, _) <- vs, prefix `elem` orderAxes]
