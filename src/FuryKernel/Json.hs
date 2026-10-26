-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | Just enough JSON to write manifests: deterministic, keys in the order
given, two-space indentation. Nothing here parses JSON.
-}
module FuryKernel.Json (
  Json (..),
  renderJson,
) where

import Data.Char (ord)
import Data.Text (Text)
import Data.Text qualified as T
import Numeric (showHex)

data Json
  = JNull
  | JBool Bool
  | JInt Integer
  | JString Text
  | JArray [Json]
  | JObject [(Text, Json)]
  deriving stock (Eq, Show)

renderJson :: Json -> Text
renderJson j = go 0 j <> "\n"
 where
  go :: Int -> Json -> Text
  go depth = \case
    JNull -> "null"
    JBool b -> if b then "true" else "false"
    JInt n -> T.pack (show n)
    JString s -> string s
    JArray [] -> "[]"
    JArray xs ->
      "[\n"
        <> T.intercalate ",\n" [pad (depth + 1) <> go (depth + 1) x | x <- xs]
        <> "\n"
        <> pad depth
        <> "]"
    JObject [] -> "{}"
    JObject kvs ->
      "{\n"
        <> T.intercalate ",\n" [pad (depth + 1) <> string k <> ": " <> go (depth + 1) v | (k, v) <- kvs]
        <> "\n"
        <> pad depth
        <> "}"
  pad n = T.replicate (2 * n) " "

string :: Text -> Text
string s = "\"" <> T.concatMap escape s <> "\""
 where
  escape = \case
    '"' -> "\\\""
    '\\' -> "\\\\"
    '\n' -> "\\n"
    '\r' -> "\\r"
    '\t' -> "\\t"
    c
      | ord c < 0x20 -> "\\u" <> T.justifyRight 4 '0' (T.pack (showHex (ord c) ""))
      | otherwise -> T.singleton c
