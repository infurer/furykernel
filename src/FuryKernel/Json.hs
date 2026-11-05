-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | Just enough JSON for manifests and selections. Writing is
deterministic: keys in the order given, two-space indentation. Reading
takes RFC 8259 JSON, except that numbers must be integers (nothing here
needs fractions) and an object's keys must be unique.
-}
module FuryKernel.Json (
  Json (..),
  renderJson,
  parseJson,
) where

import Data.Bifunctor (bimap)
import Data.Bits (shiftL, (.&.))
import Data.Char (chr, isDigit, isHexDigit, ord)
import Data.Text (Text)
import Data.Text qualified as T
import Numeric (readHex, showHex)

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

{- | One JSON value, with nothing but whitespace around it; or what is
wrong, at which character.
-}
parseJson :: Text -> Either Text Json
parseJson input = do
  (j, rest) <- value (skip input)
  if T.null rest then Right j else failAt rest "Text after the value"
 where
  failAt :: Text -> Text -> Either Text a
  failAt rest what = Left (what <> " at character " <> T.pack (show (T.length input - T.length rest)) <> ".")

  skip = T.dropWhile (`elem` [' ', '\t', '\n', '\r'])

  value :: Text -> Either Text (Json, Text)
  value t = case T.uncons t of
    Just ('{', rest) -> object (skip rest)
    Just ('[', rest) -> array (skip rest)
    Just ('"', rest) -> bimap JString skip <$> stringBody rest
    Just (c, _) | c == '-' || isDigit c -> number t
    _ -> case [(j, rest) | (word, j) <- [("true", JBool True), ("false", JBool False), ("null", JNull)], Just rest <- [T.stripPrefix word t]] of
      (j, rest) : _ -> Right (j, skip rest)
      [] -> failAt t "No JSON value"

  object t = case T.uncons t of
    Just ('}', rest) -> Right (JObject [], skip rest)
    _ -> members [] t
  members acc t = do
    (key, afterKey) <- case T.uncons t of
      Just ('"', rest) -> stringBody rest
      _ -> failAt t "No key"
    rest <- expect ':' (skip afterKey)
    (j, afterValue) <- value (skip rest)
    let acc' = (key, j) : acc
    if key `elem` map fst acc
      then failAt t ("The key " <> key <> " twice")
      else case T.uncons afterValue of
        Just (',', more) -> members acc' (skip more)
        Just ('}', more) -> Right (JObject (reverse acc'), skip more)
        _ -> failAt afterValue "No , or } after a member"

  array t = case T.uncons t of
    Just (']', rest) -> Right (JArray [], skip rest)
    _ -> elements [] t
  elements acc t = do
    (j, rest) <- value t
    case T.uncons rest of
      Just (',', more) -> elements (j : acc) (skip more)
      Just (']', more) -> Right (JArray (reverse (j : acc)), skip more)
      _ -> failAt rest "No , or ] after an element"

  expect c t = case T.uncons t of
    Just (c', rest) | c' == c -> Right rest
    _ -> failAt t ("No " <> T.singleton c)

  -- After the opening quote.
  stringBody :: Text -> Either Text (Text, Text)
  stringBody = go []
   where
    go acc t = case T.uncons t of
      Nothing -> failAt t "An unterminated string"
      Just ('"', rest) -> Right (T.concat (reverse acc), rest)
      Just ('\\', rest) -> case T.uncons rest of
        Just (e, more)
          | Just c <- lookup e simple -> go (T.singleton c : acc) more
          | e == 'u' -> do
              (code, more') <- hex4 more
              if code >= 0xD800 && code < 0xDC00
                then case T.stripPrefix "\\u" more' of
                  Just low -> do
                    (code2, more'') <- hex4 low
                    if code2 >= 0xDC00 && code2 < 0xE000
                      then go (T.singleton (chr (0x10000 + ((code .&. 0x3FF) `shiftL` 10) + (code2 .&. 0x3FF))) : acc) more''
                      else failAt low "A lone surrogate"
                  Nothing -> failAt more' "A lone surrogate"
                else
                  if code >= 0xDC00 && code < 0xE000
                    then failAt more "A lone surrogate"
                    else go (T.singleton (chr code) : acc) more'
        _ -> failAt rest "An unknown escape"
      Just (c, rest)
        | ord c < 0x20 -> failAt t "A control character in a string"
        | otherwise ->
            let (plain, more) = T.break (\x -> x == '"' || x == '\\' || ord x < 0x20) rest
             in go (plain : T.singleton c : acc) more
    simple = [('"', '"'), ('\\', '\\'), ('/', '/'), ('b', '\b'), ('f', '\f'), ('n', '\n'), ('r', '\r'), ('t', '\t')]
    hex4 t = case T.splitAt 4 t of
      (h, rest) | T.length h == 4, T.all isHexDigit h, [(code, "")] <- readHex (T.unpack h) -> Right (code, rest)
      _ -> failAt t "A \\u escape without four hex digits"

  number t = integer (maybe ("", t) ("-",) (T.stripPrefix "-" t))
  integer (sign, unsigned)
    | T.null digits = failAt unsigned "No digits"
    | T.length digits > 1 && "0" `T.isPrefixOf` digits = failAt unsigned "A leading zero"
    | Just (c, _) <- T.uncons rest, c `elem` ['.', 'e', 'E'] = failAt rest "A number that is not an integer"
    | otherwise = Right (JInt (read (T.unpack (sign <> digits))), skip rest)
   where
    (digits, rest) = T.span isDigit unsigned
