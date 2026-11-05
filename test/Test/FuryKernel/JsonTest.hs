-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

-- | JSON: what is written reads back, and what isn't JSON says where.
module Test.FuryKernel.JsonTest (tests) where

import Data.List (nubBy)
import Data.Text qualified as T
import FuryKernel.Json (Json (..), parseJson, renderJson)
import Hedgehog (Gen, Property, forAll, property, tripping)
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (testCase, (@?=))
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
  testGroup
    "Json"
    [ testProperty "what is written reads back" roundTrip
    , testCase "escapes, surrogate pairs, whitespace" $ do
        parseJson " { \"a\\\"b\\\\c\\/d\\u00e9\\ud83d\\ude00\" : [ -12 , true , null ] } "
          @?= Right (JObject [("a\"b\\c/d\233\128512", JArray [JInt (-12), JBool True, JNull])])
        parseJson "[]" @?= Right (JArray [])
        parseJson "{}" @?= Right (JObject [])
    , testCase "what is not JSON, and where" $ do
        parseJson "{\"a\": 1, \"a\": 2}" @?= Left "The key a twice at character 9."
        parseJson "1.5" @?= Left "A number that is not an integer at character 1."
        parseJson "[1 2]" @?= Left "No , or ] after an element at character 3."
        parseJson "01" @?= Left "A leading zero at character 0."
        parseJson "\"\\ud800\"" @?= Left "A lone surrogate at character 7."
        parseJson "\"a\nb\"" @?= Left "A control character in a string at character 2."
        parseJson "true false" @?= Left "Text after the value at character 5."
        parseJson "" @?= Left "No JSON value at character 0."
    ]

roundTrip :: Property
roundTrip = property $ do
  j <- forAll json
  tripping j renderJson parseJson

json :: Gen Json
json =
  Gen.recursive
    Gen.choice
    [ pure JNull
    , JBool <$> Gen.bool
    , JInt <$> Gen.integral (Range.linearFrom 0 (-(2 ^ (70 :: Int))) (2 ^ (70 :: Int)))
    , JString <$> text
    ]
    [ JArray <$> Gen.list (Range.linear 0 5) json
    , JObject . nubBy (\a b -> fst a == fst b) <$> Gen.list (Range.linear 0 5) ((,) <$> text <*> json)
    ]
 where
  text = T.pack <$> Gen.list (Range.linear 0 8) (Gen.frequency [(5, Gen.alphaNum), (2, Gen.element ("\"\\/\n\t\r\b\f\0" :: String)), (1, Gen.unicode)])
