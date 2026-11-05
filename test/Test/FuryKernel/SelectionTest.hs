-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

-- | Selections: the format, golden, and everything validation refuses.
module Test.FuryKernel.SelectionTest (tests) where

import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text qualified as T
import Data.Word (Word64)
import FuryKernel.Abi (hex64, irHash)
import FuryKernel.Selection
import FuryKernel.Variant (Outcome (..), Point (..), Profile (..), Space (..), Variant (..), axisKey, consider, parseAxisKey, shapeKey, variantName)
import Test.FuryKernel.Gemv (GemvShape (..), gemvSpace)
import Test.FuryKernel.Targets (T32)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (testCase, (@?=))

tests :: TestTree
tests =
  testGroup
    "Selection"
    [ testCase "the format, classes in key order" $
        renderSelection (written [("n8_k256_m2", "r1_t64_p8_s1", 0x0123456789abcdef), ("n8_k256_m1", "r1_t32_p8_s1", 0xfedcba9876543210)])
          @?= T.unlines
            [ "{"
            , "  \"format\": \"furykernel-selection\","
            , "  \"version\": 1,"
            , "  \"family\": \"gemv\","
            , "  \"target\": \"T32\","
            , "  \"profile\": {"
            , "    \"cores\": 2"
            , "  },"
            , "  \"classes\": {"
            , "    \"n8_k256_m1\": {"
            , "      \"variant\": \"r1_t32_p8_s1\","
            , "      \"ir_hash\": \"0xfedcba9876543210\""
            , "    },"
            , "    \"n8_k256_m2\": {"
            , "      \"variant\": \"r1_t64_p8_s1\","
            , "      \"ir_hash\": \"0x0123456789abcdef\""
            , "    }"
            , "  }"
            , "}"
            ]
    , testCase "what is written reads back" $ do
        let s = written [("n8_k256_m1", "r1_t32_p8_s1", 1), ("n8_k256_m2", "r1_t64_p8_s1", maxBound)]
        parseSelection (renderSelection s) @?= Right s
    , testCase "the format is checked" $ do
        parseSelection "[]" @?= Left ["A selection is a JSON object."]
        parseSelection (field "format" "\"other\"") @?= Left ["The format is other, not furykernel-selection."]
        parseSelection (field "version" "2") @?= Left ["Version 2 of the format; this reads version 1."]
        -- Another version's fields are its own: its version is refused first.
        parseSelection "{\"format\": \"furykernel-selection\", \"version\": 2, \"extra\": 1}" @?= Left ["Version 2 of the format; this reads version 1."]
        parseSelection (field "classes" "{\"n8_k256_m1\": \"r1_t32_p8_s1\"}") @?= Left ["The field classes has the wrong type."]
        parseSelection (field "classes" "{\"n8_k256_m1\": {\"variant\": \"r1_t32_p8_s1\", \"ir_hash\": \"0x12\"}}") @?= Left ["The field classes has the wrong type."]
        parseSelection (field "profile" "{\"cores\": -1}") @?= Left ["The field profile has the wrong type."]
        parseSelection "{\"format\": \"furykernel-selection\", \"version\": 1, \"extra\": 1}" @?= Left ["An unknown field: extra."]
        parseSelection "{\"format\": \"furykernel-selection\", \"version\": 1}" @?= Left ["The field family is missing."]
    , testCase "a valid selection gives a variant per class" $
        fmap (map (\v -> (shapeKey v.variantShape, axisKey v.variantPoint))) (validate [("n8_k256_m1", "r1_t32_p8_s1"), ("n8_k256_m2", "r2_t32_p8_s1")])
          @?= Right [("n8_k256_m1", "r1_t32_p8_s1"), ("n8_k256_m2", "r2_t32_p8_s1")]
    , testCase "the batch sizes of a class share the order of its reductions" $
        fmap length (validate [("n8_k256_m1", "r1_t32_p8_s1"), ("n8_k256_m2", "r1_t64_p8_s2")])
          @?= Left
            [ "The classes n8_k256_m1 and n8_k256_m2 differ only in batch size, and order reductions differently: p8_s1 against p8_s2."
            ]
    , testCase "a pick is the kernel it was picked as, on the profile it was picked on" $ do
        fmap length (validateSelection profile space (written [("n8_k256_m1", "r1_t32_p8_s1", 1), ("n8_k256_m2", "r2_t32_p8_s1", hashOf "n8_k256_m2" "r2_t32_p8_s1")]))
          @?= Left ["gemv_n8_k256_m1_x_r1_t32_p8_s1 was picked as the kernel of IR hash 0x0000000000000001, and is now " <> hex64 (hashOf "n8_k256_m1" "r1_t32_p8_s1") <> "."]
        fmap length (validateSelection (Profile 1) space (picks [("n8_k256_m1", "r1_t32_p8_s1"), ("n8_k256_m2", "r2_t32_p8_s1")]))
          @?= Left ["The selection was made for a GPU of 2 cores, not 1."]
    , testCase "everything else validation refuses" $ do
        fmap length (validateSelection @T32 profile space (Selection "gemm" "T64" profile Map.empty))
          @?= Left
            [ "The selection is for the family gemm, not gemv."
            , "The selection is for the target T64, not T32."
            , "No variant for the class n8_k256_m1."
            , "No variant for the class n8_k256_m2."
            ]
        fmap length (validate [("n8_k256_m1", "r1_t32_p16_s1"), ("n8_k256_m2", "r4_t128_p8_s1"), ("n9_k256_m1", "r1_t32_p8_s1"), ("n9_k256_m1", "r1_t32_p8_s1")])
          @?= Left
            [ "The class n9_k256_m1 is not one of the family's."
            , "gemv_n8_k256_m1_x_r1_t32_p16_s1 is illegal: Packs: Pack of 16 elements of 2 B = 32 B; the widest lane load on T32 is 16 B. Packs: Pack of 16 elements of 2 B = 32 B from a buffer aligned to 16 B: align it to a multiple of 32 B."
            , "gemv_n8_k256_m2_x_r4_t128_p8_s1 does not divide its class: n 8 is not a multiple of the 16 rows of a threadgroup."
            ]
        fmap length (validate [("n8_k256_m1", "r1_t128_p8_s1"), ("n8_k256_m2", "r1_t32_q8_s1")])
          @?= Left
            [ "gemv_n8_k256_m1_x_r1_t128_p8_s1 launches 2 threadgroups, fewer than twice the 2 cores."
            , "The class n8_k256_m2's axis key r1_t32_q8_s1 has q8 where pack (p) belongs."
            ]
        fmap length (validateSelection profile space {spaceBatch = ["mm"]} (picks [("n8_k256_m1", "r1_t32_p8_s1"), ("n8_k256_m2", "r2_t32_p8_s1")]))
          @?= Left
            [ "The batch dimension mm is not a dimension of the class n8_k256_m1."
            , "The batch dimension mm is not a dimension of the class n8_k256_m2."
            ]
    , testCase "a family without axes: one variant a class, named family_class, picked by the empty key" $ do
        let fixed = Point [("r", 1), ("t", 32), ("p", 8), ("s", 1)]
            flat = space {spaceAxes = [], spaceKernel = \name sh _ -> space.spaceKernel name sh fixed, spaceDivides = \sh _ -> space.spaceDivides sh fixed}
            hashIn s = case consider profile flat s (Point []) of
              Kept v -> irHash v.variantKernel
              _ -> 0
            sel = Selection "gemv" "T32" profile (Map.fromList [(shapeKey s, Pick "" (hashIn s)) | s <- flat.spaceShapes])
        fmap (map (\v -> variantName flat v.variantShape v.variantPoint)) (validateSelection profile flat sel)
          @?= Right ["gemv_n8_k256_m1", "gemv_n8_k256_m2"]
        parseSelection (renderSelection sel) @?= Right sel
    ]
 where
  space = gemvSpace @T32 [GemvShape 8 256 1, GemvShape 8 256 2]
  profile = Profile 2
  written :: [(Text, Text, Word64)] -> Selection
  written ps = Selection "gemv" "T32" profile (Map.fromList [(c, Pick a h) | (c, a, h) <- ps])
  -- Picks of what each variant's kernel is now; 0 for a class or point
  -- there is no kernel of.
  picks ps = written [(c, a, hashOf c a) | (c, a) <- ps]
  hashOf c a = case ([s | s <- space.spaceShapes, shapeKey s == c], parseAxisKey space.spaceAxes a) of
    (s : _, Right point) | Kept v <- consider profile space s point -> irHash v.variantKernel
    _ -> 0
  validate = validateSelection profile space . picks
  field :: Text -> Text -> Text
  field name value =
    "{"
      <> T.intercalate
        ", "
        [ "\"" <> k <> "\": " <> (if k == name then value else v)
        | (k, v) <-
            [ ("format", "\"furykernel-selection\"")
            , ("version", "1")
            , ("family", "\"gemv\"")
            , ("target", "\"T32\"")
            , ("profile", "{\"cores\": 2}")
            , ("classes", "{}")
            ]
        ]
      <> "}"
