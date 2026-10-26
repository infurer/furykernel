-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

-- | Variant spaces: axis keys, and what enumeration prunes and why.
module Test.FuryKernel.VariantTest (tests) where

import Data.Char (isDigit)
import Data.Text (Text)
import Data.Text qualified as T
import Data.Word (Word32)
import FuryKernel.Build (Exp, Kernel, dim, groups, kernel, kernelIR, quotE, threads)
import FuryKernel.Finding (Finding (..))
import FuryKernel.IR (DimRange (..))
import FuryKernel.Variant
import Test.FuryKernel.Gemv (GemvShape (..), gemvAxes, gemvSpace)
import Test.FuryKernel.Targets (T32)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (testCase, (@?=))

tests :: TestTree
tests =
  testGroup
    "Variant"
    [ testCase "points, the first axis slowest" $
        map axisKey (points [Axis "a" "a" [1, 2] False, Axis "b" "b" [3, 4, 5] True])
          @?= ["a1_b3", "a1_b4", "a1_b5", "a2_b3", "a2_b4", "a2_b5"]
    , testCase "axis keys read back, and say what is wrong" $ do
        parseAxisKey gemvAxes "r2_t128_p8_s1" @?= Right (Point [("r", 2), ("t", 128), ("p", 8), ("s", 1)])
        parseAxisKey gemvAxes "r2_t128" @?= Left "The axis key r2_t128 has 2 parts for 4 axes."
        parseAxisKey gemvAxes "r2_x128_p8_s1" @?= Left "The axis key r2_x128_p8_s1 has x128 where threads (t) belongs."
        parseAxisKey gemvAxes "r3_t128_p8_s1"
          @?= Left "The axis key r3_t128_p8_s1 gives rows_per_simd 3, which is not one of its values."
        parseAxisKey gemvAxes "r01_t128_p8_s1" @?= Left "The axis key r01_t128_p8_s1 writes rows_per_simd 1 as 01."
        parseAxisKey [] "" @?= Right (Point [])
    , testCase "names: a family without axes, and a class without dimensions" $ do
        variantName space shape (point "r1_t32_p8_s1") @?= "gemv_n8_k256_m1_x_r1_t32_p8_s1"
        variantName space {spaceAxes = []} shape (Point []) @?= "gemv_n8_k256_m1"
        variantName space (Shape []) (point "r1_t32_p8_s1") @?= "gemv_x_r1_t32_p8_s1"
    , testCase "names split back at the first _x_ whose right side is an axis key" $ do
        let wide = gemvSpace @T32 [GemvShape 8 256 1, GemvShape 8 256 2, GemvShape 16 512 1]
        [(s, p) | s <- wide.spaceShapes, p <- points wide.spaceAxes, variantOfName wide (variantName wide s p) /= Just (s, p)] @?= []
        variantOfName space {spaceAxes = []} "gemv_n8_k256_m1" @?= Just (shape, Point [])
        variantOfName space "gemv_n8_k256_m1_x_r3_t32_p8_s1" @?= Nothing
        -- A left side that ends in _x, or holds _x_ itself, splits back.
        let bucket r = maybe False (\ds -> not (T.null ds) && T.all isDigit ds) (T.stripPrefix "m" r)
        splitJoin bucket "model_embed_x_s0_x_x_m1" @?= Just ("model_embed_x_s0_x", "m1")
        splitJoin bucket "a_x_b_x_m16" @?= Just ("a_x_b", "m16")
        splitJoin bucket "a_x_b" @?= Nothing
    , testCase "spaces whose names wouldn't read back, or whose classes would be misread" $ do
        let problems f = checkSpace (f space)
        problems id @?= []
        problems (\s -> s {spaceFamily = "gemv_x_wide"}) @?= ["The family gemv_x_wide holds _x_, which joins a variant's name to its axis key."]
        problems (\s -> s {spaceAxes = gemvAxes <> [Axis "unused" "r" [1] False]}) @?= ["The axes rows_per_simd and unused share the prefix r."]
        problems (\s -> s {spaceAxes = [Axis "rows" "r_" [1] False, Axis "none" "n" [] False]})
          @?= ["The axis rows's prefix \"r_\" is not lowercase letters.", "The axis none has no values."]
        problems (\s -> s {spaceShapes = [Shape [("n", 8), ("k_", 256), ("m", 1)], Shape [("n", 8), ("n", 16), ("m", 1)], shape, shape]})
          @?= [ "The class n8_k_256_m1's dimension \"k_\" is not lowercase letters."
              , "The class n8_n16_m1 gives n 2 times."
              , "The class n8_k256_m1 is given 2 times."
              ]
        problems (\s -> s {spaceBatch = ["mm"]}) @?= ["The batch dimension mm is not a dimension of the class n8_k256_m1."]
    , testCase "pruning, and why" $ do
        let outcome key = describe (consider (Profile 2) space shape (point key))
        outcome "r1_t32_p8_s1" @?= "kept, 8 groups"
        outcome "r1_t32_p16_s1" @?= "illegal: [Packs,Packs]"
        outcome "r1_t32_p8_s4" @?= "illegal: [Threads]"
        outcome "r4_t128_p8_s1" @?= "indivisible: [\"n 8 is not a multiple of the 16 rows of a threadgroup.\"]"
        outcome "r1_t128_p8_s1" @?= "too few: 2 groups"
    , testCase "duplicates: the first point of a kernel stays" $ do
        -- An axis the kernel ignores makes every point twice.
        let twice = space {spaceAxes = space.spaceAxes <> [Axis "unused" "u" [1, 2] False]}
            outcomes = enumerate (Profile 2) twice shape
            kept = [axisKey v.variantPoint | v <- variants (Profile 2) space shape]
        [axisKey p | (p, Kept _) <- outcomes] @?= [k <> "_u1" | k <- kept]
        [(axisKey p, axisKey q) | (p, SameAs q) <- outcomes] @?= [(k <> "_u2", k <> "_u1") | k <- kept]
    , testCase "a launch's fewest threadgroups" $
        launchGroups (kernelIR halves) @?= 3
    , testCase "a launch's fewest threadgroups, over the legal values" $ do
        let fewest r g = launchGroups (kernelIR (gridKernel "g" r g))
        fewest (DimRange 1 4096 64) (\n -> n `quotE` 64 * 8) @?= 8
        fewest (DimRange 64 4096 64) (\n -> n `quotE` 64 * 8) @?= 8
        fewest (DimRange 0 4096 1) (+ 0) @?= 0
        fewest (DimRange 1 4096 1) (4096 `quotE`) @?= 1
        fewest (DimRange 1 4096 1) (\n -> 4096 `quotE` n + n) @?= 128
        fewest (DimRange 1 16 1) (8 -) @?= 0
    , testCase "pruning by the fewest threadgroups, and grids the gate refuses" $ do
        let outcome cores r g = case enumerate (Profile cores) (spaceOf r g) (Shape [("c", 1)]) of
              [(_, o)] -> describeAll o
              os -> show (length os) <> " outcomes"
        outcome 2 (DimRange 1 4096 64) (\n -> n `quotE` 64 * 8) @?= "kept, 8 groups"
        outcome 1 (DimRange 0 4096 1) (+ 0) @?= "too few: 0 groups"
        outcome 40 (DimRange 1 4096 1) (4096 `quotE`) @?= "too few: 1 groups"
        outcome 1 (DimRange 1 4096 1) (\n -> 64 `quotE` (n - 1))
          @?= "illegal: The grid size 64 / (n - 1) divides by 0 at n = 1: n - 1 is 0."
        outcome 2 (DimRange 1 16 1) (8 -) @?= "illegal: The grid size 8 - n leaves u32 at n = 9: 8 - n is -1."
        outcome 0 (DimRange 0 16 1) (* 0) @?= "kept, 0 groups"
        outcome 0 (DimRange 0 16 1) (const 1) @?= "kept, 1 groups"
    ]
 where
  space = gemvSpace @T32 [GemvShape 8 256 1]
  shape = Shape [("n", 8), ("k", 256), ("m", 1)]
  point key = either (error . show) id (parseAxisKey gemvAxes key)
  describe = \case
    Kept v -> "kept, " <> show v.variantGroups <> " groups"
    Illegal fs -> "illegal: " <> show (map (.findingRule) fs)
    Indivisible rs -> "indivisible: " <> show rs
    TooFewGroups n -> "too few: " <> show n <> " groups"
    SameAs p -> "the same as " <> show p

-- | What pruning made of a point, findings by their messages.
describeAll :: Outcome t -> String
describeAll = \case
  Kept v -> "kept, " <> show v.variantGroups <> " groups"
  Illegal fs -> "illegal: " <> unwords (map (T.unpack . (.findingMessage)) fs)
  Indivisible rs -> "indivisible: " <> show rs
  TooFewGroups n -> "too few: " <> show n <> " groups"
  SameAs p -> "the same as " <> show p

-- | A kernel whose grid is a function of one dimension n.
gridKernel :: Text -> DimRange -> (Exp Word32 -> Exp Word32) -> Kernel T32
gridKernel name r g = kernel name $ do
  n <- dim "n" r
  threads @32
  groups (g n)
  pure (pure (pure ()))

-- | One class and one point, around such a kernel.
spaceOf :: DimRange -> (Exp Word32 -> Exp Word32) -> Space T32
spaceOf r g =
  Space
    { spaceFamily = "f"
    , spaceAxes = [Axis "x" "x" [1] False]
    , spaceShapes = [Shape [("c", 1)]]
    , spaceBatch = []
    , spaceDivides = \_ _ -> []
    , spaceKernel = \name _ _ -> Right (gridKernel name r g)
    }

-- | Half as many threadgroups as rows, with at least six rows.
halves :: Kernel T32
halves = kernel "halves" $ do
  rows <- dim "rows" (DimRange 6 64 2)
  threads @32
  groups (rows `quotE` 2)
  pure (pure (pure ()))
