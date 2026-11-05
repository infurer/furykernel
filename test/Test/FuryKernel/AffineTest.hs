-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

-- | The Omega test against brute force, on systems small enough for it.
module Test.FuryKernel.AffineTest (tests) where

import Control.Monad (replicateM)
import Data.Map.Strict qualified as Map
import FuryKernel.Check.Affine
import Hedgehog (Gen, Property, forAll, property, withTests, (===))
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
  testGroup
    "Affine"
    [ testProperty "a system has a solution where brute force finds one, and only there" small
    , testProperty "row-major indices meet where brute force finds them meeting, and only there" rowMajor
    ]

-- | Up to four variables in [-3, 3], up to five constraints.
small :: Property
small = withTests 2000 $ property $ do
  n <- forAll (Gen.int (Range.linear 1 4))
  rows <- forAll (Gen.list (Range.linear 1 5) (constraint n))
  let cs = rows <> concat [[atMost (constant (-3)) (term v), atMost (term v) (constant 3)] | v <- [0 .. n - 1]]
  solvable cs === any (\xs -> all (holds xs) cs) (replicateM n [-3 .. 3])
 where
  constraint :: Int -> Gen (Constraint Int)
  constraint n = do
    equality <- Gen.bool
    ks <- Gen.list (Range.singleton n) (Gen.integral (Range.linear (-6) 6))
    c <- Gen.integral (Range.linear (-12) 12)
    let a = foldr plus (constant c) [times k (term v) | (v, k) <- zip [0 ..] ks]
    pure (if equality then EqualsZero a else NonNegative a)

{- | Two row-major indices of rows of @w@, rows and columns from digits,
equal, the threads apart or not: what the digits take apart.
-}
rowMajor :: Property
rowMajor = withTests 500 $ property $ do
  w <- forAll (Gen.integral (Range.linear 2 6))
  apartBy <- forAll (Gen.element [0, 1, 2, 3 :: Int])
  slack <- forAll (Gen.integral (Range.linear 0 1))
  -- Variables: rows 0 and 2, columns 1 and 3.
  let index r c = plus (times w (term r)) (term c)
      bounds = concat [[atMost (constant 0) (term v), atMost (term v) (constant hi)] | (v, hi) <- [(0, 3), (1, w - 1 + slack), (2, 3), (3, w - 1 + slack)]]
      apart = case apartBy of
        0 -> [atMost (plus (term 0) (constant 1)) (term 2)]
        1 -> [atMost (plus (term 1) (constant 1)) (term 3)]
        2 -> [equal (term 0) (term 2)]
        _ -> []
      cs = equal (index (0 :: Int) 1) (index 2 3) : bounds <> apart
  solvable cs === any (\xs -> all (holds xs) cs) (replicateM 4 [0 .. 6])

holds :: [Integer] -> Constraint Int -> Bool
holds xs = \case
  EqualsZero a -> value a == 0
  NonNegative a -> value a >= 0
 where
  value a = constantPart a + sum [k * (xs !! v) | (v, k) <- Map.toList (coefficients a)]
