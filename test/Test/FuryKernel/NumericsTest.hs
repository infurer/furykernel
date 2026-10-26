-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The narrowing conversions against an independent oracle: exact
'Rational' arithmetic, where 'round' already rounds ties to even.
-}
module Test.FuryKernel.NumericsTest (tests) where

import Data.Bits (shiftR, (.&.))
import Data.Word (Word16, Word32)
import FuryKernel.Numerics (narrowBF16, narrowF16, narrowF16Double, rationalF16, widenBF16, widenF16)
import FuryKernel.Type (BF16 (..), F16 (..))
import GHC.Float (castDoubleToWord64, castFloatToWord32, castWord32ToFloat, castWord64ToDouble, float2Double)
import Hedgehog (Gen, Property, forAll, property, withTests, (===))
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (assertBool, testCase, (@?=))
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
  testGroup
    "Numerics"
    [ testCase "bf16 → f32 → bf16 is the identity, NaNs aside" $
        roundTrip (BF16 . fromIntegral) widenBF16 narrowBF16
    , testCase "f16 → f32 → f16 is the identity, NaNs aside" $
        roundTrip (F16 . fromIntegral) widenF16 narrowF16
    , testCase "NaNs narrow to quiet NaNs of the same sign" $ do
        let nan = castWord32ToFloat 0xFF800001
        assertBool "bf16" (isNaN (widenBF16 (narrowBF16 nan)))
        assertBool "f16" (isNaN (widenF16 (narrowF16 nan)))
        bits (narrowBF16 nan) .&. 0x8040 @?= 0x8040
        f16Bits (narrowF16 nan) .&. 0x8200 @?= 0x8200
    , testCase "NaNs widen quiet, their sign and payload kept" $ do
        castFloatToWord32 (widenBF16 (BF16 0x7F81)) @?= 0x7FC10000
        castFloatToWord32 (widenBF16 (BF16 0xFF81)) @?= 0xFFC10000
        castFloatToWord32 (widenBF16 (BF16 0x7FC1)) @?= 0x7FC10000
        castFloatToWord32 (widenBF16 (BF16 0x7F80)) @?= 0x7F800000
        castFloatToWord32 (widenF16 (F16 0x7C01)) @?= 0x7FC02000
        castFloatToWord32 (widenF16 (F16 0xFD00)) @?= 0xFFE00000
        castFloatToWord32 (widenF16 (F16 0x7E01)) @?= 0x7FC02000
        castFloatToWord32 (widenF16 (F16 0xFC00)) @?= 0xFF800000
    , testCase "binary16 edge cases" $ do
        f16Bits (narrowF16 65504) @?= 0x7BFF
        f16Bits (narrowF16 65519.996) @?= 0x7BFF
        f16Bits (narrowF16 65520) @?= 0x7C00
        f16Bits (narrowF16 (2 ^^ (-24 :: Int))) @?= 0x0001
        f16Bits (narrowF16 (2 ^^ (-25 :: Int))) @?= 0x0000
        f16Bits (narrowF16 (1.5 * 2 ^^ (-24 :: Int))) @?= 0x0002
        f16Bits (narrowF16 (-0.0)) @?= 0x8000
    , testCase "rationals round to binary16 once: every finite value, and every midpoint to the even side" $ do
        let finite = [F16 b | b <- [0 .. 0x7BFF]]
            value h = toRational (widenF16 h)
        mapM_ (\h -> rationalF16 (value h) @?= h) finite
        mapM_ (\h -> rationalF16 (negate (value h)) @?= F16 (0x8000 + f16Bits h)) (drop 1 finite)
        mapM_
          ( \(a, b) -> do
              let even' = if even (f16Bits a) then a else b
              rationalF16 ((value a + value b) / 2) @?= even'
              rationalF16 ((3 * value a + value b) / 4) @?= a
              rationalF16 ((value a + 3 * value b) / 4) @?= b
          )
          (zip finite (drop 1 finite))
    , testCase "rationals to binary16: zero, overflow, and what rounding through binary32 would get wrong" $ do
        rationalF16 0 @?= F16 0
        rationalF16 65519 @?= F16 0x7BFF
        rationalF16 65520 @?= F16 0x7C00
        rationalF16 (-65520) @?= F16 0xFC00
        rationalF16 (10 ^ (6 :: Int)) @?= F16 0x7C00
        rationalF16 (2 ^^ (-26 :: Int)) @?= F16 0
        rationalF16 (1 + 1 / 2048 + 1 / 1073741824) @?= F16 0x3C01
        narrowF16 (fromRational (1 + 1 / 2048 + 1 / 1073741824)) @?= F16 0x3C00
    , testCase "bfloat16 ties go to even" $ do
        bits (narrowBF16 (1 + 2 ^^ (-8 :: Int))) @?= 0x3F80
        bits (narrowBF16 (1 + 3 * 2 ^^ (-8 :: Int))) @?= 0x3F82
    , testCase "f16 narrowing at every midpoint: ties to even, a binary32 ulp either side to the nearer" $
        midpoints narrowF16 widenF16 f16Bits F16 0x7BFF
    , testCase "bf16 narrowing at every midpoint: ties to even, a binary32 ulp either side to the nearer" $
        midpoints narrowBF16 widenBF16 bits BF16 0x7F7F
    , testCase "binary64 to binary16: edge cases, and what rounding through binary32 would get wrong" $ do
        f16Bits (narrowF16Double 65504) @?= 0x7BFF
        f16Bits (narrowF16Double 65519.99999999999) @?= 0x7BFF
        f16Bits (narrowF16Double 65520) @?= 0x7C00
        f16Bits (narrowF16Double (-1e300)) @?= 0xFC00
        f16Bits (narrowF16Double (2 ^^ (-24 :: Int))) @?= 0x0001
        f16Bits (narrowF16Double (2 ^^ (-25 :: Int))) @?= 0x0000
        f16Bits (narrowF16Double (2 ^^ (-25 :: Int) + 2 ^^ (-60 :: Int))) @?= 0x0001
        f16Bits (narrowF16Double (-(2 ^^ (-26 :: Int)))) @?= 0x8000
        f16Bits (narrowF16Double 5e-324) @?= 0x0000
        f16Bits (narrowF16Double (-0.0)) @?= 0x8000
        f16Bits (narrowF16Double (1 / 0)) @?= 0x7C00
        f16Bits (narrowF16Double (castWord64ToDouble 0xFFF0000000000001)) .&. 0xFE00 @?= 0xFE00
        -- Just past a tie, by a bit below binary64's upper 32: once, up;
        -- through binary32, or from the upper bits alone, the tie, to even.
        f16Bits (narrowF16Double (1 + 2 ^^ (-11 :: Int) + 2 ^^ (-25 :: Int))) @?= 0x3C01
        f16Bits (narrowF16 (realToFrac (1 + 2 ^^ (-11 :: Int) + 2 ^^ (-25 :: Int) :: Double))) @?= 0x3C00
    , testCase "f16 narrowing from binary64 at every midpoint: ties to even, a binary64 ulp either side to the nearer" $
        mapM_ midpoint64 [(lo, negative) | lo <- [0 .. 0x7BFE], negative <- [False, True]]
    , testProperty "binary64 narrows to the binary16 a rational would, the sign of zero kept" $
        withTests 20000 . property $ do
          x <- forAll doubles
          let want
                | x == 0 = F16 (if isNegativeZero x then 0x8000 else 0)
                | otherwise = rationalF16 (toRational x)
          narrowF16Double x === want
    , testProperty "bf16 narrowing rounds to nearest, ties to even" $
        nearest narrowBF16 widenBF16 bf16Ulp (2 ^^ (128 :: Int))
    , testProperty "f16 narrowing rounds to nearest, ties to even" $
        nearest narrowF16 widenF16 f16Ulp 65536
    ]

bits :: BF16 -> Word16
bits (BF16 b) = b

f16Bits :: F16 -> Word16
f16Bits (F16 h) = h

{- | Every two neighbouring finite values of a 16-bit float type, and their
negations, up to the largest: their midpoint, exact in binary32, narrows to
the one whose last bit is 0, and the binary32 values next to it to the
nearer of the two.
-}
midpoints :: (Float -> h) -> (h -> Float) -> (h -> Word16) -> (Word16 -> h) -> Word16 -> IO ()
midpoints narrow widen toBits mk largest =
  mapM_ pair [(lo, lo + 1, s) | lo <- [0 .. largest - 1], s <- [False, True]]
 where
  pair (lo, hi, negative) = do
    let a = widen (mk lo)
        mid = a + (widen (mk hi) - a) / 2
        step d = castWord32ToFloat (castFloatToWord32 mid + d)
        signed x = if negative then negate x else x
        expect w = if negative then w + 0x8000 else w
    toBits (narrow (signed mid)) @?= expect (if even lo then lo else hi)
    toBits (narrow (signed (step 1))) @?= expect hi
    toBits (narrow (signed (castWord32ToFloat (castFloatToWord32 mid - 1)))) @?= expect lo

roundTrip :: (Eq h, Show h) => (Int -> h) -> (h -> Float) -> (Float -> h) -> IO ()
roundTrip mk widen narrow =
  mapM_
    ( \i -> do
        let h = mk i
            x = widen h
        if isNaN x then pure () else narrow x @?= h
    )
    [0 .. 0xFFFF]

{- | The narrowed value is the representable value nearest to the input,
ties to even; past the largest finite value the next step is infinity,
whose distance counts as if it were 'overflow'.
-}
nearest :: (Float -> h) -> (h -> Float) -> (Rational -> Rational) -> Rational -> Property
nearest narrow' widen' ulpOf overflow = withTests 20000 $ property $ do
  x <- forAll floats
  let a = abs (toRational x)
      ulp = ulpOf a
      n = round (a / ulp) :: Integer
      want = fromInteger n * ulp
      got = widen' (narrow' x)
  if want >= overflow
    then isInfinite got === True
    else toRational (abs got) === want

{- | Two neighbouring binary16 values (and their negations): their
midpoint, exact in binary64, narrows to the even one, and the binary64
values next to it to the nearer.
-}
midpoint64 :: (Word16, Bool) -> IO ()
midpoint64 (lo, negative) = do
  let a = float2Double (widenF16 (F16 lo))
      mid = a + (float2Double (widenF16 (F16 (lo + 1))) - a) / 2
      step d = castWord64ToDouble (castDoubleToWord64 mid + d)
      down = castWord64ToDouble (castDoubleToWord64 mid - 1)
      signed x = if negative then negate x else x
      expect w = if negative then w + 0x8000 else w
  f16Bits (narrowF16Double (signed mid)) @?= expect (if even lo then lo else lo + 1)
  f16Bits (narrowF16Double (signed (step 1))) @?= expect (lo + 1)
  f16Bits (narrowF16Double (signed down)) @?= expect lo

-- | Finite doubles: any bit pattern, plus the ranges near binary16's limits.
doubles :: Gen Double
doubles =
  Gen.filter (\x -> not (isNaN x || isInfinite x)) $
    Gen.choice
      [ castWord64ToDouble <$> Gen.word64 Range.constantBounded
      , Gen.double (Range.linearFracFrom 0 (-70000) 70000)
      , Gen.double (Range.linearFracFrom 0 (-1e-4) 1e-4)
      , -- A binary16 midpoint with a tail far below it.
        (\h t -> let a = float2Double (widenF16 (F16 h)) in a + (float2Double (widenF16 (F16 (h + 1))) - a) / 2 + t * a * 2 ^^ (-40 :: Int))
          <$> Gen.word16 (Range.constant 0 0x7BFE)
          <*> Gen.double (Range.constantFrom 0 (-1) 1)
      ]

-- | Finite floats: any bit pattern, plus the ranges near binary16's limits.
floats :: Gen Float
floats =
  Gen.filter (\x -> not (isNaN x || isInfinite x)) $
    Gen.choice
      [ castWord32ToFloat <$> Gen.word32 Range.constantBounded
      , Gen.float (Range.linearFracFrom 0 (-70000) 70000)
      , Gen.float (Range.linearFracFrom 0 (-1e-4) 1e-4)
      ]

-- | The spacing of bfloat16 values around a magnitude.
bf16Ulp :: Rational -> Rational
bf16Ulp a = 2 ^^ (max (exponentOf a) (-126) - 7)

-- | The spacing of binary16 values around a magnitude.
f16Ulp :: Rational -> Rational
f16Ulp a = 2 ^^ (max (exponentOf a) (-14) - 10)

{- | floor (log2 a) for the magnitude of a binary32 value (which a always
is here), and -127 for zero and the subnormals.
-}
exponentOf :: Rational -> Int
exponentOf a = fromIntegral ((w `shiftR` 23) .&. 0xFF) - 127
 where
  w = castFloatToWord32 (fromRational a) :: Word32
