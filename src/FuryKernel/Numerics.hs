-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | Conversions between binary32 and the 16-bit floating types, as kernels
perform them, and from binary64 to binary16: widening is exact, narrowing
rounds to nearest, ties to even, once.

NaNs stay NaNs and come out quiet, as IEEE 754's convertFormat has them:
widening keeps the sign and the payload and sets the quiet bit; narrowing
keeps the sign and the top payload bits and sets the quiet bit, so a NaN
never turns into an infinity. Overflow rounds to the infinity of the same
sign; values below the smallest binary16 subnormal round to a signed zero.
-}
module FuryKernel.Numerics (
  -- * bfloat16
  widenBF16,
  narrowBF16,

  -- * binary16
  widenF16,
  narrowF16,
  narrowF16Double,
  rationalF16,
) where

import Data.Bits (Bits, shiftL, shiftR, testBit, (.&.), (.|.))
import Data.Ratio (denominator, numerator)
import Data.Word (Word16, Word32, Word64)
import FuryKernel.Type (BF16 (..), F16 (..))
import GHC.Float (castDoubleToWord64, castFloatToWord32, castWord32ToFloat)

-- | Exact: a bfloat16 is the upper half of a binary32. A NaN comes out quiet.
widenBF16 :: BF16 -> Float
widenBF16 (BF16 b)
  | w .&. 0x7F800000 == 0x7F800000 && w .&. 0x007FFFFF /= 0 = castWord32ToFloat (w .|. 0x00400000)
  | otherwise = castWord32ToFloat w
 where
  w = fromIntegral b `shiftL` 16 :: Word32

-- | Round to nearest, ties to even.
narrowBF16 :: Float -> BF16
narrowBF16 x
  | isNaN x = BF16 (fromIntegral (w `shiftR` 16) .|. 0x0040)
  | otherwise = BF16 (fromIntegral ((w + 0x7FFF + lsb) `shiftR` 16))
 where
  w = castFloatToWord32 x
  -- The lowest bit that survives, which decides ties.
  lsb = (w `shiftR` 16) .&. 1

-- | Exact: every binary16 value is a binary32 value. A NaN comes out quiet.
widenF16 :: F16 -> Float
widenF16 (F16 h) = castWord32ToFloat (sign .|. magnitude)
 where
  sign = fromIntegral (h .&. 0x8000) `shiftL` 16 :: Word32
  e = fromIntegral ((h `shiftR` 10) .&. 0x1F) :: Word32
  m = fromIntegral (h .&. 0x3FF) :: Word32
  magnitude
    | e == 0x1F && m /= 0 = 0x7FC00000 .|. (m `shiftL` 13)
    | e == 0x1F = 0x7F800000
    | e /= 0 = ((e + 112) `shiftL` 23) .|. (m `shiftL` 13)
    | m == 0 = 0
    -- A subnormal: normalise it. The value is m * 2^-24.
    | otherwise =
        let shift = countLeadingZeros10 m
            m' = (m `shiftL` (shift + 1)) .&. 0x3FF
         in ((113 - fromIntegral shift - 1) `shiftL` 23) .|. (m' `shiftL` 13)

-- | Leading zeros of a 10-bit mantissa (0 < m < 1024).
countLeadingZeros10 :: Word32 -> Int
countLeadingZeros10 m = length (takeWhile (not . testBit m) [9, 8 .. 0])

{- | The binary16 nearest a rational, ties to even, in one rounding, as a
literal's value is: past the largest finite binary16 by half an ulp or
more, infinity; zero is +0.
-}
rationalF16 :: Rational -> F16
rationalF16 r
  | r < 0 = let F16 b = rationalF16 (negate r) in F16 (b .|. 0x8000)
  | r == 0 = F16 0
  -- Subnormals and the smallest normals: multiples of 2^-24.
  | e < -14 = F16 (fromInteger (round (r * 2 ^ (24 :: Int))))
  | m == 2048 = normal (e + 1) 1024
  | otherwise = normal e m
 where
  -- 2^e <= r < 2^(e + 1), from the bit lengths of r's numerator and
  -- denominator, corrected by one either way.
  e = settle (bits (numerator r) - bits (denominator r))
  settle k
    | 2 ^^ k > r = settle (k - 1)
    | 2 ^^ (k + 1) <= r = settle (k + 1)
    | otherwise = k
  bits n = length (takeWhile (> 0) (iterate (`div` 2) n)) - 1 :: Int
  -- r's 11 significant bits, rounded: 1024 to 2048.
  m = round (r / 2 ^^ (e - 10)) :: Integer
  normal k bits11
    | k > 15 = F16 0x7C00
    | otherwise = F16 (fromIntegral ((k + 15) `shiftL` 10) .|. fromInteger (bits11 - 1024))

-- | Round to nearest, ties to even, with gradual underflow.
narrowF16 :: Float -> F16
narrowF16 x = F16 (sign .|. magnitude)
 where
  w = castFloatToWord32 x
  sign = fromIntegral ((w `shiftR` 16) .&. 0x8000) :: Word16
  e = fromIntegral ((w `shiftR` 23) .&. 0xFF) :: Int
  m = w .&. 0x7FFFFF
  unbiased = e - 127
  magnitude
    | e == 0xFF && m /= 0 = 0x7E00 .|. fromIntegral (m `shiftR` 13)
    | e == 0xFF = 0x7C00
    -- 2^16 and above overflow whatever the mantissa; just below, the
    -- rounding carry below produces the infinity itself.
    | unbiased >= 16 = 0x7C00
    | unbiased >= -14 =
        let r = fromIntegral ((fromIntegral (unbiased + 15) `shiftL` 10) .|. (m `shiftR` 13)) :: Word32
         in fromIntegral (roundShifted r (m .&. 0x1FFF) 13)
    -- Subnormal or zero: the value in units of 2^-24 is M * 2^(unbiased + 1),
    -- with the implicit bit made explicit.
    | otherwise =
        let shift = negate (unbiased + 1)
            full = m .|. 0x800000
         in if shift > 24
              then 0
              else
                fromIntegral
                  (roundShifted (full `shiftR` shift) (full .&. ((1 `shiftL` shift) - 1)) shift)

{- | The binary16 nearest a binary64, ties to even, rounded once from all
its bits, not through binary32, which would round twice. NaNs, overflow
and underflow as 'narrowF16' has them.
-}
narrowF16Double :: Double -> F16
narrowF16Double x = F16 (sign .|. magnitude)
 where
  w = castDoubleToWord64 x
  sign = fromIntegral ((w `shiftR` 48) .&. 0x8000) :: Word16
  e = fromIntegral ((w `shiftR` 52) .&. 0x7FF) :: Int
  m = w .&. 0xFFFFFFFFFFFFF
  unbiased = e - 1023
  magnitude
    | e == 0x7FF && m /= 0 = 0x7E00 .|. fromIntegral (m `shiftR` 42)
    | e == 0x7FF = 0x7C00
    | unbiased >= 16 = 0x7C00
    | unbiased >= -14 =
        let r = (fromIntegral (unbiased + 15) `shiftL` 10) .|. (m `shiftR` 42) :: Word64
         in fromIntegral (roundShifted r (m .&. ((1 `shiftL` 42) - 1)) 42)
    -- Subnormal or zero: the value in units of 2^-24 is M * 2^(unbiased - 28),
    -- with the implicit bit made explicit; below 2^-25, zero.
    | otherwise =
        let shift = 28 - unbiased
            full = m .|. (1 `shiftL` 52)
         in if shift > 53
              then 0
              else
                fromIntegral
                  (roundShifted (full `shiftR` shift) (full .&. ((1 `shiftL` shift) - 1)) shift)

{- | Round a truncated value up when the dropped bits are above half, or
exactly half and the kept value is odd.
-}
roundShifted :: (Bits w, Num w, Ord w) => w -> w -> Int -> w
roundShifted kept dropped width
  | dropped > half = kept + 1
  | dropped == half && testBit kept 0 = kept + 1
  | otherwise = kept
 where
  half = 1 `shiftL` (width - 1)
