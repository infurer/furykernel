-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE AllowAmbiguousTypes #-}

{- | Scalar types: the element types of buffers, threadgroup arrays and
expressions.

Compute types: 'Float', 'F16', 'Int32' and 'Word32' carry arithmetic, and
'Bool' logic. Storage types only live in memory: 'BF16', 'Int8' and 'Word8'.
Kernels widen them to a compute type, compute, and narrow the result back,
always explicitly: floats rounding to nearest with ties to even, integers
clamped to the narrower range or wrapped around, as the kernel says.
-}
module FuryKernel.Type (
  -- * Scalar types
  ScalarType (..),
  scalarSize,
  leastAlign,
  isStorageOnly,
  isFloating,
  isIntegral,
  isSigned,
  scalarTypeName,

  -- * Host representations
  F16 (..),
  BF16 (..),

  -- * Reflection
  IsScalar (..),
  LeastAlign,
) where

import Data.Int (Int32, Int8)
import Data.Kind (Constraint, Type)
import Data.Text (Text)
import Data.Type.Bool (If)
import Data.Word (Word16, Word32, Word8)
import GHC.TypeLits (Nat, type (<=?))
import Numeric.Natural (Natural)

-- | The scalar types, as data.
data ScalarType
  = TF32
  | TF16
  | TBF16
  | TI32
  | TU32
  | TI8
  | TU8
  | TBool
  deriving stock (Eq, Ord, Show, Enum, Bounded)

-- | Size in bytes in memory and in parameter structs.
scalarSize :: ScalarType -> Int
scalarSize = \case
  TF32 -> 4
  TF16 -> 2
  TBF16 -> 2
  TI32 -> 4
  TU32 -> 4
  TI8 -> 1
  TU8 -> 1
  TBool -> 1

{- | The least alignment of an array of the type in memory, in bytes: 16,
or the type's size if larger. Buffers, tables and threadgroup arrays are
aligned to it at least.
-}
leastAlign :: ScalarType -> Natural
leastAlign t = max 16 (fromIntegral (scalarSize t))

-- | Types that have no arithmetic and must be widened first.
isStorageOnly :: ScalarType -> Bool
isStorageOnly t = t `elem` [TBF16, TI8, TU8]

isFloating :: ScalarType -> Bool
isFloating t = t `elem` [TF32, TF16, TBF16]

isIntegral :: ScalarType -> Bool
isIntegral t = t `elem` [TI32, TU32, TI8, TU8]

isSigned :: ScalarType -> Bool
isSigned t = t `elem` [TF32, TF16, TBF16, TI32, TI8]

-- | A short, stable name: in manifests, ABI descriptions and messages.
scalarTypeName :: ScalarType -> Text
scalarTypeName = \case
  TF32 -> "f32"
  TF16 -> "f16"
  TBF16 -> "bf16"
  TI32 -> "i32"
  TU32 -> "u32"
  TI8 -> "i8"
  TU8 -> "u8"
  TBool -> "bool"

-- | An IEEE 754 binary16 value, as its bits.
newtype F16 = F16 Word16
  deriving stock (Eq, Ord, Show)

-- | A bfloat16 value (the upper half of a binary32), as its bits.
newtype BF16 = BF16 Word16
  deriving stock (Eq, Ord, Show)

{- | The Haskell types that stand for scalar types in typed code: these
eight, the class closed (its superclass is this module's own).
-}
type IsScalar :: Type -> Constraint
class (Sealed a) => IsScalar a where
  scalarType :: ScalarType

  -- | Size in bytes, at the type level, for static resource checks.
  type ByteSize a :: Nat

-- | The superclass that keeps 'IsScalar' to the types this module gives it.
type Sealed :: Type -> Constraint
class Sealed a

instance Sealed Float

instance Sealed F16

instance Sealed BF16

instance Sealed Int32

instance Sealed Word32

instance Sealed Int8

instance Sealed Word8

instance Sealed Bool

instance IsScalar Float where
  scalarType = TF32
  type ByteSize Float = 4

instance IsScalar F16 where
  scalarType = TF16
  type ByteSize F16 = 2

instance IsScalar BF16 where
  scalarType = TBF16
  type ByteSize BF16 = 2

instance IsScalar Int32 where
  scalarType = TI32
  type ByteSize Int32 = 4

instance IsScalar Word32 where
  scalarType = TU32
  type ByteSize Word32 = 4

instance IsScalar Int8 where
  scalarType = TI8
  type ByteSize Int8 = 1

instance IsScalar Word8 where
  scalarType = TU8
  type ByteSize Word8 = 1

instance IsScalar Bool where
  scalarType = TBool
  type ByteSize Bool = 1

-- | 'leastAlign' at the type level.
type LeastAlign :: Type -> Nat
type LeastAlign a = If (ByteSize a <=? 16) 16 (ByteSize a)
