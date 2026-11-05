-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

{- | The resource rules, each written once in two forms.

* As a type family, for static mode: a kernel that asks for
  @threads \@1000@ on a target with 32-wide SIMD groups does not compile,
  and the message carries the arithmetic.
* As a term-level decider, for the gate and for dynamic mode, where the
  numbers are only known at run time (tuning enumerates them).

Each condition of a rule is a named family, which both its 'Bool' form and
the constraint static mode enforces are built from, so the test suite's
agreement of the 'Bool' forms and the deciders on sample points covers the
enforced rule. Dynamic mode then reflects a runtime number into the static
constraint ('withThreads', 'withTgArray'): the decider vouches for it, so
code written against the static API also serves runtime parameters.

The static constraints are classes without methods, whose single instance
carries the rule: their evidence is empty, which is what makes the
reflection sound.
-}
module FuryKernel.Constraint (
  -- * Threads per threadgroup
  ThreadsOk,
  ThreadsOkB,
  ThreadsSome,
  ThreadsWhole,
  ThreadsWithin,
  checkThreads,
  withThreads,

  -- * Threadgroup arrays
  TgArrayOk,
  TgArrayOkB,
  TgArraySome,
  TgArrayWithin,
  RoundUp,
  checkTgArray,
  checkTgArrayOf,
  withTgArray,

  -- * Alignments of arrays in device memory
  AlignOk,
  AlignOkB,
  AlignPowerOfTwo,
  AlignAtLeast,
  PowerOfTwo,
  checkAlign,

  -- * Packs
  PackOk,
  PackOkB,
  PackNotBool,
  PackSome,
  PackPowerOfTwo,
  PackWithin,
  PackAligned,
  checkPack,
  withPack,

  -- * Totals, checked by the gate only
  checkTgTotal,
  checkBuffers,

  -- * Reflection helpers
  KnownBool (..),
) where

import Data.Kind (Constraint, Type)
import Data.Proxy (Proxy (..))
import Data.Text (Text)
import Data.Text qualified as T
import Data.Type.Bool (Not, type (&&))
import Data.Type.Equality (type (==))
import FuryKernel.Finding (Finding (..), Rule (..))
import FuryKernel.Target (Target (..), TargetCaps (..), targetCaps)
import FuryKernel.Type (IsScalar (..), LeastAlign, ScalarType (..), leastAlign, scalarSize)
import GHC.TypeError (Assert, ErrorMessage (..), TypeError)
import GHC.TypeNats (Div, KnownNat, Mod, Nat, SomeNat (..), natVal, someNatVal, type (*), type (+), type (-), type (<=?))
import Numeric.Natural (Natural)
import Unsafe.Coerce (unsafeCoerce)

-- | Reflects a type-level 'Bool', for the agreement tests.
type KnownBool :: Bool -> Constraint
class KnownBool b where
  boolVal :: Bool

instance KnownBool 'True where
  boolVal = True

instance KnownBool 'False where
  boolVal = False

------------------------------------------------------------------------------
-- Threads

-- | The rule's conditions: at least one thread, a multiple of the SIMD width, within the limit.
type ThreadsSome :: Nat -> Bool
type ThreadsSome n = 1 <=? n

type ThreadsWhole :: Type -> Nat -> Bool
type ThreadsWhole t n = Mod n (SimdWidth t) == 0

type ThreadsWithin :: Type -> Nat -> Bool
type ThreadsWithin t n = n <=? MaxThreads t

-- | The rule as a 'Bool'.
type ThreadsOkB :: Type -> Nat -> Bool
type ThreadsOkB t n = ThreadsSome n && ThreadsWhole t n && ThreadsWithin t n

-- | @n@ threads per threadgroup are legal on @t@.
type ThreadsOk :: Type -> Nat -> Constraint
class ThreadsOk t n

instance
  ( Assert
      (ThreadsSome n)
      (TypeError ('Text "A threadgroup needs at least one thread."))
  , Assert
      (ThreadsWhole t n)
      ( TypeError
          ( 'Text "Threads per threadgroup on "
              ':<>: 'Text (TargetName t)
              ':<>: 'Text ": "
              ':<>: 'ShowType n
              ':<>: 'Text " is not a multiple of the SIMD width, "
              ':<>: 'ShowType (SimdWidth t)
              ':<>: 'Text "."
          )
      )
  , Assert
      (ThreadsWithin t n)
      ( TypeError
          ( 'Text "Threads per threadgroup on "
              ':<>: 'Text (TargetName t)
              ':<>: 'Text ": "
              ':<>: 'ShowType n
              ':<>: 'Text "; the limit is "
              ':<>: 'ShowType (MaxThreads t)
              ':<>: 'Text "."
          )
      )
  ) =>
  ThreadsOk t n

checkThreads :: TargetCaps -> Natural -> [Finding]
checkThreads caps _
  | caps.capSimdWidth == 0 = [Finding Target (caps.capName <> "'s SIMD groups have no lanes.")]
checkThreads caps n =
  [ Finding Threads msg
  | (False, msg) <-
      [ (n >= 1, "A threadgroup needs at least one thread.")
      ,
        ( n `mod` caps.capSimdWidth == 0
        , prefix
            <> " is not a multiple of the SIMD width, "
            <> showT caps.capSimdWidth
            <> "."
        )
      , (n <= caps.capMaxThreads, prefix <> "; the limit is " <> showT caps.capMaxThreads <> ".")
      ]
  ]
 where
  prefix = "Threads per threadgroup on " <> caps.capName <> ": " <> showT n

-- | Dynamic mode: run static-mode code for a thread count known at run time.
withThreads ::
  forall t r.
  (Target t) =>
  Natural ->
  (forall n. (KnownNat n, ThreadsOk t n) => Proxy n -> r) ->
  Either [Finding] r
withThreads n k = case checkThreads (targetCaps @t) n of
  [] -> case someNatVal n of
    SomeNat (p :: Proxy n) -> case vouch @(ThreadsOk t n) of
      Dict -> Right (k p)
  findings -> Left findings

------------------------------------------------------------------------------
-- Threadgroup arrays

{- | The rule's conditions: at least one element, within the threadgroup
memory, the array taking its size rounded up to its alignment ('LeastAlign')
as in the kernel's layout ('FuryKernel.IR.tgLayout').
-}
type TgArraySome :: Nat -> Bool
type TgArraySome n = 1 <=? n

type TgArrayWithin :: Type -> Nat -> Type -> Bool
type TgArrayWithin t n a = RoundUp (n * ByteSize a) (LeastAlign a) <=? TgMemBytes t

-- | @x@ rounded up to a multiple of @m@, which is positive.
type RoundUp :: Nat -> Nat -> Nat
type RoundUp x m = Div (x + m - 1) m * m

-- | The rule as a 'Bool'.
type TgArrayOkB :: Type -> Nat -> Type -> Bool
type TgArrayOkB t n a = TgArraySome n && TgArrayWithin t n a

-- | An array of @n@ elements of @a@ fits the threadgroup memory of @t@.
type TgArrayOk :: Type -> Nat -> Type -> Constraint
class TgArrayOk t n a

instance
  ( Assert
      (TgArraySome n)
      (TypeError ('Text "A threadgroup array needs at least one element."))
  , Assert
      (TgArrayWithin t n a)
      ( TypeError
          ( 'Text "Threadgroup array of "
              ':<>: 'ShowType n
              ':<>: 'Text " elements of "
              ':<>: 'ShowType (ByteSize a)
              ':<>: 'Text " B = "
              ':<>: 'ShowType (n * ByteSize a)
              ':<>: 'Text " B, which takes "
              ':<>: 'ShowType (RoundUp (n * ByteSize a) (LeastAlign a))
              ':<>: 'Text " B aligned to "
              ':<>: 'ShowType (LeastAlign a)
              ':<>: 'Text " B; the threadgroup memory of "
              ':<>: 'Text (TargetName t)
              ':<>: 'Text " is "
              ':<>: 'ShowType (TgMemBytes t)
              ':<>: 'Text " B."
          )
      )
  ) =>
  TgArrayOk t n a

checkTgArray :: forall a. (IsScalar a) => TargetCaps -> Natural -> [Finding]
checkTgArray caps = checkTgArrayOf caps (scalarType @a)

-- | 'checkTgArray' for an element type known as a term.
checkTgArrayOf :: TargetCaps -> ScalarType -> Natural -> [Finding]
checkTgArrayOf caps t n =
  [ Finding ThreadgroupMemory msg
  | (False, msg) <-
      [ (n >= 1, "A threadgroup array needs at least one element.")
      ,
        ( taken <= caps.capTgMemBytes
        , "Threadgroup array of "
            <> showT n
            <> " elements of "
            <> showT size
            <> " B = "
            <> showT bytes
            <> " B, which takes "
            <> showT taken
            <> " B aligned to "
            <> showT align
            <> " B; the threadgroup memory of "
            <> caps.capName
            <> " is "
            <> showT caps.capTgMemBytes
            <> " B."
        )
      ]
  ]
 where
  size = fromIntegral (scalarSize t) :: Natural
  bytes = n * size
  align = leastAlign t
  taken = (bytes + align - 1) `div` align * align

-- | Dynamic mode: run static-mode code for an array size known at run time.
withTgArray ::
  forall t a r.
  (Target t, IsScalar a) =>
  Natural ->
  (forall n. (KnownNat n, TgArrayOk t n a) => Proxy n -> r) ->
  Either [Finding] r
withTgArray n k = case checkTgArray @a (targetCaps @t) n of
  [] -> case someNatVal n of
    SomeNat (p :: Proxy n) -> case vouch @(TgArrayOk t n a) of
      Dict -> Right (k p)
  findings -> Left findings

------------------------------------------------------------------------------
-- Alignments

type PowerOfTwo :: Nat -> Bool
type family PowerOfTwo n where
  PowerOfTwo 0 = 'False
  PowerOfTwo 1 = 'True
  PowerOfTwo n = Mod n 2 == 0 && PowerOfTwo (Div n 2)

-- | The rule's conditions: a power of two, at least 'LeastAlign'.
type AlignPowerOfTwo :: Nat -> Bool
type AlignPowerOfTwo align = PowerOfTwo align

type AlignAtLeast :: Type -> Nat -> Bool
type AlignAtLeast a align = LeastAlign a <=? align

-- | The rule as a 'Bool'.
type AlignOkB :: Type -> Nat -> Bool
type AlignOkB a align = AlignPowerOfTwo align && AlignAtLeast a align

{- | The start of an array of @a@ in device memory may be aligned to
@align@ bytes: a power of two, and at least 'LeastAlign' (16, or the size
of @a@ if larger).
-}
type AlignOk :: Type -> Nat -> Constraint
class AlignOk a align

instance
  ( Assert
      (AlignPowerOfTwo align)
      ( TypeError
          ( 'Text "An alignment of "
              ':<>: 'ShowType align
              ':<>: 'Text " B, which is not a power of two."
          )
      )
  , Assert
      (AlignAtLeast a align)
      ( TypeError
          ( 'Text "An alignment of "
              ':<>: 'ShowType align
              ':<>: 'Text " B for elements of "
              ':<>: 'ShowType (ByteSize a)
              ':<>: 'Text " B; the least is "
              ':<>: 'ShowType (LeastAlign a)
              ':<>: 'Text " B."
          )
      )
  ) =>
  AlignOk a align

-- | The rule for an array of elements of the type, which @subject@ names in the messages.
checkAlign :: Text -> ScalarType -> Natural -> [Finding]
checkAlign subject t align =
  [ Finding Alignment (subject <> ": an alignment of " <> showT align <> " B" <> msg)
  | (False, msg) <-
      [ (powerOfTwo align, ", which is not a power of two.")
      , (align >= leastAlign t, " for elements of " <> showT (scalarSize t) <> " B; the least is " <> showT (leastAlign t) <> " B.")
      ]
  ]

------------------------------------------------------------------------------
-- Packs

-- | The rule's conditions: no bool, at least two elements, a power of two in size, within a lane load, aligned.
type PackNotBool :: Type -> Bool
type PackNotBool a = Not (a == Bool)

type PackSome :: Nat -> Bool
type PackSome n = 2 <=? n

type PackPowerOfTwo :: Nat -> Type -> Bool
type PackPowerOfTwo n a = PowerOfTwo (n * ByteSize a)

type PackWithin :: Type -> Nat -> Type -> Bool
type PackWithin t n a = n * ByteSize a <=? LaneLoadMax t

type PackAligned :: Nat -> Type -> Nat -> Bool
type PackAligned n a align = Divides (n * ByteSize a) align

-- | A size the alignment is a multiple of: never 0.
type Divides :: Nat -> Nat -> Bool
type family Divides size align where
  Divides 0 _ = 'False
  Divides size align = Mod align size == 0

-- | The rule as a 'Bool'.
type PackOkB :: Type -> Nat -> Type -> Nat -> Bool
type PackOkB t n a align =
  PackNotBool a && PackSome n && PackPowerOfTwo n a && PackWithin t n a && PackAligned n a align

{- | A pack of @n@ elements of @a@, loaded or stored by one lane at once
from a buffer whose start is aligned to @align@ bytes, is legal on @t@.

Packs are indexed in packs, not elements, so a pack is aligned to its size
whenever the buffer's start is: the rule needs the size to divide the
buffer's alignment, and to be a power of two no wider than the target's
widest lane load.
-}
type PackOk :: Type -> Nat -> Type -> Nat -> Constraint
class PackOk t n a align

instance
  ( Assert
      (PackNotBool a)
      (TypeError ('Text "No packs of bool: its layout in memory is the target's."))
  , Assert
      (PackSome n)
      (TypeError ('Text "A pack needs at least two elements."))
  , Assert
      (PackPowerOfTwo n a)
      ( TypeError
          ( 'Text "Pack of "
              ':<>: 'ShowType n
              ':<>: 'Text " elements of "
              ':<>: 'ShowType (ByteSize a)
              ':<>: 'Text " B = "
              ':<>: 'ShowType (n * ByteSize a)
              ':<>: 'Text " B, which is not a power of two."
          )
      )
  , Assert
      (PackWithin t n a)
      ( TypeError
          ( 'Text "Pack of "
              ':<>: 'ShowType n
              ':<>: 'Text " elements of "
              ':<>: 'ShowType (ByteSize a)
              ':<>: 'Text " B = "
              ':<>: 'ShowType (n * ByteSize a)
              ':<>: 'Text " B; the widest lane load on "
              ':<>: 'Text (TargetName t)
              ':<>: 'Text " is "
              ':<>: 'ShowType (LaneLoadMax t)
              ':<>: 'Text " B."
          )
      )
  , Assert
      (PackAligned n a align)
      ( TypeError
          ( 'Text "Pack of "
              ':<>: 'ShowType n
              ':<>: 'Text " elements of "
              ':<>: 'ShowType (ByteSize a)
              ':<>: 'Text " B = "
              ':<>: 'ShowType (n * ByteSize a)
              ':<>: 'Text " B from a buffer aligned to "
              ':<>: 'ShowType align
              ':<>: 'Text " B: align it to a multiple of "
              ':<>: 'ShowType (n * ByteSize a)
              ':<>: 'Text " B."
          )
      )
  ) =>
  PackOk t n a align

checkPack :: TargetCaps -> ScalarType -> Natural -> Natural -> [Finding]
checkPack caps t n align =
  [ Finding Packs msg
  | (False, msg) <-
      [ (t /= TBool, "No packs of bool: its layout in memory is the target's.")
      , (n >= 2, "A pack needs at least two elements.")
      , (powerOfTwo bytes, prefix <> ", which is not a power of two.")
      ,
        ( bytes <= caps.capLaneLoadMax
        , prefix <> "; the widest lane load on " <> caps.capName <> " is " <> showT caps.capLaneLoadMax <> " B."
        )
      ,
        ( bytes /= 0 && align `mod` bytes == 0
        , if bytes == 0
            then prefix <> " from a buffer aligned to " <> showT align <> " B: a pack of no bytes has nothing to align."
            else
              prefix
                <> " from a buffer aligned to "
                <> showT align
                <> " B: align it to a multiple of "
                <> showT bytes
                <> " B."
        )
      ]
  ]
 where
  size = fromIntegral (scalarSize t) :: Natural
  bytes = n * size
  prefix = "Pack of " <> showT n <> " elements of " <> showT size <> " B = " <> showT bytes <> " B"

-- | Dynamic mode: run static-mode code for a pack width known at run time.
withPack ::
  forall t a align r.
  (Target t, IsScalar a, KnownNat align) =>
  Natural ->
  (forall n. (KnownNat n, PackOk t n a align) => Proxy n -> r) ->
  Either [Finding] r
withPack n k = case checkPack (targetCaps @t) (scalarType @a) n (natVal (Proxy @align)) of
  [] -> case someNatVal n of
    SomeNat (p :: Proxy n) -> case vouch @(PackOk t n a align) of
      Dict -> Right (k p)
  findings -> Left findings

------------------------------------------------------------------------------
-- Totals

{- | The threadgroup arrays of a kernel together, by name and the bytes each
takes in the kernel's layout ('FuryKernel.IR.tgBytes').

The message carries the arithmetic, as in
@Threadgroup memory on AppleGPU9, each array taking its size rounded up to
its alignment: partial 32 B + tile 36864 B = 36896 B; the limit is 32768 B.@
-}
checkTgTotal :: TargetCaps -> [(Text, Natural)] -> [Finding]
checkTgTotal caps arrays
  | total <= caps.capTgMemBytes = []
  | otherwise =
      [ Finding ThreadgroupMemory $
          "Threadgroup memory on "
            <> caps.capName
            <> ", each array taking its size rounded up to its alignment: "
            <> T.intercalate " + " [name <> " " <> showT bytes <> " B" | (name, bytes) <- arrays]
            <> " = "
            <> showT total
            <> " B; the limit is "
            <> showT caps.capTgMemBytes
            <> " B."
      ]
 where
  total = sum (map snd arrays)

-- | Buffer bindings, the parameter struct counted as one.
checkBuffers :: TargetCaps -> Natural -> [Finding]
checkBuffers caps n
  | n <= caps.capMaxBuffers = []
  | otherwise =
      [ Finding Buffers $
          "Buffer bindings on "
            <> caps.capName
            <> ": "
            <> showT n
            <> "; the limit is "
            <> showT caps.capMaxBuffers
            <> "."
      ]

------------------------------------------------------------------------------
-- Reflection

data Dict (c :: Constraint) where
  Dict :: (c) => Dict c

{- | Evidence for a rule the decider has accepted. Sound because the rule
classes have no methods and no superclasses: their evidence is empty, the
same as the unit constraint's.
-}
vouch :: forall c. Dict c
vouch = unsafeCoerce (Dict :: Dict ())

showT :: (Show a) => a -> Text
showT = T.pack . show

powerOfTwo :: Natural -> Bool
powerOfTwo x = x == 1 || (x > 1 && even x && powerOfTwo (x `div` 2))
