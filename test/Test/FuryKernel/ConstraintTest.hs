-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE UndecidableInstances #-}

{- | Each condition of a rule, as the named family both its 'Bool' form and
its enforced constraint are built from, agrees with the decider's finding
for it, sample by sample; and dynamic mode reflects exactly what the decider
accepts.
-}
module Test.FuryKernel.ConstraintTest (tests) where

import Control.Monad (forM_)
import Data.Kind (Constraint, Type)
import Data.Proxy (Proxy (..))
import Data.Text qualified as T
import Data.Word (Word32, Word8)
import FuryKernel.Constraint
import FuryKernel.Finding (Finding (..), Rule (..))
import FuryKernel.Target (Target (..), targetCaps)
import FuryKernel.Type (BF16, IsScalar (..), ScalarType (..))
import GHC.TypeNats (KnownNat, Nat, natVal)
import Numeric.Natural (Natural)
import Test.FuryKernel.Targets (T16, T32, T64, T8)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (assertBool, testCase, (@?=))

tests :: TestTree
tests =
  testGroup
    "Constraint"
    [ testCase "threads: each condition's family and the decider agree" $
        agree
          ["at least one", "a multiple of the SIMD width", "within the limit"]
          [ ("T8", threadSamples @T8 @ThreadSamples)
          , ("T16", threadSamples @T16 @ThreadSamples)
          , ("T32", threadSamples @T32 @ThreadSamples)
          , ("T64", threadSamples @T64 @ThreadSamples)
          ]
    , testCase "threadgroup arrays: each condition's family and the decider agree" $
        agree
          ["at least one", "within the threadgroup memory"]
          [ ("T8 f32", arraySamples @T8 @Float @ArraySamples)
          , ("T32 f32", arraySamples @T32 @Float @ArraySamples)
          , ("T32 bf16", arraySamples @T32 @BF16 @ArraySamples)
          , ("T64 u32", arraySamples @T64 @Word32 @ArraySamples)
          , ("Uneven f32", arraySamples @Uneven @Float @ArraySamples)
          , ("Uneven u8", arraySamples @Uneven @Word8 @ArraySamples)
          ]
    , testCase "threadgroup arrays: each takes its size rounded up to its alignment" $ do
        boolVal @(TgArrayOkB Uneven 4092 Float) @?= True
        boolVal @(TgArrayOkB Uneven 4093 Float) @?= False
        checkTgArray @Float (targetCaps @Uneven) 4093
          @?= [Finding ThreadgroupMemory "Threadgroup array of 4093 elements of 4 B = 16372 B, which takes 16384 B aligned to 16 B; the threadgroup memory of Uneven is 16380 B."]
    , testCase "alignments: each condition's family and the decider agree" $
        agree
          ["a power of two", "at least the least alignment"]
          [ ("f32", alignSamples @Float @AlignSamples)
          , ("bf16", alignSamples @BF16 @AlignSamples)
          , ("u8", alignSamples @Word8 @AlignSamples)
          ]
    , testCase "packs: each condition's family and the decider agree" $
        agree
          ["not bool", "at least two", "a power of two", "within a lane load", "aligned"]
          [ ("T32 bf16", packSamples @T32 @BF16 @PackSamples)
          , ("T32 f32", packSamples @T32 @Float @PackSamples)
          , ("T8 u8", packSamples @T8 @Word8 @PackSamples)
          , ("T64 bool", packSamples @T64 @Bool @PackSamples)
          ]
    , testCase "dynamic mode reflects what the decider accepts" $ do
        withThreads @T32 256 natVal @?= Right 256
        withThreads @T32 1000 natVal @?= Left (checkThreads (targetCaps @T32) 1000)
        withTgArray @T32 @Float 8192 natVal @?= Right 8192
        withTgArray @T32 @Float 8193 natVal @?= Left (checkTgArray @Float (targetCaps @T32) 8193)
        withPack @T32 @BF16 @16 8 natVal @?= Right 8
        withPack @T32 @BF16 @8 8 natVal @?= Left (checkPack (targetCaps @T32) TBF16 8 8)
    , testCase "messages carry the arithmetic" $ do
        checkThreads (targetCaps @T32) 1000
          @?= [Finding Threads "Threads per threadgroup on T32: 1000 is not a multiple of the SIMD width, 32."]
        checkThreads (targetCaps @T32) 2048
          @?= [Finding Threads "Threads per threadgroup on T32: 2048; the limit is 1024."]
        checkTgArray @Float (targetCaps @T32) 9216
          @?= [Finding ThreadgroupMemory "Threadgroup array of 9216 elements of 4 B = 36864 B, which takes 36864 B aligned to 16 B; the threadgroup memory of T32 is 32768 B."]
        checkTgTotal (targetCaps @T32) [("partial", 32), ("tile", 36864)]
          @?= [Finding ThreadgroupMemory "Threadgroup memory on T32, each array taking its size rounded up to its alignment: partial 32 B + tile 36864 B = 36896 B; the limit is 32768 B."]
        checkTgTotal (targetCaps @T32) [("partial", 32), ("tile", 16384)] @?= []
        checkBuffers (targetCaps @T32) 32
          @?= [Finding Buffers "Buffer bindings on T32: 32; the limit is 31."]
        checkPack (targetCaps @T32) TBF16 16 32
          @?= [Finding Packs "Pack of 16 elements of 2 B = 32 B; the widest lane load on T32 is 16 B."]
        checkPack (targetCaps @T32) TBF16 8 8
          @?= [Finding Packs "Pack of 8 elements of 2 B = 16 B from a buffer aligned to 8 B: align it to a multiple of 16 B."]
        checkPack (targetCaps @T32) TF32 3 16
          @?= [ Finding Packs "Pack of 3 elements of 4 B = 12 B, which is not a power of two."
              , Finding Packs "Pack of 3 elements of 4 B = 12 B from a buffer aligned to 16 B: align it to a multiple of 12 B."
              ]
        checkPack (targetCaps @T32) TBF16 8 16 @?= []
        checkAlign "Buffer x" TF32 8 @?= [Finding Alignment "Buffer x: an alignment of 8 B for elements of 4 B; the least is 16 B."]
        checkAlign "Table t" TBF16 24 @?= [Finding Alignment "Table t: an alignment of 24 B, which is not a power of two."]
        checkAlign "Buffer x" TU8 16 @?= []
    ]

{- | Each sample's conditions, the families' against the decider's, and
the rule's 'Bool' form their conjunction. Every condition must pass and fail
somewhere among the runs, or the agreement says nothing about it.
-}
agree :: (Show k) => [String] -> [(String, [(k, Bool, [Bool], [Bool])])] -> IO ()
agree conditions runs = do
  forM_ runs $ \(what, samples) -> forM_ samples $ \(n, whole, static, dynamic) -> do
    assertBool (what <> " at " <> show n <> ": families " <> show static <> ", decider " <> show dynamic) (static == dynamic)
    assertBool (what <> " at " <> show n <> ": the rule is its conditions together") (whole == and static)
  forM_ (zip [0 :: Int ..] conditions) $ \(i, c) -> do
    let seen = [s !! i | (_, samples) <- runs, (_, _, s, _) <- samples]
    assertBool (c <> ": passes in some sample") (or seen)
    assertBool (c <> ": fails in some sample") (not (and seen))

-- | Whether the decider's findings hold none with this in its message.
none :: [Finding] -> T.Text -> Bool
none findings part = not (any (T.isInfixOf part . (.findingMessage)) findings)

type ThreadSamples :: [Nat]
type ThreadSamples = '[0, 1, 7, 8, 16, 31, 32, 33, 48, 64, 96, 128, 256, 512, 1000, 1024, 1025, 1056, 2048]

type ArraySamples :: [Nat]
type ArraySamples = '[0, 1, 8, 1024, 4092, 4093, 4095, 4096, 8192, 8193, 16380, 16381, 16384, 16385, 32768]

type AlignSamples :: [Nat]
type AlignSamples = '[0, 1, 2, 4, 8, 12, 15, 16, 17, 24, 32, 48, 64, 4096]

-- | A target whose threadgroup memory is not a multiple of 16 B, so that an array's padding can decide.
data Uneven

instance Target Uneven where
  type TargetName Uneven = "Uneven"
  type SimdWidth Uneven = 32
  type MaxThreads Uneven = 1024
  type TgMemBytes Uneven = 16380
  type MaxBuffers Uneven = 31
  type LaneLoadMax Uneven = 16

-- | Elements and the buffer's alignment.
type PackSamples :: [(Nat, Nat)]
type PackSamples =
  '[ '(0, 16)
   , '(1, 16)
   , '(2, 2)
   , '(2, 4)
   , '(2, 16)
   , '(3, 16)
   , '(4, 4)
   , '(4, 8)
   , '(4, 16)
   , '(6, 16)
   , '(8, 8)
   , '(8, 16)
   , '(8, 48)
   , '(16, 16)
   , '(16, 32)
   , '(32, 64)
   ]

type ThreadSamplesOf :: Type -> [Nat] -> Constraint
class ThreadSamplesOf t ns where
  threadSamples :: [(Natural, Bool, [Bool], [Bool])]

instance ThreadSamplesOf t '[] where
  threadSamples = []

instance
  ( Target t
  , KnownNat n
  , KnownBool (ThreadsOkB t n)
  , KnownBool (ThreadsSome n)
  , KnownBool (ThreadsWhole t n)
  , KnownBool (ThreadsWithin t n)
  , ThreadSamplesOf t ns
  ) =>
  ThreadSamplesOf t (n ': ns)
  where
  threadSamples =
    ( n
    , boolVal @(ThreadsOkB t n)
    , [boolVal @(ThreadsSome n), boolVal @(ThreadsWhole t n), boolVal @(ThreadsWithin t n)]
    , map (none (checkThreads (targetCaps @t) n)) ["at least one thread", "is not a multiple of the SIMD width", "; the limit is"]
    )
      : threadSamples @t @ns
   where
    n = natVal (Proxy @n)

type ArraySamplesOf :: Type -> Type -> [Nat] -> Constraint
class ArraySamplesOf t a ns where
  arraySamples :: [(Natural, Bool, [Bool], [Bool])]

instance ArraySamplesOf t a '[] where
  arraySamples = []

instance
  ( Target t
  , IsScalar a
  , KnownNat n
  , KnownBool (TgArrayOkB t n a)
  , KnownBool (TgArraySome n)
  , KnownBool (TgArrayWithin t n a)
  , ArraySamplesOf t a ns
  ) =>
  ArraySamplesOf t a (n ': ns)
  where
  arraySamples =
    ( n
    , boolVal @(TgArrayOkB t n a)
    , [boolVal @(TgArraySome n), boolVal @(TgArrayWithin t n a)]
    , map (none (checkTgArray @a (targetCaps @t) n)) ["at least one element", "; the threadgroup memory of"]
    )
      : arraySamples @t @a @ns
   where
    n = natVal (Proxy @n)

type AlignSamplesOf :: Type -> [Nat] -> Constraint
class AlignSamplesOf a ns where
  alignSamples :: [(Natural, Bool, [Bool], [Bool])]

instance AlignSamplesOf a '[] where
  alignSamples = []

instance
  ( IsScalar a
  , KnownNat align
  , KnownBool (AlignOkB a align)
  , KnownBool (AlignPowerOfTwo align)
  , KnownBool (AlignAtLeast a align)
  , AlignSamplesOf a ns
  ) =>
  AlignSamplesOf a (align ': ns)
  where
  alignSamples =
    ( align
    , boolVal @(AlignOkB a align)
    , [boolVal @(AlignPowerOfTwo align), boolVal @(AlignAtLeast a align)]
    , map (none (checkAlign "Buffer b" (scalarType @a) align)) ["not a power of two", "; the least is"]
    )
      : alignSamples @a @ns
   where
    align = natVal (Proxy @align)

type PackSamplesOf :: Type -> Type -> [(Nat, Nat)] -> Constraint
class PackSamplesOf t a ss where
  packSamples :: [((Natural, Natural), Bool, [Bool], [Bool])]

instance PackSamplesOf t a '[] where
  packSamples = []

instance
  ( Target t
  , IsScalar a
  , KnownNat n
  , KnownNat align
  , KnownBool (PackOkB t n a align)
  , KnownBool (PackNotBool a)
  , KnownBool (PackSome n)
  , KnownBool (PackPowerOfTwo n a)
  , KnownBool (PackWithin t n a)
  , KnownBool (PackAligned n a align)
  , PackSamplesOf t a ss
  ) =>
  PackSamplesOf t a ('(n, align) ': ss)
  where
  packSamples =
    ( (n, align)
    , boolVal @(PackOkB t n a align)
    , [boolVal @(PackNotBool a), boolVal @(PackSome n), boolVal @(PackPowerOfTwo n a), boolVal @(PackWithin t n a), boolVal @(PackAligned n a align)]
    , map (none (checkPack (targetCaps @t) (scalarType @a) n align)) ["No packs of bool", "at least two elements", "not a power of two", "the widest lane load", "from a buffer aligned to"]
    )
      : packSamples @t @a @ss
   where
    n = natVal (Proxy @n)
    align = natVal (Proxy @align)
