-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE AllowAmbiguousTypes #-}

-- | The ABI: layout, what its hash covers and what it doesn't; manifests; overrides.
module Test.FuryKernel.AbiTest (tests) where

import Control.Monad (when)
import Data.Text (Text)
import Data.Text qualified as T
import Data.Word (Word32)
import FuryKernel.Abi
import FuryKernel.Build (Exp, apart, buffer, dim, groups, kernel, kernelIR, load, narrow, param, store, strided, threads, widen)
import FuryKernel.Build qualified as B
import FuryKernel.Check (Checked, check)
import FuryKernel.Constraint (AlignOk)
import FuryKernel.Finding (Finding (..), Rule (..))
import FuryKernel.IR
import FuryKernel.Json (renderJson)
import FuryKernel.Manifest (kernelManifest)
import FuryKernel.Type (BF16, ScalarType (..))
import GHC.TypeNats (KnownNat)
import Numeric.Natural (Natural)
import Test.FuryKernel.Kernels (rmsnorm, scale)
import Test.FuryKernel.Targets (T32)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (assertBool, assertFailure, testCase, (@?=))

tests :: TestTree
tests =
  testGroup
    "Abi"
    [ testCase "FNV-1a reference vectors" $ do
        fnv1a64 "" @?= 0xcbf29ce484222325
        fnv1a64 "a" @?= 0xaf63dc4c8601ec8c
        fnv1a64 "foobar" @?= 0x85944171f73967e8
    , testCase "the parameter struct is laid out as C lays it out" $ do
        let a = abiOf (withParams [("a", TU8), ("b", TF32), ("c", TF16), ("d", TU32)])
        [(f.fieldName, f.fieldOffset) | f <- a.abiParams] @?= [("a", 0), ("b", 4), ("c", 8), ("d", 12)]
        a.abiParamsSize @?= 16
        (abiOf (withParams [])).abiParamsSize @?= 0
        (abiOf (withParams [("a", TF32), ("b", TU8)])).abiParamsSize @?= 8
    , testCase "the description the hash covers" $
        abiDescription (abiOf (kernelIR (rmsnorm @T32 2816)))
          @?= T.unlines
            [ "abi 1"
            , "buffer x read bf16 align 16"
            , "buffer w read bf16 align 16"
            , "buffer y write bf16 align 16"
            , "param rows u32 at 0"
            , "param eps f32 at 4"
            , "params size 8"
            ]
    , testCase "the ABI hash covers the interface, the IR hash the body too" $ do
        let k = kernelIR (rmsnorm @T32 2816)
            renamed = k {kernelBuffers = [b {bufName = if b.bufName == "w" then "gamma" else b.bufName} | b <- k.kernelBuffers]}
            retyped = k {kernelParams = [p {paramType = if p.paramName == "eps" then TF16 else p.paramType} | p <- k.kernelParams]}
            realigned = k {kernelBuffers = [b {bufAlign = 32} | b <- k.kernelBuffers]}
            resized = k {kernelBuffers = [b {bufExtent = ELit (LU32 1)} | b <- k.kernelBuffers]}
            rebodied = k {kernelBody = []}
        assertBool "a renamed buffer" ((abiOf renamed).abiHash /= (abiOf k).abiHash)
        assertBool "a retyped parameter" ((abiOf retyped).abiHash /= (abiOf k).abiHash)
        assertBool "a realigned buffer" ((abiOf realigned).abiHash /= (abiOf k).abiHash)
        (abiOf resized).abiHash @?= (abiOf k).abiHash
        (abiOf rebodied).abiHash @?= (abiOf k).abiHash
        one <- gate (storing 1)
        two <- gate (storing 2)
        assertBool "the IR hash sees the body" (irHash one /= irHash two)
    , testCase "the IR and ABI hashes of a kernel, pinned: what changes them changes every committed hash" $ do
        c <- gate (scale @T32)
        irHash c @?= 0x60c50bd78967a3d0
        (abiOf (kernelIR (scale @T32))).abiHash @?= 0xe99ad54061a304ff
    , testCase "a manifest" $ case check (scale @T32) of
        Left fs -> assertFailure (show fs)
        Right c -> do
          let a = abiOf (kernelIR (scale @T32))
          renderJson (kernelManifest c)
            @?= T.unlines
              [ "{"
              , "  \"format\": \"furykernel-manifest\","
              , "  \"version\": 1,"
              , "  \"target\": \"T32\","
              , "  \"name\": \"scale\","
              , "  \"abi_hash\": \"" <> hex64 a.abiHash <> "\","
              , "  \"ir_hash\": \"" <> hex64 (irHash c) <> "\","
              , "  \"threads\": 64,"
              , "  \"groups\": \"1u\","
              , "  \"buffers\": ["
              , "    {"
              , "      \"name\": \"x\","
              , "      \"access\": \"read\","
              , "      \"type\": \"bf16\","
              , "      \"align\": 16,"
              , "      \"extent\": \"$n\""
              , "    },"
              , "    {"
              , "      \"name\": \"y\","
              , "      \"access\": \"write\","
              , "      \"type\": \"bf16\","
              , "      \"align\": 16,"
              , "      \"extent\": \"$n\""
              , "    }"
              , "  ],"
              , "  \"params\": ["
              , "    {"
              , "      \"name\": \"n\","
              , "      \"type\": \"u32\","
              , "      \"offset\": 0,"
              , "      \"dim\": {"
              , "        \"min\": 1,"
              , "        \"max\": 65536,"
              , "        \"multiple_of\": 1"
              , "      }"
              , "    },"
              , "    {"
              , "      \"name\": \"s\","
              , "      \"type\": \"f32\","
              , "      \"offset\": 4"
              , "    }"
              , "  ],"
              , "  \"params_size\": 8,"
              , "  \"threadgroup_memory\": 0"
              , "}"
              ]
    , testCase "an override: the kernel's ABI, and a contract at least as permissive" $ do
        portable <- gate (scale @T32)
        let override name r extent apartY = gate (scaleLike name r extent apartY) >>= overrides portable . pure
        override "scale_v8" (DimRange 8 65536 8) (+ 8) False
          >>= ( @?=
                  [ "No override of scale takes n = 1, which it takes."
                  , "The override scale_v8's extent of x, n + 8, is larger than scale's, n, at n = 8: 16 against 8."
                  , "The override scale_v8's extent of y, n + 8, is larger than scale's, n, at n = 8: 16 against 8."
                  ]
              )
        override "scale_wide" (DimRange 0 131072 1) id False >>= (@?= [])
        override "scale_more" (DimRange 1 65536 1) (+ 1) False
          >>= ( @?=
                  [ "The override scale_more's extent of x, n + 1, is larger than scale's, n, at n = 1: 2 against 1."
                  , "The override scale_more's extent of y, n + 1, is larger than scale's, n, at n = 1: 2 against 1."
                  ]
              )
        override "scale_apart" (DimRange 1 65536 1) id True >>= (@?= ["The override scale_apart declares y apart, and scale doesn't."])
        realigned <- gate (B.kernel "scale_32" (scaleSig @32 (DimRange 1 65536 1) id False))
        overrides portable [realigned]
          >>= (@?= ["The override scale_32 has another ABI than scale: buffer x read bf16 align 32 where scale has buffer x read bf16 align 16."])
        scalarN <- gate scaleScalar
        overrides portable [scalarN] >>= (@?= ["The override scale_scalar takes n as a scalar, and scale as a dimension."])
        overrides portable [] >>= (@?= ["No override replaces scale."])
    , testCase "overrides together: buckets that take every legal value" $ do
        portable <- gate (rows "rows" 16 (* 8))
        buckets <- traverse (\b -> gate (rows ("rows_m" <> T.pack (show b)) b (* 8))) [1, 2, 4, 8, 16 :: Natural]
        overrides portable buckets >>= (@?= [])
        overrides portable (take 4 buckets) >>= (@?= ["No override of rows takes m = 9, which it takes."])
        overrides portable (drop 1 buckets) >>= (@?= [])
        wide <- gate (rows "rows_m4" 4 (\m -> (m + 1) * 8))
        overrides portable (take 2 buckets <> [wide] <> drop 3 buckets)
          >>= (@?= ["The override rows_m4's extent of y, (m + 1) * 8, is larger than rows's, m * 8, at m = 1: 16 against 8."])
    ]

-- | A kernel through the gate.
gate :: B.Kernel T32 -> IO (Checked T32)
gate k = either (\fs -> assertFailure (show fs) >> error "refused") pure (check k)

-- | A kernel whose threads each store @v@ to an element of their own.
storing :: Exp Float -> B.Kernel T32
storing v = kernel "storing" $ do
  y <- buffer @B.WriteOnly @16 @Float "y" 32
  threads @32
  groups 1
  pure . pure $ store y B.threadIndex v

-- | The findings on overrides of a kernel, all of the rule's, by message.
overrides :: Checked T32 -> [Checked T32] -> IO [Text]
overrides k os = do
  let fs = checkOverrides k os
  when (any ((/= Overrides) . (.findingRule)) fs) $ assertFailure (show fs)
  pure (map (.findingMessage) fs)

-- | 'scale' with another range, extent and apart: what an override may change.
scaleLike :: Name -> DimRange -> (Exp Word32 -> Exp Word32) -> Bool -> B.Kernel T32
scaleLike name r extent apartY = kernel name (scaleSig @16 r extent apartY)

scaleSig :: forall align. (KnownNat align, AlignOk BF16 align) => DimRange -> (Exp Word32 -> Exp Word32) -> Bool -> B.Sig T32 (B.Scope T32 (B.K B.Uniform ()))
scaleSig r extent apartY = do
  n <- dim "n" r
  s <- param @Float "s"
  x <- buffer @B.ReadOnly @align @BF16 "x" (extent n)
  y <- buffer @B.WriteOnly @16 @BF16 "y" (extent n)
  when apartY (apart y)
  threads @64
  groups 1
  pure . pure $
    strided "i" (extent n) $
      \i -> store y i (narrow @BF16 (widen @Float (load x i) * s))

-- | 'scale' with n a scalar, over 64 elements.
scaleScalar :: B.Kernel T32
scaleScalar = kernel "scale_scalar" $ do
  _ <- param @Word32 "n"
  s <- param @Float "s"
  x <- buffer @B.ReadOnly @16 @BF16 "x" 64
  y <- buffer @B.WriteOnly @16 @BF16 "y" 64
  threads @64
  groups 1
  pure . pure $
    strided "i" 64 $
      \i -> store y i (narrow @BF16 (widen @Float (load x i) * s))

-- | y = x over m rows of eight, for m up to a bucket: decode's bucketed kernels.
rows :: Name -> Natural -> (Exp Word32 -> Exp Word32) -> B.Kernel T32
rows name bucket extentY = kernel name $ do
  m <- dim "m" (DimRange 1 bucket 1)
  x <- buffer @B.ReadOnly @16 @Float "x" (m * 8)
  y <- buffer @B.WriteOnly @16 @Float "y" (extentY m)
  threads @32
  groups 1
  pure . pure $ strided "i" (m * 8) $ \i -> store y i (load x i)

withParams :: [(Name, ScalarType)] -> Kernel
withParams ps =
  Kernel
    { kernelName = "k"
    , kernelThreads = 32
    , kernelSimdWidth = 32
    , kernelGroups = ELit (LU32 1)
    , kernelParams = [Param n t Scalar | (n, t) <- ps]
    , kernelBuffers = []
    , kernelTables = []
    , kernelTgArrays = []
    , kernelBody = []
    }
