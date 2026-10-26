-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The CPU target's Rust, golden. That it computes what the interpreter
does is checked where Rust runs: by the runtimes that compile it.
-}
module Test.FuryKernel.CpuTest (tests) where

import Control.Monad (void)
import Data.Text qualified as T
import Data.Word (Word32)
import FuryKernel
import FuryKernel.Abi (abiOf, hex64, irHash)
import FuryKernel.Abi qualified as Abi
import FuryKernel.Backend (Backend (..), Lowered (..), LoweredKernel (..), compileOptions)
import FuryKernel.Cpu (Cpu, cpu, rustLanguage)
import FuryKernel.Finding (Finding (..), Rule (..))
import FuryKernel.Json (renderJson)
import FuryKernel.Manifest (kernelManifest)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (assertBool, assertFailure, testCase, (@?=))

tests :: TestTree
tests =
  testGroup
    "Cpu"
    [ testCase "a kernel's Rust: lanes in step, branches splitting them" $ withLowered $ \c l ->
        dropWhile (not . T.isPrefixOf "/// `sum_rows`") (T.lines l.loweredSource)
          @?= [ "/// `sum_rows`: IR hash " <> hex64 (irHash c) <> ", ABI hash " <> hex64 (Abi.abiHash (abiOf (kernelIR sumRows))) <> "."
              , "pub fn sum_rows("
              , "    fk_p: &[u8],"
              , "    x: &[f32],"
              , "    y: &mut [f32],"
              , "    rows: &[&[f32]],"
              , ") {"
              , "    let n: u32 = u32::from_ne_bytes(fk_bytes(fk_p, 0));"
              , "    assert!(n >= 1 && n <= 16, \"sum_rows: the dimension n is {n}, outside its range 1..16.\");"
              , "    const FK_THREADS: usize = 32;"
              , "    const FK_WIDTH: usize = 32;"
              , "    let fk_groups = (u64::from(n)) as usize;"
              , "    let mut acc: Vec<f32> = vec![0.0; FK_THREADS];"
              , "    let mut i: Vec<u32> = vec![0u32; FK_THREADS];"
              , "    let mut total: Vec<f32> = vec![0.0; FK_THREADS];"
              , "    let fk_all: Vec<usize> = (0..FK_THREADS).collect();"
              , "    for fk_group in 0..fk_groups {"
              , "        let fk_group = fk_group as u32;"
              , "        let fk_act: &[usize] = &fk_all;"
              , "        for &fk_l in fk_act {"
              , "            acc[fk_l] = f32::from_bits(0x0000_0000);"
              , "        }"
              , "        {"
              , "            for &fk_l in fk_act {"
              , "                i[fk_l] = (fk_l as u32);"
              , "            }"
              , "            let mut fk_live: Vec<usize> = fk_act.to_vec();"
              , "            loop {"
              , "                fk_live.retain(|&fk_l| i[fk_l] < 64u32);"
              , "                if fk_live.is_empty() {"
              , "                    break;"
              , "                }"
              , "                {"
              , "                    let fk_act: &[usize] = &fk_live;"
              , "                    for &fk_l in fk_act {"
              , "                        acc[fk_l] = (acc[fk_l] + (x[(fk_group.wrapping_mul(64u32).wrapping_add(i[fk_l])) as usize] * rows[(fk_group) as usize][(i[fk_l]) as usize]));"
              , "                    }"
              , "                }"
              , "                for &fk_l in &fk_live {"
              , "                    i[fk_l] = i[fk_l].wrapping_add(32u32);"
              , "                }"
              , "            }"
              , "        }"
              , "        {"
              , "            let mut fk_i = 0;"
              , "            while fk_i < fk_act.len() {"
              , "                let fk_simd = fk_act[fk_i] / FK_WIDTH;"
              , "                let mut fk_j = fk_i;"
              , "                let mut fk_v: [f32; FK_WIDTH] = [Default::default(); FK_WIDTH];"
              , "                while fk_j < fk_act.len() && fk_act[fk_j] / FK_WIDTH == fk_simd {"
              , "                    let fk_l = fk_act[fk_j];"
              , "                    fk_v[fk_j - fk_i] = acc[fk_l];"
              , "                    fk_j += 1;"
              , "                }"
              , "                // Pairwise, neighbours first, as the IR orders a reduction."
              , "                let mut fk_n = fk_j - fk_i;"
              , "                while fk_n > 1 {"
              , "                    for fk_k in 0..fk_n / 2 {"
              , "                        fk_v[fk_k] = (fk_v[2 * fk_k] + fk_v[2 * fk_k + 1]);"
              , "                    }"
              , "                    if fk_n % 2 == 1 {"
              , "                        fk_v[fk_n / 2] = fk_v[fk_n - 1];"
              , "                    }"
              , "                    fk_n = fk_n.div_ceil(2);"
              , "                }"
              , "                for &fk_l in &fk_act[fk_i..fk_j] {"
              , "                    total[fk_l] = fk_v[0];"
              , "                }"
              , "                fk_i = fk_j;"
              , "            }"
              , "        }"
              , "        {"
              , "            let fk_then: Vec<usize> = fk_act.iter().copied().filter(|&fk_l| ((fk_l as u32) == 0u32)).collect();"
              , "            if !fk_then.is_empty() {"
              , "                let fk_act: &[usize] = &fk_then;"
              , "                for &fk_l in fk_act {"
              , "                    let fk_value = total[fk_l];"
              , "                    y[(fk_group) as usize] = fk_value;"
              , "                }"
              , "            }"
              , "        }"
              , "    }"
              , "}"
              ]
    , testCase "its launch table: buffers, then tables, the parameters first" $ withLowered $ \_ l -> case l.loweredKernels of
        [t] -> do
          l.loweredLanguage @?= rustLanguage
          l.loweredOptions @?= compileOptions
          (t.lkName, t.lkEntry, t.lkThreads) @?= ("sum_rows", "sum_rows", 32)
          t.lkBindings @?= [("x", 0), ("y", 1), ("rows", 2)]
          t.lkParamsBinding @?= Nothing
        ts -> assertFailure ("tables: " <> show (length ts))
    , testCase "its launch table and the manifest: threadgroup memory as the kernel's layout" $ case check padded of
        Left fs -> assertFailure (show fs)
        Right c -> do
          either (assertFailure . show) (\l -> map (.lkTgMemory) l.loweredKernels @?= [16416]) (backendLower cpu [c])
          assertBool "threadgroup_memory" ("\"threadgroup_memory\": 16416" `T.isInfixOf` renderJson (kernelManifest c))
    , testCase "dimensions held to their ranges and multiples before a launch size is computed: a message naming the kernel, the dimension, its value and its range" $
        case check ranges >>= backendLower cpu . pure of
          Left fs -> assertFailure (show fs)
          Right l ->
            [T.strip t | t <- T.lines l.loweredSource, "assert!(" `T.isInfixOf` t]
              @?= [ "assert!(m >= 8 && m <= 64 && m % 8 == 0, \"ranges: the dimension m is {m}, outside its range 8..64 multiple of 8.\");"
                  , "assert!(r#loop >= 1 && r#loop <= 100, \"ranges: the dimension loop is {}, outside its range 1..100.\", r#loop);"
                  ]
    , testCase "fma: mul_add in f32, through binary64 in f16; exp in binary64, rounded once" $
        case check fused >>= backendLower cpu . pure of
          Left fs -> assertFailure (show fs)
          Right l ->
            [T.strip t | t <- T.lines l.loweredSource, any (`T.isInfixOf` t) ["mul_add(", "fk_fma_f16(", ".exp()"], not ("fn " `T.isInfixOf` t)]
              @?= [ "let fk_value = f32::mul_add(x[(0u32) as usize], x[(1u32) as usize], x[(2u32) as usize]);"
                  , "let fk_value = (f64::from(x[(0u32) as usize]).exp() as f32);"
                  , "let fk_value = fk_fma_f16(h[(0u32) as usize], h[(1u32) as usize], h[(2u32) as usize]);"
                  , "let fk_value = fk_narrow_f16_f64(f64::from(fk_widen_f16(h[(0u32) as usize])).exp());"
                  ]
    , testCase "a module refuses two kernels of one name, and two that make one Rust type name" $
        case traverse (check . named) ["sum_rows", "a_b", "aB"] of
          Right [s, ab, aB] -> do
            void (backendLower cpu [s, s]) @?= Left [Finding Names "The kernel sum_rows is lowered 2 times into one module."]
            void (backendLower cpu [ab, aB]) @?= Left [Finding Names "The kernels a_b, aB make one Rust type name, AB."]
          other -> assertFailure (show (fmap length other))
    ]
 where
  withLowered f = case check sumRows of
    Left fs -> assertFailure (show fs)
    Right c -> either (assertFailure . show) (f c) (backendLower cpu [c])

{- | Dimensions of a multiple, of a Rust keyword's name, and of every u32,
which needs no check.
-}
ranges :: Kernel Cpu
ranges = kernel "ranges" $ do
  m <- dim "m" (DimRange 8 64 8)
  l <- dim "loop" (DimRange 1 100 1)
  n <- dim "n" (DimRange 0 4294967295 1)
  y <- buffer @WriteOnly @16 @Word32 "y" 3
  threads @32
  groups 1
  pure . pure $ whenLane (threadIndex .== 0) $ do
    store y 0 m
    store y 1 l
    store y 2 n

-- | An fma and an exp in each float type, by thread 0.
fused :: Kernel Cpu
fused = kernel "fused" $ do
  x <- buffer @ReadOnly @16 @Float "x" 3
  h <- buffer @ReadOnly @16 @F16 "h" 3
  y <- buffer @WriteOnly @16 @Float "y" 2
  g <- buffer @WriteOnly @16 @F16 "g" 2
  threads @32
  groups 1
  pure . pure $ whenLane (threadIndex .== 0) $ do
    store y 0 (fmaE (load x 0) (load x 1) (load x 2))
    store y 1 (expE (load x 0))
    store g 0 (fmaE (load h 0) (load h 1) (load h 2))
    store g 1 (expE (load h 0))

-- | A kernel of this name that does nothing.
named :: Name -> Kernel Cpu
named name = kernel name $ do
  threads @32
  groups 1
  pure (pure (pure ()))

-- | Threadgroup arrays of 4093 and 5 f32, 16372 B and 20 B: 16384 B and 32 B with their alignment.
padded :: Kernel Cpu
padded = kernel "padded" $ do
  threads @32
  groups 1
  pure $ do
    _ <- tgArray @4093 @Float "a"
    _ <- tgArray @5 @Float "b"
    pure (pure ())

{- | Each threadgroup takes a row: the dot product of its row of @x@ and the
array an entry of the table points at, summed by the SIMD group.
-}
sumRows :: Kernel Cpu
sumRows = kernel "sum_rows" $ do
  n <- dim "n" (DimRange 1 16 1)
  x <- buffer @ReadOnly @16 @Float "x" (n * 64)
  y <- buffer @WriteOnly @16 @Float "y" n
  rows <- table @16 @Float "rows" n 64
  threads @32
  groups n
  pure . pure $ do
    acc <- var "acc" 0
    strided "i" 64 $ \i -> acc .= get acc + load x (groupIndex * 64 + i) * loadVia rows groupIndex i
    total <- simdSum "total" (get acc)
    whenLane (threadIndex .== 0) $ store y groupIndex total
