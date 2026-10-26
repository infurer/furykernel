-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

-- | The typed surface builds the IR it should, and notices its own problems.
module Test.FuryKernel.BuildTest (tests) where

import Control.Exception qualified as E
import Data.Int (Int32)
import Data.Text qualified as T
import Data.Word (Word32)
import FuryKernel.Build
import FuryKernel.Finding (Finding (..), Rule (..))
import FuryKernel.IR (DimRange (..), Expr (..), Literal (..), Overflow (..), renderExpr, renderKernel)
import FuryKernel.Type (F16, ScalarType (..))
import Test.FuryKernel.Kernels (scale, scalePacked)
import Test.FuryKernel.Targets (T32)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (testCase, (@?=))

tests :: TestTree
tests =
  testGroup
    "Build"
    [ testCase "a kernel renders as its IR" $
        renderKernel (kernelIR (scale @T32))
          @?= T.unlines
            [ "ir 1"
            , "kernel scale threads 64 simd 32 groups 1u"
            , "  param n : u32 dim 1..65536 multiple of 1"
            , "  param s : f32"
            , "  buffer x : read bf16[$n] align 16"
            , "  buffer y : write bf16[$n] align 16"
            , "{"
            , "  for divergent i = @thread until $n step 64u {"
            , "    let v : f32 = f32(x[i])"
            , "    y[i] = bf16((v * $s))"
            , "  }"
            , "}"
            ]
    , testCase "packs render with their width" $
        renderKernel (kernelIR (scalePacked @T32))
          @?= T.unlines
            [ "ir 1"
            , "kernel scale_packed threads 64 simd 32 groups 1u"
            , "  param packs : u32 dim 1..8192 multiple of 1"
            , "  param s : f32"
            , "  buffer x : read bf16[($packs * 8u)] align 16"
            , "  buffer y : write bf16[($packs * 8u)] align 16"
            , "{"
            , "  for divergent j = @thread until $packs step 64u {"
            , "    let (v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7) : bf16 = x.pack8[j]"
            , "    y.pack8[j] = ("
                <> T.intercalate ", " ["bf16((f32(v_" <> T.pack (show i) <> ") * $s))" | i <- [0 .. 7 :: Int]]
                <> ")"
            , "  }"
            , "}"
            ]
    , testCase "an if, a SIMD-group sum and threadgroup memory render as their IR" $
        renderKernel (kernelIR controlled)
          @?= T.unlines
            [ "ir 1"
            , "kernel controlled threads 64 simd 32 groups 1u"
            , "  buffer y : write f32[2u] align 16"
            , "  threadgroup partial : f32[2]"
            , "{"
            , "  let s : f32 = simd_sum(f32:0x3f800000)"
            , "  if divergent (@lane == 0u) {"
            , "    partial[@simd_group] = s"
            , "  }"
            , "  barrier"
            , "  if divergent (@thread < 2u) {"
            , "    y[@thread] = partial[@thread]"
            , "  }"
            , "}"
            ]
    , testCase "a NaN literal is the canonical NaN, a binary16 literal its value rounded once" $
        renderKernel (kernelIR literals)
          @?= T.unlines
            [ "ir 1"
            , "kernel literals threads 32 simd 32 groups 1u"
            , "{"
            , "  let nan : f32 = f32:0x7fc00000"
            , "  let h : f16 = f16:0x3c01"
            , "}"
            ]
    , testCase "a pack stored with the wrong number of values" $
        kernelProblems short @?= [Finding Typing "A pack of 4 stored to y with 3 values."]
    , testCase "an integer narrowing renders with what it does out of range" $
        map (\o -> renderExpr (ENarrowInt o TI8 (EVar "x"))) [Clamp, Wrap] @?= ["i8_clamped(x)", "i8_wrapped(x)"]
    , testCase "an fma renders as its IR, its operands in order" $
        renderExpr (expIR (fmaE (1 :: Exp Float) 2 3)) @?= "fma(f32:0x3f800000, f32:0x40000000, f32:0x40400000)"
    , testCase "an i32 or u32 literal outside its type's range is an error, not a wrapped value" $ do
        let failure k = either (\(E.ErrorCall m) -> m) (const "") <$> E.try (E.evaluate (T.length (renderKernel (kernelIR k))))
        failure (outOfRange 3000000000 0) >>= (@?= "FuryKernel.Build: the literal 3000000000 is outside i32's range, -2147483648 to 2147483647.")
        failure (outOfRange 0 4294967296) >>= (@?= "FuryKernel.Build: the literal 4294967296 is outside u32's range, 0 to 4294967295.")
        failure (outOfRange (-2147483647) 4294967295) >>= (@?= "")
    , testCase "literals compare by their bits, as their text does: a NaN equals itself, -0 differs from +0" $ do
        let nan = ELit (LF32 (0 / 0))
            pairs = [(nan, nan), (ELit (LF32 (-0)), ELit (LF32 0)), (ELit (LF16 0x7e00), ELit (LF16 0x7e00)), (ELit (LI32 1), ELit (LU32 1))]
        map (uncurry (==)) pairs @?= [True, False, True, False]
        map (\(x, y) -> renderExpr x == renderExpr y) pairs @?= [True, False, True, False]
    , testCase "locals get fresh names, interface names stay" $
        renderKernel (kernelIR twice)
          @?= T.unlines
            [ "ir 1"
            , "kernel twice threads 32 simd 32 groups 1u"
            , "  param v : f32"
            , "{"
            , "  let v_1 : f32 = $v"
            , "  let v_2 : f32 = (v_1 + v_1)"
            , "}"
            ]
    , testCase "the builder reports what it notices" $ do
        kernelProblems (scale @T32) @?= []
        kernelProblems twice @?= []
        kernelProblems clash
          @?= [ Finding Names "The name x is declared twice."
              , Finding Threads "No thread count: declare one with threads."
              , Finding Threads "No grid size: declare one with groups."
              ]
    ]

-- | An i32 and a u32, stored.
outOfRange :: Exp Int32 -> Exp Word32 -> Kernel T32
outOfRange i u = kernel "out_of_range" $ do
  y <- buffer @WriteOnly @16 @Int32 "y" 1
  z <- buffer @WriteOnly @16 @Word32 "z" 1
  threads @32
  groups 1
  pure . pure $ whenLane (threadIndex .== 0) $ do
    store y 0 i
    store z 0 u

twice :: Kernel T32
twice = kernel "twice" $ do
  v <- param @Float "v"
  threads @32
  groups 1
  pure . pure $ do
    a <- let_ "v" v
    _ <- let_ "v" (a + a)
    pure ()

-- | An if, a SIMD-group sum, and a threadgroup array with its barrier.
controlled :: Kernel T32
controlled = kernel "controlled" $ do
  y <- buffer @WriteOnly @16 @Float "y" 2
  threads @64
  groups 1
  pure $ do
    partial <- tgArray @2 @Float "partial"
    pure $ do
      s <- simdSum "s" 1
      whenLane (simdLane .== 0) $ tgStore partial simdGroupIndex s
      barrier
      whenLane (threadIndex .< 2) $ store y threadIndex (tgLoad partial threadIndex)

{- | A NaN whatever the host makes of 0 / 0, and 1 + 2^-11 + 2^-30, which
rounding through binary32 would make 1.
-}
literals :: Kernel T32
literals = kernel "literals" $ do
  threads @32
  groups 1
  pure . pure $ do
    _ <- let_ "nan" (f32 (0 / 0))
    _ <- let_ "h" (fromRational (1 + 1 / 2048 + 1 / 1073741824) :: Exp F16)
    pure ()

clash :: Kernel T32
clash = kernel "clash" $ do
  n <- dim "x" (DimRange 1 8 1)
  _ <- buffer @ReadOnly @16 @Word32 "x" n
  pure (pure (pure ()))

short :: Kernel T32
short = kernel "short" $ do
  y <- buffer @WriteOnly @16 @Float "y" 4
  threads @32
  groups 1
  pure . pure $ storePack @4 y 0 [1, 2, 3]
