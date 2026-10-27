-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE GHC2024 #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}

{- | checks.cpu's cases: kernels on the CPU target and inputs for them. For
each, the CPU target's Rust goes to @src/gen@, the interpreter's inputs and
outputs to @data@, and the code that runs the one on the other and the
pieces of "FuryKernel.Emit.Rust" that runtimes generate to @src/cases.rs@
and @src/launch.rs@, which the crate in this directory compares bit for
bit.

Where an output of arithmetic is a NaN, it is compared by class only
(quiet or signalling): the IR leaves a NaN's payload unspecified. Every
other output is compared bit for bit, the negations, magnitudes and
conversions of NaNs included.
-}
module Main (main) where

import Control.Monad (forM, forM_, void)
import Data.Bits (shiftR, xor, (.&.))
import Data.ByteString qualified as BS
import Data.Int (Int32, Int8)
import Data.List (nub)
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import Data.Text qualified as T
import Data.Text.IO qualified as T
import Data.Word (Word16, Word32, Word64, Word8)
import FuryKernel
import FuryKernel.Abi (Abi (..), Field (..), abiOf)
import FuryKernel.Backend (Backend (..), Lowered (..))
import FuryKernel.Check (checkedIR)
import FuryKernel.Cpu (Cpu, cpu)
import FuryKernel.Emit.Rust (dimRangeText, rustIdent, rustLaunchExpr, rustParamsStruct, rustScalarType)
import FuryKernel.IR (Expr (..), Literal (..))
import FuryKernel.IR qualified as IR
import FuryKernel.Interp (RunError (..), Value (..), launch, run, withSeed, withTables)
import FuryKernel.Numerics (narrowBF16, narrowF16, widenF16)
import GHC.Float (castFloatToWord32, castWord32ToFloat)
import GHC.TypeNats (KnownNat)
import System.Directory (createDirectoryIfMissing)
import System.Environment (getArgs)
import Test.FuryKernel.Gemv (GemvShape (..), GemvVariant (..), gemv)
import Test.FuryKernel.Kernels (rmsnorm, scale, scalePacked)

------------------------------------------------------------------------------
-- Cases

-- | A kernel, the launches to compare it on, and the outputs compared by NaN class only.
data Case = Case
  { caseId :: Text
  , caseKernel :: Checked Cpu
  , caseRuns :: [Inputs]
  , caseByClass :: [Name]
  }

data Inputs = Inputs
  { inParams :: [(Name, Value)]
  , inBuffers :: [(Name, [Value])]
  , inTables :: [(Name, [[Value]])]
  }

{- | A case's kernel launched where the interpreter refuses it, a dimension
outside its range or off its multiple: the CPU target panics with the
message that names the kernel, the dimension, its value and its range.
-}
data Refusal = Refusal
  { refusalCase :: Case
  , refusalRuns :: [Inputs]
  }

checked :: Text -> Kernel Cpu -> Checked Cpu
checked what k = either (\fs -> error (T.unpack what <> ": " <> show fs)) id (check k)

cases :: [Case]
cases =
  [ Case "sum_rows" (checked "sum_rows" sumRows) [Inputs [("n", u 3)] [("x", f32s 3 192), ("y", zeros 3)] [("rows", [f32s (10 + i) 64 | i <- [0 .. 2]])]] []
  , Case "rmsnorm" (checked "rmsnorm" (rmsnorm @Cpu 1000)) [Inputs [("rows", u 3), ("eps", VF32 1e-6)] [("x", bf16s 1 3000), ("w", bf16s 2 1000), ("y", bf16Zeros 3000)] []] []
  , Case "scale" (checked "scale" (scale @Cpu)) [Inputs [("n", u 300), ("s", VF32 0.7)] [("x", bf16s 4 300), ("y", bf16Zeros 300)] []] []
  , Case "scale_packed" (checked "scale_packed" (scalePacked @Cpu)) [Inputs [("packs", u 37), ("s", VF32 (-1.3))] [("x", bf16s 5 296), ("y", bf16Zeros 296)] []] []
  , simdSumsCase
  , narrowingsCase
  ]
    <> zipWith gemvCase [0 :: Int ..] gemvs
    <> [ gemmCase "gemm_bf16" (checked "gemm_bf16" (gemm @BF16)) (bf16s 30) (bf16s 31)
       , gemmCase "gemm_f16" (checked "gemm_f16" (gemm @F16)) (f16s 30) (f16s 31)
       , gemmCase "gemm_f32" (checked "gemm_f32" (gemm @Float)) (f32s 30) (f32s 31)
       , gemmCase "staged" (checked "staged" staged) (bf16s 32) (bf16s 33)
       , tileCase "tiled" (checked "tiled" (tiled @1024 @1024 @4096 (GemmTile 64 64 16 4))) (70, 72, 48)
       , tileCase "tiled_narrow" (checked "tiled_narrow" (tiled @1024 @2048 @512 (GemmTile 32 64 32 1))) (70, 72, 64)
       , arithF32Case
       , arithF16Case
       , fmaF32Case
       , fmaF16Case
       , arithIntCase
       , convCase
       , Case "flow" (checked "flow" flow) [Inputs [("n", u 50)] [("x", f32s 60 50), ("cnt", [VU32 (fromIntegral (w `mod` 13)) | w <- randoms 61 288]), ("y", zeros 288), ("z", zeros 288), ("w", replicate 288 (VU32 0))] []] []
       , reductionsCase
       , paramsCase
       , rangedCase
       ]
 where
  gemvs =
    [ (GemvShape 64 256 1, GemvVariant 1 32 4 1)
    , (GemvShape 64 256 3, GemvVariant 2 64 8 2)
    , (GemvShape 128 512 2, GemvVariant 4 128 8 4)
    , (GemvShape 64 512 3, GemvVariant 1 256 8 8)
    , (GemvShape 32 1024 2, GemvVariant 8 128 8 4)
    ]
  gemvCase i (sh, v) =
    let name = "gemv_" <> T.pack (show i)
        k = either (error . show) id (gemv @Cpu name sh v)
        outs = fromIntegral sh.outputs
        ins = fromIntegral sh.inputs
        m = fromIntegral sh.batch
     in Case name (checked name k) [Inputs [] [("x", bf16s 20 (m * ins)), ("w", bf16s 21 (outs * ins)), ("y", bf16Zeros (m * outs))] []] []
  gemmCase name k a b = Case name k [Inputs [("k", u 64)] [("a", a (16 * 64)), ("b", b (16 * 64)), ("c", zeros 256)] []] []
  tileCase name k (m, n, kk) = Case name k [Inputs [("m", u m), ("n", u n), ("k", u kk)] [("a", bf16s 40 (m * kk)), ("b", bf16s 41 (n * kk)), ("c", zeros (m * n))] []] []

------------------------------------------------------------------------------
-- Kernels

-- | Each threadgroup's row of x against the array of the row's table entry, summed by the SIMD group.
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

-- | C = A · Bᵀ in 8×8 fragments, A and B 16 rows of k, B's tiles loaded transposed.
gemm :: forall a. (IsScalar a, AlignOk a 16, FragInput a) => Kernel Cpu
gemm = kernel "gemm" $ do
  k <- dim "k" (DimRange 8 256 8)
  a <- buffer @ReadOnly @16 @a "a" (16 * k)
  b <- buffer @ReadOnly @16 @a "b" (16 * k)
  c <- buffer @WriteOnly @16 @Float "c" 256
  threads @128
  groups 1
  pure . pure $ do
    row0 <- let_ "row0" ((simdGroupIndex `quotE` 2) * 8)
    col0 <- let_ "col0" ((simdGroupIndex `remE` 2) * 8)
    acc <- fragment @Float "acc"
    fa <- fragment @a "fa"
    fb <- fragment @a "fb"
    forU "p" 0 (k `quotE` 8) $ \p -> do
      fragLoad fa a (row0 * k + p * 8) k
      fragLoadT fb b (col0 * k + p * 8) k
      fragMma acc fa fb acc
    fragStore c (row0 * 16 + col0) 16 acc

-- | 'gemm' in bf16 with B's tiles staged in threadgroup memory and loaded from there transposed.
staged :: Kernel Cpu
staged = kernel "staged" $ do
  k <- dim "k" (DimRange 8 256 8)
  a <- buffer @ReadOnly @16 @BF16 "a" (16 * k)
  b <- buffer @ReadOnly @16 @BF16 "b" (16 * k)
  c <- buffer @WriteOnly @16 @Float "c" 256
  threads @128
  groups 1
  pure $ do
    tile <- tgArray @128 @BF16 "tile"
    pure $ do
      row0 <- let_ "row0" ((simdGroupIndex `quotE` 2) * 8)
      col0 <- let_ "col0" ((simdGroupIndex `remE` 2) * 8)
      acc <- fragment @Float "acc"
      fa <- fragment @BF16 "fa"
      fb <- fragment @BF16 "fb"
      forU "p" 0 (k `quotE` 8) $ \p -> do
        tgStore tile threadIndex (load b ((threadIndex `quotE` 8) * k + p * 8 + threadIndex `remE` 8))
        barrier
        fragLoad fa a (row0 * k + p * 8) k
        fragLoadTgT fb tile (col0 * 8) 8
        fragMma acc fa fb acc
        barrier
      fragStore c (row0 * 16 + col0) 16 acc

-- | C = A · Bᵀ of bf16 rows through a GEMM tile, edges clamped.
tiled :: forall na nb nc. (TgArrayOk Cpu na Float, TgArrayOk Cpu nb Float, TgArrayOk Cpu nc Float, KnownNat na, KnownNat nb, KnownNat nc) => GemmTile -> Kernel Cpu
tiled t = kernel "tiled" $ do
  m <- dim "m" (DimRange 1 256 1)
  n <- dim "n" (DimRange 8 256 8)
  k <- dim "k" (DimRange t.gtStep 256 t.gtStep)
  a <- buffer @ReadOnly @16 @BF16 "a" (m * k)
  b <- buffer @ReadOnly @16 @BF16 "b" (n * k)
  c <- buffer @WriteOnly @16 @Float "c" (m * n)
  threads @256
  let rows = fromIntegral t.gtRows
      cols = fromIntegral t.gtCols
  groups (((m + rows - 1) `quotE` rows) * ((n + cols - 1) `quotE` cols))
  pure $ do
    aSlice <- tgArray @na @Float "a_slice"
    bSlice <- tgArray @nb @Float "b_slice"
    cTile <- tgArray @nc @Float "c_tile"
    pure $ do
      tilesN <- let_ "tiles_n" ((n + cols - 1) `quotE` cols)
      row0 <- let_ "row0" ((groupIndex `quotE` tilesN) * rows)
      col0 <- let_ "col0" ((groupIndex `remE` tilesN) * cols)
      gemmTile
        "t"
        t
        aSlice
        bSlice
        cTile
        k
        (\r p -> pure [widen @Float (load a (minE (row0 + r) (m - 1) * k + p + fromIntegral x)) | x <- [0 .. 3 :: Int]])
        (\col p -> pure [widen @Float (load b (minE (col0 + col) (n - 1) * k + p + fromIntegral x)) | x <- [0 .. 7 :: Int]])
        (\r col v -> whenLane (row0 + r .< m .&& col0 + col .< n) (store c ((row0 + r) * n + col0 + col) v))

-- | Every f32 operation on pairs of elements.
arithF32 :: Kernel Cpu
arithF32 = kernel "arith_f32" $ do
  n <- dim "n" (DimRange 1 100000 1)
  x <- buffer @ReadOnly @16 @Float "x" n
  y <- buffer @ReadOnly @16 @Float "y" n
  oAdd <- buffer @WriteOnly @16 @Float "o_add" n
  oSub <- buffer @WriteOnly @16 @Float "o_sub" n
  oMul <- buffer @WriteOnly @16 @Float "o_mul" n
  oDiv <- buffer @WriteOnly @16 @Float "o_div" n
  oMin <- buffer @WriteOnly @16 @Float "o_min" n
  oMax <- buffer @WriteOnly @16 @Float "o_max" n
  oSqrt <- buffer @WriteOnly @16 @Float "o_sqrt" n
  oRsqrt <- buffer @WriteOnly @16 @Float "o_rsqrt" n
  oExp <- buffer @WriteOnly @16 @Float "o_exp" n
  oLog <- buffer @WriteOnly @16 @Float "o_log" n
  oTanh <- buffer @WriteOnly @16 @Float "o_tanh" n
  oAbs <- buffer @WriteOnly @16 @Float "o_abs" n
  oNeg <- buffer @WriteOnly @16 @Float "o_neg" n
  oCmp <- buffer @WriteOnly @16 @Word32 "o_cmp" n
  oBf <- buffer @WriteOnly @16 @BF16 "o_bf" n
  oH <- buffer @WriteOnly @16 @F16 "o_h" n
  oSel <- buffer @WriteOnly @16 @Float "o_sel" n
  threads @64
  groups 1
  pure . pure $
    strided "i" n $ \i -> do
      a <- let_ "a" (load x i)
      b <- let_ "b" (load y i)
      store oAdd i (a + b)
      store oSub i (a - b)
      store oMul i (a * b)
      store oDiv i (a / b)
      store oMin i (minE a b)
      store oMax i (maxE a b)
      store oSqrt i (sqrtE a)
      store oRsqrt i (rsqrtE a)
      store oExp i (expE a)
      store oLog i (logE a)
      store oTanh i (tanhE a)
      store oAbs i (abs a)
      store oNeg i (negate a)
      store oCmp i (select (a .< b) 1 0 + select (a .<= b) 2 0 + select (a .> b) 4 0 + select (a .>= b) 8 0 + select (a .== b) 16 0 + select (a ./= b) 32 0)
      store oBf i (narrow @BF16 a)
      store oH i (narrow @F16 a)
      store oSel i (select (a .< b) a b)

-- | Every f16 operation on pairs of elements.
arithF16 :: Kernel Cpu
arithF16 = kernel "arith_f16" $ do
  n <- dim "n" (DimRange 1 100000 1)
  x <- buffer @ReadOnly @16 @F16 "x" n
  y <- buffer @ReadOnly @16 @F16 "y" n
  oAdd <- buffer @WriteOnly @16 @F16 "o_add" n
  oSub <- buffer @WriteOnly @16 @F16 "o_sub" n
  oMul <- buffer @WriteOnly @16 @F16 "o_mul" n
  oDiv <- buffer @WriteOnly @16 @F16 "o_div" n
  oMin <- buffer @WriteOnly @16 @F16 "o_min" n
  oMax <- buffer @WriteOnly @16 @F16 "o_max" n
  oSqrt <- buffer @WriteOnly @16 @F16 "o_sqrt" n
  oRsqrt <- buffer @WriteOnly @16 @F16 "o_rsqrt" n
  oExp <- buffer @WriteOnly @16 @F16 "o_exp" n
  oLog <- buffer @WriteOnly @16 @F16 "o_log" n
  oTanh <- buffer @WriteOnly @16 @F16 "o_tanh" n
  oAbs <- buffer @WriteOnly @16 @F16 "o_abs" n
  oNeg <- buffer @WriteOnly @16 @F16 "o_neg" n
  oCmp <- buffer @WriteOnly @16 @Word32 "o_cmp" n
  oWide <- buffer @WriteOnly @16 @Float "o_wide" n
  oLit <- buffer @WriteOnly @16 @F16 "o_lit" n
  threads @64
  groups 1
  pure . pure $
    strided "i" n $ \i -> do
      a <- let_ "a" (load x i)
      b <- let_ "b" (load y i)
      store oAdd i (a + b)
      store oSub i (a - b)
      store oMul i (a * b)
      store oDiv i (a / b)
      store oMin i (minE a b)
      store oMax i (maxE a b)
      store oSqrt i (sqrtE a)
      store oRsqrt i (rsqrtE a)
      store oExp i (expE a)
      store oLog i (logE a)
      store oTanh i (tanhE a)
      store oAbs i (abs a)
      store oNeg i (negate a)
      store oCmp i (select (a .< b) 1 0 + select (a .<= b) 2 0 + select (a .> b) 4 0 + select (a .>= b) 8 0 + select (a .== b) 16 0 + select (a ./= b) 32 0)
      store oWide i (widen @Float a)
      store oLit i (a * 0.1 + (-3))

-- | Every i32 and u32 operation on pairs of elements, divisors that would be errors replaced.
arithInt :: Kernel Cpu
arithInt = kernel "arith_int" $ do
  n <- dim "n" (DimRange 1 100000 1)
  xi <- buffer @ReadOnly @16 @Int32 "xi" n
  yi <- buffer @ReadOnly @16 @Int32 "yi" n
  xu <- buffer @ReadOnly @16 @Word32 "xu" n
  yu <- buffer @ReadOnly @16 @Word32 "yu" n
  iAdd <- buffer @WriteOnly @16 @Int32 "i_add" n
  iSub <- buffer @WriteOnly @16 @Int32 "i_sub" n
  iMul <- buffer @WriteOnly @16 @Int32 "i_mul" n
  iNeg <- buffer @WriteOnly @16 @Int32 "i_neg" n
  iAbs <- buffer @WriteOnly @16 @Int32 "i_abs" n
  iMin <- buffer @WriteOnly @16 @Int32 "i_min" n
  iMax <- buffer @WriteOnly @16 @Int32 "i_max" n
  iDiv <- buffer @WriteOnly @16 @Int32 "i_div" n
  iRem <- buffer @WriteOnly @16 @Int32 "i_rem" n
  iCmp <- buffer @WriteOnly @16 @Word32 "i_cmp" n
  uAdd <- buffer @WriteOnly @16 @Word32 "u_add" n
  uSub <- buffer @WriteOnly @16 @Word32 "u_sub" n
  uMul <- buffer @WriteOnly @16 @Word32 "u_mul" n
  uNeg <- buffer @WriteOnly @16 @Word32 "u_neg" n
  uDiv <- buffer @WriteOnly @16 @Word32 "u_div" n
  uRem <- buffer @WriteOnly @16 @Word32 "u_rem" n
  uMin <- buffer @WriteOnly @16 @Word32 "u_min" n
  uCmp <- buffer @WriteOnly @16 @Word32 "u_cmp" n
  uTof <- buffer @WriteOnly @16 @Float "u_tof" n
  iLit <- buffer @WriteOnly @16 @Int32 "i_lit" n
  threads @64
  groups 1
  pure . pure $
    strided "i" n $ \i -> do
      a <- let_ "a" (load xi i)
      b <- let_ "b" (load yi i)
      p <- let_ "p" (load xu i)
      q <- let_ "q" (load yu i)
      store iAdd i (a + b)
      store iSub i (a - b)
      store iMul i (a * b)
      store iNeg i (negate a)
      store iAbs i (abs a)
      store iMin i (minE a b)
      store iMax i (maxE a b)
      d <- let_ "d" (select (b .== 0 .|| (b .== (-1) .&& a .== (-2147483647 - 1))) 7 b)
      store iDiv i (a `quotE` d)
      store iRem i (a `remE` d)
      store iCmp i (select (a .< b) 1 0 + select (a .<= b) 2 0 + select (a .> b) 4 0 + select (a .>= b) 8 0 + select (a .== b) 16 0 + select (a ./= b) 32 0)
      store uAdd i (p + q)
      store uSub i (p - q)
      store uMul i (p * q)
      store uNeg i (negate p)
      e <- let_ "e" (select (q .== 0) 3 q)
      store uDiv i (p `quotE` e)
      store uRem i (p `remE` e)
      store uMin i (minE p q + maxE p q * 3)
      store uCmp i (select (p .< q) 1 0 + select (p .<= q) 2 0 + select (p .> q) 4 0 + select (p .>= q) 8 0 + select (p .== q) 16 0 + select (p ./= q) 32 0)
      store uTof i (toF32 p)
      negated <- let_ "negated" (negate (-2147483647))
      store iLit i ((a * (-2147483647 - 1)) + (-1) - abs (-5) + negated)

-- | Every conversion.
conv :: Kernel Cpu
conv = kernel "conv" $ do
  n <- dim "n" (DimRange 1 100000 1)
  xb <- buffer @ReadOnly @16 @BF16 "xb" n
  xh <- buffer @ReadOnly @16 @F16 "xh" n
  xs <- buffer @ReadOnly @16 @Int8 "xs" n
  xc <- buffer @ReadOnly @16 @Word8 "xc" n
  oB <- buffer @WriteOnly @16 @Float "o_b" n
  oH <- buffer @WriteOnly @16 @Float "o_h" n
  oS <- buffer @WriteOnly @16 @Int32 "o_s" n
  oC <- buffer @WriteOnly @16 @Word32 "o_c" n
  oRb <- buffer @WriteOnly @16 @BF16 "o_rb" n
  oRh <- buffer @WriteOnly @16 @F16 "o_rh" n
  oPb <- buffer @WriteOnly @16 @BF16 "o_pb" n
  threads @64
  groups 1
  pure . pure $
    strided "i" n $ \i -> do
      store oB i (widen @Float (load xb i))
      store oH i (widen @Float (load xh i))
      store oS i (widen @Int32 (load xs i))
      store oC i (widen @Word32 (load xc i))
      store oRb i (narrow @BF16 (widen @Float (load xb i)))
      store oRh i (narrow @F16 (widen @Float (load xh i)))
      store oPb i (load xb i)

-- | Divergent loops, SIMD-uniform branches, reductions, threadgroup memory.
flow :: Kernel Cpu
flow = kernel "flow" $ do
  n <- dim "n" (DimRange 1 4096 1)
  x <- buffer @ReadOnly @16 @Float "x" n
  cnt <- buffer @ReadOnly @16 @Word32 "cnt" 288
  y <- buffer @WriteOnly @16 @Float "y" 288
  z <- buffer @WriteOnly @16 @Float "z" 288
  w <- buffer @WriteOnly @16 @Word32 "w" 288
  threads @96
  groups 3
  pure $ do
    tg <- tgArray @96 @Float "tg"
    pure $ do
      g <- let_ "g" (groupIndex * 96 + threadIndex)
      acc <- var "acc" 0
      hits <- var "hits" 0
      forStep "j" (simdLane `remE` 3) (load cnt g) 2 $ \j -> do
        acc .= get acc + load x (j `remE` n)
        whenLane (j .== 4) $ do
          acc .= get acc * 1.5
          hits .= get hits + 1
      s <- simdSum "s" (get acc)
      mx <- simdMax "mx" (get acc)
      hs <- simdSum "hs" (get hits)
      whenSimd (simdGroupIndex .== 1) $ do
        t <- simdSum "t" (get acc * 3)
        whenLane (simdLane .< 5) $ store z g t
      tgStore tg threadIndex (s + mx)
      barrier
      store y g (tgLoad tg ((threadIndex + 1) `remE` 96))
      store w g hs
      barrier
      total <- tgReduceSum "total" tg (get acc)
      whenLane (threadIndex .== 0) $ store z (groupIndex * 96 + 95) total

-- | SIMD-group sums where the order shows, in f32 and in f16.
simdSums :: Kernel Cpu
simdSums = kernel "simd_sums" $ do
  x <- buffer @ReadOnly @16 @Float "x" 64
  h <- buffer @ReadOnly @16 @F16 "h" 64
  y <- buffer @WriteOnly @16 @Float "y" 64
  g <- buffer @WriteOnly @16 @F16 "g" 64
  threads @64
  groups 1
  pure . pure $ do
    s <- simdSum "s" (load x threadIndex)
    store y threadIndex s
    t <- simdSum "t" (load h threadIndex)
    store g threadIndex t

{- | A group of 2^24 (2048 in f16) and ones, where lane order rounds every
one away, and one of spread magnitudes; then groups of 2^24 at lane 0 and
ones at lanes 16 and 17 only, which neighbours first sums to 2^24 + 2 and
lane order, or lane 16 first, to 2^24.
-}
simdSumsCase :: Case
simdSumsCase = Case "simd_sums" (checked "simd_sums" simdSums) [run xs hs, run (order 16777216) (order 2048)] []
 where
  run as bs = Inputs [] [("x", map VF32 as), ("h", map (VF16 . narrowF16) bs), ("y", zeros 64), ("g", replicate 64 (VF16 (narrowF16 0)))] []
  xs = [if l == 0 then 16777216 else 1 | l <- [0 .. 31 :: Int]] <> [encodeFloat (1 + fromIntegral l) (l `mod` 41 - 20) * (if even l then 1 else -1) | l <- [0 .. 31 :: Int]]
  hs = [if l == 0 then 2048 else 1 | l <- [0 .. 31 :: Int]] <> [encodeFloat (1 + fromIntegral l) (l `mod` 15 - 8) * (if even l then 1 else -1) | l <- [0 .. 31 :: Int]]
  order big = concat (replicate 2 [if l == 0 then big else if l `elem` [16, 17] then 1 else 0 | l <- [0 .. 31 :: Int]])

-- | fma of each triple, in f32.
fmaF32 :: Kernel Cpu
fmaF32 = kernel "fma_f32" $ do
  n <- dim "n" (DimRange 1 100000 1)
  a <- buffer @ReadOnly @16 @Float "a" n
  b <- buffer @ReadOnly @16 @Float "b" n
  c <- buffer @ReadOnly @16 @Float "c" n
  o <- buffer @WriteOnly @16 @Float "o_fma" n
  threads @64
  groups 1
  pure . pure $ strided "i" n $ \i -> store o i (fmaE (load a i) (load b i) (load c i))

-- | fma of each triple, in f16.
fmaF16 :: Kernel Cpu
fmaF16 = kernel "fma_f16" $ do
  n <- dim "n" (DimRange 1 100000 1)
  a <- buffer @ReadOnly @16 @F16 "a" n
  b <- buffer @ReadOnly @16 @F16 "b" n
  c <- buffer @ReadOnly @16 @F16 "c" n
  o <- buffer @WriteOnly @16 @F16 "o_fma" n
  threads @64
  groups 1
  pure . pure $ strided "i" n $ \i -> store o i (fmaE (load a i) (load b i) (load c i))

-- | Each i32 of a and u32 of b narrowed both ways.
narrowings :: Kernel Cpu
narrowings = kernel "narrowings" $ do
  a <- buffer @ReadOnly @16 @Int32 "a" 16
  b <- buffer @ReadOnly @16 @Word32 "b" 16
  ca <- buffer @WriteOnly @16 @Int8 "clamped_a" 16
  wa <- buffer @WriteOnly @16 @Int8 "wrapped_a" 16
  cb <- buffer @WriteOnly @16 @Word8 "clamped_b" 16
  wb <- buffer @WriteOnly @16 @Word8 "wrapped_b" 16
  threads @32
  groups 1
  pure . pure $ whenLane (threadIndex .< 16) $ do
    store ca threadIndex (narrowClamped @Int8 (load a threadIndex))
    store wa threadIndex (narrowWrapped @Int8 (load a threadIndex))
    store cb threadIndex (narrowClamped @Word8 (load b threadIndex))
    store wb threadIndex (narrowWrapped @Word8 (load b threadIndex))

-- | Each narrowing's edges and values past them.
narrowingsCase :: Case
narrowingsCase =
  Case
    "narrowings"
    (checked "narrowings" narrowings)
    [Inputs [] [("a", map VI32 as), ("b", map VU32 bs), ("clamped_a", i8s), ("wrapped_a", i8s), ("clamped_b", u8s), ("wrapped_b", u8s)] []]
    []
 where
  as = [minBound, minBound + 1, -32768, -300, -129, -128, -127, -1, 0, 1, 126, 127, 128, 300, 65535, maxBound]
  bs = [0, 1, 127, 128, 254, 255, 256, 257, 300, 511, 512, 65535, 65536, 2147483648, maxBound - 1, maxBound]
  i8s = replicate 16 (VI8 0)
  u8s = replicate 16 (VU8 0)

-- | SIMD-group and threadgroup maxima over NaNs and signed zeros.
reductions :: Kernel Cpu
reductions = kernel "reductions" $ do
  x <- buffer @ReadOnly @16 @Float "x" 256
  y <- buffer @WriteOnly @16 @Float "y" 256
  z <- buffer @WriteOnly @16 @Float "z" 4
  threads @64
  groups 4
  pure $ do
    scratch <- tgArray @2 @Float "scratch"
    pure $ do
      g <- let_ "g" (groupIndex * 64 + threadIndex)
      v <- let_ "v" (load x g)
      m <- simdMax "m" v
      store y g m
      t <- tgReduceMax "t" scratch v
      whenLane (threadIndex .== 0) $ store z groupIndex t

-- | Parameters of every type, decoded from the struct's bytes.
params :: Kernel Cpu
params = kernel "params" $ do
  n <- dim "n" (DimRange 1 64 1)
  a <- param @Bool "a"
  b <- param @Float "b"
  c <- param @F16 "c"
  d <- param @Int32 "d"
  e <- param @BF16 "e"
  f <- param @Word8 "f"
  g <- param @Int8 "g"
  y <- buffer @WriteOnly @16 @Float "y" n
  yh <- buffer @WriteOnly @16 @F16 "yh" n
  yi <- buffer @WriteOnly @16 @Int32 "yi" n
  threads @32
  groups 1
  pure . pure $ do
    whenLane (threadIndex .== 0) $ do
      store y 0 (select a b (widen @Float e))
      store y 1 (widen @Float e + toF32 (widen @Word32 f))
      store yh 0 (negate c * (-2.5) + 65504)
      store yi 0 (d + widen @Int32 g)

-- | Ones up to a dimension that is a multiple of 8 from 8 to 64.
ranged :: Kernel Cpu
ranged = kernel "ranged" $ do
  n <- dim "n" (DimRange 8 64 8)
  y <- buffer @WriteOnly @16 @Float "y" 64
  threads @32
  groups 1
  pure . pure $ strided "i" n $ \i -> store y i 1

------------------------------------------------------------------------------
-- Inputs

-- | A NaN out of arithmetic is only a NaN: its payload is unspecified.
byClass :: [Name]
byClass = ["o_add", "o_sub", "o_mul", "o_div", "o_min", "o_max", "o_sqrt", "o_rsqrt", "o_exp", "o_log", "o_tanh"]

arithF32Case :: Case
arithF32Case =
  Case "arith_f32" (checked "arith_f32" arithF32) [Inputs [("n", u count)] (inputs <> outputs) []] byClass
 where
  ps = pairs f32Specials <> [(fromIntegral (w .&. 0xFFFFFFFF), fromIntegral (w `shiftR` 32)) | w <- randoms 50 2000]
  count = length ps
  inputs = [("x", [VF32 (castWord32ToFloat a) | (a, _) <- ps]), ("y", [VF32 (castWord32ToFloat b) | (_, b) <- ps])]
  outputs =
    [(o, zeros count) | o <- ["o_add", "o_sub", "o_mul", "o_div", "o_min", "o_max", "o_sqrt", "o_rsqrt", "o_exp", "o_log", "o_tanh", "o_abs", "o_neg", "o_sel"]]
      <> [("o_cmp", replicate count (VU32 0)), ("o_bf", bf16Zeros count), ("o_h", replicate count (VF16 (F16 0)))]

arithF16Case :: Case
arithF16Case =
  Case "arith_f16" (checked "arith_f16" arithF16) [Inputs [("n", u count)] (inputs <> outputs) []] ("o_lit" : byClass)
 where
  ps = pairs f16Specials <> [(fromIntegral (w .&. 0xFFFF), fromIntegral (w `shiftR` 16 .&. 0xFFFF)) | w <- randoms 51 3000]
  count = length ps
  inputs = [("x", [VF16 (F16 a) | (a, _) <- ps]), ("y", [VF16 (F16 b) | (_, b) <- ps])]
  outputs =
    [(o, replicate count (VF16 (F16 0))) | o <- ["o_add", "o_sub", "o_mul", "o_div", "o_min", "o_max", "o_sqrt", "o_rsqrt", "o_exp", "o_log", "o_tanh", "o_abs", "o_neg", "o_lit"]]
      <> [("o_cmp", replicate count (VU32 0)), ("o_wide", zeros count)]

arithIntCase :: Case
arithIntCase =
  Case "arith_int" (checked "arith_int" arithInt) [Inputs [("n", u count)] (inputs <> outputs) []] []
 where
  is = pairs i32Specials <> [(fromIntegral w, fromIntegral (w `shiftR` 32)) | w <- randoms 52 500]
  us = take count (cycle (pairs u32Specials <> [(fromIntegral w, fromIntegral (w `shiftR` 40)) | w <- randoms 53 500]))
  count = length is
  inputs = [("xi", [VI32 a | (a, _) <- is]), ("yi", [VI32 b | (_, b) <- is]), ("xu", [VU32 a | (a, _) <- us]), ("yu", [VU32 b | (_, b) <- us])]
  outputs =
    [(o, replicate count (VI32 0)) | o <- ["i_add", "i_sub", "i_mul", "i_neg", "i_abs", "i_min", "i_max", "i_div", "i_rem", "i_lit"]]
      <> [(o, replicate count (VU32 0)) | o <- ["i_cmp", "u_add", "u_sub", "u_mul", "u_neg", "u_div", "u_rem", "u_min", "u_cmp"]]
      <> [("u_tof", zeros count)]

{- | Specials against specials, over addends that make ties, cancellation
and overflow; random triples; each random product's rounding error, the
product plus its negated rounded value; and ties settled by an addend
below them.
-}
fmaF32Case :: Case
fmaF32Case = Case "fma_f32" (checked "fma_f32" fmaF32) [Inputs [("n", u count)] (inputs <> [("o_fma", zeros count)]) []] ["o_fma"]
 where
  addends = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x00000001, 0x80000001, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0x28800000, 0x33800000]
  edge = [(a, b, c) | a <- f32Specials, b <- f32Specials, c <- addends]
  rand = [(fromIntegral (w .&. 0xFFFFFFFF), fromIntegral (w `shiftR` 32), fromIntegral (v .&. 0xFFFFFFFF)) | (w, v) <- zip (randoms 56 3000) (randoms 57 3000)]
  errors = [(bits x, bits y, bits (negate (x * y))) | (x, y) <- zip (floats 58 3000) (floats 59 3000)]
  ties =
    [ (0x3F800001, 0x3F7FFFFF, 0x28800000)
    , (0x3F800001, 0x3F7FFFFF, 0xA8800000)
    , (0x3F800001, 0x3F800001, 0xBF800002)
    , (0x3F800800, 0x3F800800, 0xBF801000)
    , (0x7F7FFFFF, 0x40000000, 0xFF7FFFFF)
    , (0x1A000000, 0x1A400000, 0x00000000)
    , (0x9A000000, 0x19800000, 0x00000000)
    , (0x1A000000, 0x1A000000, 0x00000000)
    ]
  triples = edge <> rand <> errors <> ties
  count = length triples
  bits = castFloatToWord32
  f32 = VF32 . castWord32ToFloat
  inputs = [("a", [f32 a | (a, _, _) <- triples]), ("b", [f32 b | (_, b, _) <- triples]), ("c", [f32 c | (_, _, c) <- triples])]

{- | As 'fmaF32Case', in f16: through binary64 on the CPU target, exactly
in the interpreter, so every product's rounding error, every tie and every
subnormal shows where the two would differ.
-}
fmaF16Case :: Case
fmaF16Case = Case "fma_f16" (checked "fma_f16" fmaF16) [Inputs [("n", u count)] (inputs <> [("o_fma", replicate count (VF16 (F16 0)))]) []] ["o_fma"]
 where
  addends = [0x0000, 0x8000, 0x3C00, 0xBC00, 0x0001, 0x8001, 0x0400, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0x1400, 0x3C02]
  edge = [(a, b, c) | a <- f16Specials, b <- f16Specials, c <- addends]
  rand = [(fromIntegral (w .&. 0xFFFF), fromIntegral (w `shiftR` 16 .&. 0xFFFF), fromIntegral (w `shiftR` 32 .&. 0xFFFF)) | w <- randoms 60 6000]
  errors = [(h x, h y, h (negate (widenF16 (narrowF16 (widenF16 (narrowF16 x) * widenF16 (narrowF16 y)))))) | (x, y) <- zip (floats 61 3000) (floats 62 3000)]
  ties = [(0x2740, 0x246A, 0x3C00), (0x2740, 0x246A, 0xBC00), (0x5BF8, 0x5C04, 0x0000), (0x5BF8, 0x5C04, 0xCC00), (0x3C01, 0x3C01, 0xBC02), (0x0C00, 0x0800, 0x0000), (0x0C00, 0x0A00, 0x0000), (0x8C00, 0x0400, 0x0000)]
  triples = edge <> rand <> errors <> ties
  count = length triples
  h v = let F16 b = narrowF16 v in b
  inputs = [(n, [VF16 (F16 (pick t)) | t <- triples]) | (n, pick) <- [("a", \(a, _, _) -> a), ("b", \(_, b, _) -> b), ("c", \(_, _, c) -> c)]]

convCase :: Case
convCase =
  Case
    "conv"
    (checked "conv" conv)
    [ Inputs
        [("n", u count)]
        [ ("xb", bs)
        , ("xh", take count (cycle ([VF16 (F16 h) | h <- f16Specials] <> [VF16 (F16 (fromIntegral (w .&. 0xFFFF))) | w <- randoms 55 3000])))
        , ("xs", take count (cycle [VI8 (fromIntegral i) | i <- [0 .. 255 :: Int]]))
        , ("xc", take count (cycle [VU8 (fromIntegral i) | i <- [0 .. 255 :: Int]]))
        , ("o_b", zeros count)
        , ("o_h", zeros count)
        , ("o_s", replicate count (VI32 0))
        , ("o_c", replicate count (VU32 0))
        , ("o_rb", bf16Zeros count)
        , ("o_rh", replicate count (VF16 (F16 0)))
        , ("o_pb", bf16Zeros count)
        ]
        []
    ]
    []
 where
  bs = [VBF16 (BF16 (fromIntegral (w .&. 0xFFFF))) | w <- randoms 54 4000] <> [VBF16 (BF16 b) | b <- [0x7F81, 0xFF81, 0x7FC0, 0x7FFF, 0x7F80, 0x0001, 0x8000, 0x0000, 0xFFC1]]
  count = length bs

-- | Threadgroups of NaNs alone, NaNs among numbers, signed zeros alone, and numbers.
reductionsCase :: Case
reductionsCase = Case "reductions" (checked "reductions" reductions) [Inputs [] [("x", map VF32 xs), ("y", zeros 256), ("z", zeros 4)] []] ["y"]
 where
  nans = map castWord32ToFloat [0x7FC00000, 0xFFC00001, 0x7F800001, 0xFF812345]
  zeroes = map castWord32ToFloat [0x00000000, 0x80000000]
  xs =
    take 64 (cycle nans)
      <> [if even i then nans !! (i `mod` 4) else fromIntegral (i - 32) | i <- [0 .. 63 :: Int]]
      <> [zeroes !! ((i `div` 3) `mod` 2) | i <- [0 .. 63 :: Int]]
      <> floats 62 63
      <> [castWord32ToFloat 0x7F800000]

rangedCase :: Case
rangedCase = Case "ranged" (checked "ranged" ranged) [Inputs [("n", u 16)] [("y", zeros 64)] []] []

-- | Past its range and off its multiple.
refusals :: [Refusal]
refusals = [Refusal rangedCase [Inputs [("n", u 72)] [("y", zeros 64)] [], Inputs [("n", u 12)] [("y", zeros 64)] []]]

paramsCase :: Case
paramsCase =
  Case
    "params"
    (checked "params" params)
    [ Inputs [("n", u 4), ("a", VBool False), ("b", VF32 1.5), ("c", VF16 (narrowF16 0.75)), ("d", VI32 (-123456)), ("e", VBF16 (narrowBF16 (-2.25))), ("f", VU8 200), ("g", VI8 (-77))] outs []
    , Inputs [("n", u 4), ("a", VBool True), ("b", VF32 1.5), ("c", VF16 (narrowF16 (-1000))), ("d", VI32 maxBound), ("e", VBF16 (narrowBF16 3)), ("f", VU8 1), ("g", VI8 minBound)] outs []
    ]
    []
 where
  outs = [("y", zeros 4), ("yh", replicate 4 (VF16 (F16 0))), ("yi", replicate 4 (VI32 0))]

f32Specials :: [Word32]
f32Specials =
  [ 0x00000000
  , 0x80000000
  , 0x3F800000
  , 0xBF800000
  , 0x3F000000
  , 0x40000000
  , 0x40400000
  , 0x00000001
  , 0x007FFFFF
  , 0x00800000
  , 0x7F7FFFFF
  , 0x7F800000
  , 0xFF800000
  , 0x7FC00000
  , 0x7FC00001
  , 0xFFC00000
  , 0x7F800001
  , 0x7FA00000
  , 0xFF800001
  , 0x477FE000
  , 0x477FF000
  , 0x477FEFFF
  , 0x387FC000
  , 0x38800000
  , 0x33800000
  , 0x33000000
  , 0x33000001
  , 0x3F808000
  , 0x3F818000
  , 0x3F808001
  , 0x7F7F8000
  , 0x3EAAAAAB
  , 0x40490FDB
  , 0xC2C80000
  , 0x42C80000
  , 0x3F801000
  , 0x3F803000
  , 0xC1200000
  , 0x41200000
  , 0x3DCCCCCD
  ]

f16Specials :: [Word16]
f16Specials =
  [ 0x0000
  , 0x8000
  , 0x3C00
  , 0xBC00
  , 0x3800
  , 0x4000
  , 0x4200
  , 0x0001
  , 0x03FF
  , 0x0400
  , 0x7BFF
  , 0xFBFF
  , 0x7C00
  , 0xFC00
  , 0x7E00
  , 0x7E01
  , 0xFE00
  , 0x7C01
  , 0x7D00
  , 0x3555
  , 0x4248
  , 0x5640
  , 0x1400
  , 0x0200
  , 0x3C01
  , 0x7800
  , 0x5BFF
  , 0xC900
  ]

i32Specials :: [Int32]
i32Specials = [0, 1, -1, 2, -2, minBound, maxBound, 7, -7, 3, 65536, -65536, 46341, -46341, 1000000007]

u32Specials :: [Word32]
u32Specials =
  [ 0
  , 1
  , 2
  , 3
  , 0xFFFFFFFF
  , 0x80000000
  , 0x7FFFFFFF
  , 65536
  , 16777216
  , 16777217
  , 16777218
  , 16777219
  , 33554433
  , 33554434
  , 33554435
  , 33554438
  , 0xFFFFFF80
  , 0xFFFFFF7F
  , 0xFFFFFF81
  , 0x7FFFFFC0
  , 0x7FFFFFBF
  ]

pairs :: [a] -> [(a, a)]
pairs xs = [(a, b) | a <- xs, b <- xs]

-- | SplitMix's finaliser, over a seed's sequence.
randoms :: Word64 -> Int -> [Word64]
randoms seed n = [mix (seed * 0x9E3779B97F4A7C15 + fromIntegral i) | i <- [1 .. n]]
 where
  mix z0 =
    let z1 = (z0 `xor` (z0 `shiftR` 30)) * 0xBF58476D1CE4E5B9
        z2 = (z1 `xor` (z1 `shiftR` 27)) * 0x94D049BB133111EB
     in z2 `xor` (z2 `shiftR` 31)

-- | Values in [-2, 2), multiples of 2^-22.
floats :: Word64 -> Int -> [Float]
floats seed n = [fromIntegral (fromIntegral (w `shiftR` 40) - 2 ^ (23 :: Int) :: Int) / 2 ^ (22 :: Int) | w <- randoms seed n]

f32s, bf16s, f16s :: Word64 -> Int -> [Value]
f32s seed = map VF32 . floats seed
bf16s seed = map (VBF16 . narrowBF16) . floats seed
f16s seed = map (VF16 . narrowF16) . floats seed

zeros, bf16Zeros :: Int -> [Value]
zeros n = replicate n (VF32 0)
bf16Zeros n = replicate n (VBF16 (narrowBF16 0))

u :: Int -> Value
u = VU32 . fromIntegral

------------------------------------------------------------------------------
-- The crate's generated half

main :: IO ()
main = do
  [root] <- getArgs
  createDirectoryIfMissing True (root <> "/src/gen")
  runs <- forM cases $ \c -> do
    let k = checkedIR c.caseKernel
    case backendLower cpu [c.caseKernel] of
      Left fs -> error (T.unpack c.caseId <> ": " <> show fs)
      Right l -> T.writeFile (root <> "/src/gen/" <> T.unpack c.caseId <> ".rs") l.loweredSource
    forM (zip [0 :: Int ..] c.caseRuns) $ \(i, inp) -> do
      let name = c.caseId <> "_" <> T.pack (show i)
          dir = "data/" <> name
      createDirectoryIfMissing True (root <> "/" <> T.unpack dir)
      let file f = root <> "/" <> T.unpack dir <> "/" <> f
      BS.writeFile (file "params.bin") (paramBytes (abiOf k) inp.inParams)
      forM_ inp.inBuffers $ \(n, vs) -> BS.writeFile (file ("in_" <> T.unpack n <> ".bin")) (bytes vs)
      forM_ inp.inTables $ \(n, es) -> forM_ (zip [0 :: Int ..] es) $ \(j, vs) ->
        BS.writeFile (file ("tbl_" <> T.unpack n <> "_" <> show j <> ".bin")) (bytes vs)
      case run c.caseKernel (withTables (Map.fromList inp.inTables) (withSeed 7 (launch (Map.fromList inp.inParams) (Map.fromList inp.inBuffers)))) of
        Left e -> error (T.unpack name <> ": the interpreter: " <> show e)
        Right out -> forM_ (Map.toList out) $ \(n, vs) -> BS.writeFile (file ("out_" <> T.unpack n <> ".bin")) (bytes vs)
      pure (runFunction c k name dir inp)
  refused <- forM refusals $ \r -> forM (zip [0 :: Int ..] r.refusalRuns) $ \(i, inp) -> do
    let c = r.refusalCase
        k = checkedIR c.caseKernel
        name = c.caseId <> "_refused_" <> T.pack (show i)
        dir = "data/" <> name
        file f = root <> "/" <> T.unpack dir <> "/" <> f
    createDirectoryIfMissing True (root <> "/" <> T.unpack dir)
    BS.writeFile (file "params.bin") (paramBytes (abiOf k) inp.inParams)
    forM_ inp.inBuffers $ \(n, vs) -> BS.writeFile (file ("in_" <> T.unpack n <> ".bin")) (bytes vs)
    message <- case run c.caseKernel (launch (Map.fromList inp.inParams) (Map.fromList inp.inBuffers)) of
      Left (DimOutOfRange d v range) -> pure (k.kernelName <> ": the dimension " <> d <> " is " <> T.pack (show v) <> ", outside its range " <> dimRangeText range <> ".")
      other -> error (T.unpack name <> ": the interpreter doesn't refuse it: " <> show (void other))
    pure (refusedFunction c k name dir message)
  T.writeFile (root <> "/src/cases.rs") (casesModule (zip cases runs) (concat refused))
  T.writeFile (root <> "/src/launch.rs") (launchModule [(c.caseId, checkedIR c.caseKernel) | c <- cases])

-- | Each case's run: its inputs loaded, the kernel run on them, its outputs compared, and its launch sizes.
runFunction :: Case -> IR.Kernel -> Text -> Text -> Inputs -> (Text, [Text])
runFunction c k name dir inp =
  ( "run_" <> name
  , ["fn run_" <> name <> "() -> usize {", "    let p = read(\"" <> dir <> "/params.bin\");"]
      <> [ "    let " <> mutable b <> rustVar b.bufName <> ": Vec<" <> rustScalarType b.bufType <> "> = load(\"" <> dir <> "/in_" <> b.bufName <> ".bin\");"
         | b <- k.kernelBuffers
         ]
      <> concat
        [ [ "    let " <> rustVar t.tableName <> "_entries: Vec<Vec<" <> rustScalarType t.tableType <> ">> = (0.." <> tshow (length entries) <> ").map(|j| load(&format!(\"" <> dir <> "/tbl_" <> t.tableName <> "_{j}.bin\"))).collect();"
          , "    let " <> rustVar t.tableName <> ": Vec<&[" <> rustScalarType t.tableType <> "]> = " <> rustVar t.tableName <> "_entries.iter().map(Vec::as_slice).collect();"
          ]
        | t <- k.kernelTables
        , let entries = fromMaybe [] (lookup t.tableName inp.inTables)
        ]
      <> [ "    " <> c.caseId <> "::" <> rustIdent k.kernelName <> "(" <> T.intercalate ", " arguments <> ");"
         , "    let mut bad = 0;"
         ]
      <> [ "    bad += compare(\"" <> name <> "\", \"" <> b.bufName <> "\", " <> how b.bufName <> ", &" <> rustVar b.bufName <> ", &load::<" <> rustScalarType b.bufType <> ">(\"" <> dir <> "/out_" <> b.bufName <> ".bin\"));"
         | b <- k.kernelBuffers
         , b.bufAccess /= IR.ReadOnly
         ]
      <> [ "    bad += size(\"" <> name <> "\", \"" <> what <> "\", crate::launch::" <> c.caseId <> "::" <> what <> "(" <> T.intercalate ", " [tshow v | v <- dimValues e] <> "), " <> tshow (evalSize e) <> ");"
         | (what, e) <- launchSizes k
         ]
      <> ["    bad", "}"]
  )
 where
  mutable b = if b.bufAccess == IR.ReadOnly then "" else "mut "
  arguments = ["&p"] <> [(if b.bufAccess == IR.ReadOnly then "&" else "&mut ") <> rustVar b.bufName | b <- k.kernelBuffers] <> ["&" <> rustVar t.tableName | t <- k.kernelTables]
  how n = if n `elem` c.caseByClass then "Compare::NanClass" else "Compare::Bits"
  dims = Map.fromList [(n, w) | (n, VU32 w) <- inp.inParams]
  dimValues e = [Map.findWithDefault 0 d dims | d <- dimsOf e]
  evalSize :: Expr -> Integer
  evalSize = \case
    ELit (LU32 v) -> toInteger v
    EParam n -> toInteger (Map.findWithDefault 0 n dims)
    EBin op a b ->
      let x = evalSize a
          y = evalSize b
       in case op of
            IR.Add -> x + y
            IR.Sub -> x - y
            IR.Mul -> x * y
            IR.Div -> x `div` y
            IR.Rem -> x `mod` y
            _ -> error "a launch size operator"
    _ -> error "a launch size"

-- | A refused launch's run: the kernel's panic caught, its message against the one expected.
refusedFunction :: Case -> IR.Kernel -> Text -> Text -> Text -> (Text, [Text])
refusedFunction c k name dir message =
  ( "run_" <> name
  , ["fn run_" <> name <> "() -> usize {", "    let p = read(\"" <> dir <> "/params.bin\");"]
      <> [ "    let " <> mutable b <> rustVar b.bufName <> ": Vec<" <> rustScalarType b.bufType <> "> = load(\"" <> dir <> "/in_" <> b.bufName <> ".bin\");"
         | b <- k.kernelBuffers
         ]
      <> [ "    let got = refusal(|| " <> c.caseId <> "::" <> rustIdent k.kernelName <> "(" <> T.intercalate ", " arguments <> "));"
         , "    refused(\"" <> name <> "\", got.as_deref(), \"" <> message <> "\")"
         , "}"
         ]
  )
 where
  mutable b = if b.bufAccess == IR.ReadOnly then "" else "mut "
  arguments = ["&p"] <> [(if b.bufAccess == IR.ReadOnly then "&" else "&mut ") <> rustVar b.bufName | b <- k.kernelBuffers]

casesModule :: [(Case, [(Text, [Text])])] -> [(Text, [Text])] -> Text
casesModule runs refused =
  T.unlines $
    [ "// Generated by checks.cpu from test/cpu/Cases.hs."
    , ""
    , "// The kernels' buffer names, as they are."
    , "#![allow(clippy::similar_names)]"
    , ""
    , "use crate::{Compare, compare, load, read, refusal, refused, size};"
    , ""
    ]
      <> concat [["#[path = \"gen/" <> c.caseId <> ".rs\"]", "mod " <> c.caseId <> ";"] | (c, _) <- runs]
      <> ["", "/// Every case's mismatches.", "pub fn run() -> usize {", "    let mut bad = 0;"]
      <> ["    bad += " <> f <> "();" | (_, rs) <- runs, (f, _) <- rs]
      <> ["    bad += " <> f <> "();" | (f, _) <- refused]
      <> ["    bad", "}"]
      <> concat [[""] <> body | (_, rs) <- runs, (_, body) <- rs]
      <> concat [[""] <> body | (_, body) <- refused]

{- | Per kernel, what runtimes' generators put together from
"FuryKernel.Emit.Rust": the parameter struct with its layout asserted, and
the launch sizes.
-}
launchModule :: [(Text, IR.Kernel)] -> Text
launchModule ks =
  T.unlines $
    [ "// Generated by checks.cpu from test/cpu/Cases.hs: per kernel, what runtimes'"
    , "// generators put together from FuryKernel.Emit.Rust."
    , ""
    , "#![allow(dead_code)]"
    ]
      <> concat
        [ ["", "pub mod " <> name <> " {"]
            <> (if null k.kernelParams then [] else T.lines (rustParamsStruct "Params" (abiOf k)))
            <> concat
              [ [ ""
                , "    #[must_use]"
                , "    pub fn " <> what <> "(" <> T.intercalate ", " [rustIdent d <> ": u32" | d <- dimsOf e] <> ") -> u64 {"
                , "        " <> rustLaunchExpr e
                , "    }"
                ]
              | (what, e) <- launchSizes k
              ]
            <> ["}"]
        | (name, k) <- ks
        ]

-- | A kernel's launch sizes: its grid, each buffer's extent, each table's entries and extent.
launchSizes :: IR.Kernel -> [(Text, Expr)]
launchSizes k =
  [("groups", k.kernelGroups)]
    <> [("extent_" <> b.bufName, b.bufExtent) | b <- k.kernelBuffers]
    <> concat [[("entries_" <> t.tableName, t.tableEntries), ("extent_" <> t.tableName, t.tableExtent)] | t <- k.kernelTables]

-- | The dimensions a launch size reads, in order.
dimsOf :: Expr -> [Name]
dimsOf = nub . go
 where
  go = \case
    EParam n -> [n]
    EBin _ a b -> go a <> go b
    _ -> []

rustVar :: Name -> Text
rustVar n = "v_" <> n

tshow :: (Show a) => a -> Text
tshow = T.pack . show

paramBytes :: Abi -> [(Name, Value)] -> BS.ByteString
paramBytes abi ps =
  BS.pack [Map.findWithDefault 0 i placed | i <- [0 .. fromIntegral abi.abiParamsSize - 1 :: Int]]
 where
  placed = Map.fromList (concat [zip [fromIntegral f.fieldOffset ..] (maybe [] valueBytes (lookup f.fieldName ps)) | f <- abi.abiParams])

bytes :: [Value] -> BS.ByteString
bytes = BS.pack . concatMap valueBytes

valueBytes :: Value -> [Word8]
valueBytes = \case
  VF32 x -> le32 (castFloatToWord32 x)
  VF16 (F16 w) -> le16 w
  VBF16 (BF16 w) -> le16 w
  VI32 n -> le32 (fromIntegral n)
  VU32 n -> le32 n
  VI8 n -> [fromIntegral n]
  VU8 n -> [n]
  VBool b -> [if b then 1 else 0]
 where
  le16 w = [fromIntegral w, fromIntegral (w `shiftR` 8)]
  le32 w = [fromIntegral (w `shiftR` s) | s <- [0, 8, 16, 24 :: Int]]
