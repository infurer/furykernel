-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The CPU target: checked kernels as Rust, so that a runtime without a GPU
runs the kernels its GPU backends run.

The Rust follows the interpreter's model ("FuryKernel.Interp"): a
threadgroup's threads run each statement together, as lanes of vectors.
Every local is a vector with a lane per thread; a statement runs for the
active lanes in order; a branch splits them and a loop keeps those whose
condition holds, so lanes reconverge after the construct as on the GPU.
Barriers need no code, since no lane runs ahead. SIMD-group reductions
combine each group's lanes pairwise, in the IR's order. Threadgroups run one after
another.

The arithmetic is the interpreter's too: binary32 as Rust computes it,
binary16 rounded after every operation, conversions to bf16 and f16
rounding to nearest with ties to even (from binary32 the @half@ crate's;
from binary64 its own, since the crate's goes through binary32 on x86 and
drops low bits in its portable code), integers wrapping, minimum and
maximum ignoring a NaN operand. Rsqrt, Exp, Log and Tanh are computed in
binary64, by the platform's libm as the interpreter's are, and rounded
once to the operand's type. A fused multiply-add of binary32 is Rust's
@mul_add@; of binary16, the exact product, which binary64 holds, plus the
addend, rounded in binary64 (exactly, unless the product is too small to
change the binary16 result) and then once to binary16.

A kernel asserts that its dimensions lie in their declared ranges and on
their multiples before it computes anything from them ('rustDimChecks'):
those are the values the gate proves its launch sizes for, and one outside
them panics with the kernel's name, the dimension, its value and its range,
where the interpreter refuses the launch.

What generated code needs: the @half@ crate, its types spelled with their
paths. A kernel is a function of its parameter struct's bytes, its buffers
as slices (shared ones read-only, mutable ones written) and its tables as
slices of the arrays their entries point at, in that order.
-}
module FuryKernel.Cpu (
  Cpu,
  cpu,
  rustLanguage,
) where

import Control.Monad.Trans.State.Strict (State, evalState, gets, modify')
import Data.Bits (shiftR, (.&.))
import Data.Containers.ListUtils (nubOrd)
import Data.List (nub, sortOn)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import Data.Text qualified as T
import Data.Word (Word32)
import FuryKernel.Abi (Abi (..), Field (..), abiOf, hex64, irHash)
import FuryKernel.Backend (Backend (..), Lowered (..), LoweredKernel (..), compileOptions)
import FuryKernel.Check (Checked, checkedIR)
import FuryKernel.Emit.Rust (rustDimChecks, rustIdent, rustLaunchExpr, rustTypeName)
import FuryKernel.Finding (Finding (..), Rule (..))
import FuryKernel.IR hiding (Kernel (..))
import FuryKernel.IR qualified as IR
import FuryKernel.Target (Target (..))
import FuryKernel.Type (ScalarType (..))
import GHC.Float (castFloatToWord32)
import Numeric (showHex)

-- | A CPU, with the limits of the portable kernels' reference.
data Cpu

instance Target Cpu where
  type TargetName Cpu = "Cpu"
  type SimdWidth Cpu = 32
  type MaxThreads Cpu = 1024
  type TgMemBytes Cpu = 32768
  type MaxBuffers Cpu = 31
  type LaneLoadMax Cpu = 16
  targetFragments = [TBF16, TF16, TF32]

-- | The language the source is in.
rustLanguage :: Text
rustLanguage = "rust"

cpu :: Backend Cpu
cpu =
  Backend
    { backendName = "cpu"
    , backendLower = \ks -> case clashes (map checkedIR ks) of
        [] -> Right (lower ks)
        findings -> Left findings
    }

{- | Kernels one module can't hold together: two of one name, or two whose
names make one Rust type name ('rustTypeName', the name of their
parameter structs).
-}
clashes :: [IR.Kernel] -> [Finding]
clashes ks =
  [ Finding Names ("The kernel " <> n <> " is lowered " <> T.pack (show c) <> " times into one module.")
  | (n, c) <- Map.toList (Map.fromListWith (+) [(k.kernelName, 1 :: Int) | k <- ks])
  , c > 1
  ]
    <> [ Finding Names ("The kernels " <> T.intercalate ", " names <> " make one Rust type name, " <> t <> ".")
       | (t, names) <- Map.toList (Map.fromListWith (flip (<>)) [(rustTypeName n, [n]) | n <- nubOrd (map (.kernelName) ks)])
       , length names > 1
       ]

{- | Rust, compiled as it is: its arithmetic is the IR's
('FuryKernel.Backend.Strict'), and the language has no versions to name.
-}
lower :: [Checked t] -> Lowered
lower ks =
  Lowered
    { loweredLanguage = rustLanguage
    , loweredOptions = compileOptions
    , loweredSource = renderRust ks
    , loweredKernels = map table ks
    }
 where
  table c =
    LoweredKernel
      { lkName = k.kernelName
      , lkEntry = k.kernelName
      , lkAbi = abiOf k
      , lkIrHash = irHash c
      , lkThreads = k.kernelThreads
      , lkGroups = k.kernelGroups
      , lkBindings = zip (map (.bufName) k.kernelBuffers <> map (.tableName) k.kernelTables) [0 ..]
      , lkParamsBinding = Nothing
      , lkTgMemory = snd (tgLayout k.kernelTgArrays)
      }
   where
    k = checkedIR c

-- | One Rust module for a set of kernels, in name order.
renderRust :: [Checked t] -> Text
renderRust ks =
  T.unlines $
    [ "// Generated by FuryKernel's CPU target. Do not edit: regenerate."
    , ""
    , "//! Kernels, as FuryKernel's CPU target lowers them: lanes in step."
    , ""
    , "#![allow(clippy::all, clippy::pedantic, dead_code, unused_mut, unused_parens, unused_variables, unused_comparisons, non_snake_case)]"
    , ""
    , "/// The `N` bytes of a parameter struct from `at`."
    , "fn fk_bytes<const N: usize>(params: &[u8], at: usize) -> [u8; N] {"
    , "    let mut bytes = [0u8; N];"
    , "    bytes.copy_from_slice(&params[at..at + N]);"
    , "    bytes"
    , "}"
    , ""
    , "/// A select: both sides evaluated, as the IR's are."
    , "fn fk_select<T>(c: bool, a: T, b: T) -> T {"
    , "    if c { a } else { b }"
    , "}"
    , ""
    , "/// IEEE 754-2019 minimumNumber: a NaN gives way to the other operand, two"
    , "/// NaNs give the quiet NaN, and -0 is below +0."
    , "fn fk_min(a: f32, b: f32) -> f32 {"
    , "    if a.is_nan() {"
    , "        if b.is_nan() { f32::from_bits(0x7fc0_0000) } else { b }"
    , "    } else if b.is_nan() || a < b {"
    , "        a"
    , "    } else if b < a {"
    , "        b"
    , "    } else {"
    , "        f32::from_bits(a.to_bits() | b.to_bits())"
    , "    }"
    , "}"
    , ""
    , "/// IEEE 754-2019 maximumNumber: a NaN gives way to the other operand, two"
    , "/// NaNs give the quiet NaN, and +0 is above -0."
    , "fn fk_max(a: f32, b: f32) -> f32 {"
    , "    if a.is_nan() {"
    , "        if b.is_nan() { f32::from_bits(0x7fc0_0000) } else { b }"
    , "    } else if b.is_nan() || a > b {"
    , "        a"
    , "    } else if b > a {"
    , "        b"
    , "    } else {"
    , "        f32::from_bits(a.to_bits() & b.to_bits())"
    , "    }"
    , "}"
    , ""
    , "/// A bf16 widened, exactly; a NaN comes out quiet."
    , "fn fk_widen_bf16(x: ::half::bf16) -> f32 {"
    , "    let y = x.to_f32();"
    , "    if y.is_nan() { f32::from_bits(y.to_bits() | 0x0040_0000) } else { y }"
    , "}"
    , ""
    , "/// An f16 widened, exactly; a NaN comes out quiet."
    , "fn fk_widen_f16(x: ::half::f16) -> f32 {"
    , "    let y = x.to_f32();"
    , "    if y.is_nan() { f32::from_bits(y.to_bits() | 0x0040_0000) } else { y }"
    , "}"
    , ""
    , "/// A truncated value rounded up when the dropped bits are above half, or half"
    , "/// and the kept value odd."
    , "fn fk_round_shifted(kept: u64, dropped: u64, width: i32) -> u64 {"
    , "    let half = 1u64 << (width - 1);"
    , "    if dropped > half || (dropped == half && kept & 1 == 1) { kept + 1 } else { kept }"
    , "}"
    , ""
    , "/// An f64 rounded once to the nearest f16, ties to even, from all its bits:"
    , "/// overflow to infinity, gradual underflow; a NaN comes out quiet, its sign"
    , "/// and top payload bits kept."
    , "fn fk_narrow_f16_f64(x: f64) -> ::half::f16 {"
    , "    let w = x.to_bits();"
    , "    let sign = ((w >> 48) & 0x8000) as u16;"
    , "    let e = ((w >> 52) & 0x7ff) as i32;"
    , "    let m = w & 0x000f_ffff_ffff_ffff;"
    , "    let unbiased = e - 1023;"
    , "    let magnitude = if e == 0x7ff {"
    , "        if m != 0 { 0x7e00 | (m >> 42) as u16 } else { 0x7c00 }"
    , "    } else if unbiased >= 16 {"
    , "        0x7c00"
    , "    } else if unbiased >= -14 {"
    , "        let kept = (((unbiased + 15) as u64) << 10) | (m >> 42);"
    , "        fk_round_shifted(kept, m & ((1u64 << 42) - 1), 42) as u16"
    , "    } else {"
    , "        // In units of 2^-24: the significand times 2^(unbiased - 28)."
    , "        let shift = 28 - unbiased;"
    , "        if shift > 53 {"
    , "            0"
    , "        } else {"
    , "            let full = m | (1u64 << 52);"
    , "            fk_round_shifted(full >> shift, full & ((1u64 << shift) - 1), shift) as u16"
    , "        }"
    , "    };"
    , "    ::half::f16::from_bits(sign | magnitude)"
    , "}"
    , ""
    , "/// An f16 fused multiply-add: the exact product plus c, rounded in f64"
    , "/// (exactly, unless the product is too small to change the f16 result),"
    , "/// then once to f16."
    , "fn fk_fma_f16(a: ::half::f16, b: ::half::f16, c: ::half::f16) -> ::half::f16 {"
    , "    fk_narrow_f16_f64(f64::from(fk_widen_f16(a)) * f64::from(fk_widen_f16(b)) + f64::from(fk_widen_f16(c)))"
    , "}"
    ]
      <> concatMap kernelSource (sortOn ((.kernelName) . checkedIR) ks)

kernelSource :: Checked t -> [Text]
kernelSource c =
  [ ""
  , "/// `" <> k.kernelName <> "`: IR hash " <> hex64 (irHash c) <> ", ABI hash " <> hex64 abi.abiHash <> "."
  , "pub fn " <> rustIdent k.kernelName <> "("
  ]
    <> map (("    " <>) . (<> ",")) arguments
    <> [") {"]
    <> map ("    " <>) params
    <> map ("    " <>) (rustDimChecks k.kernelName k.kernelParams)
    <> [ "    const FK_THREADS: usize = " <> showT k.kernelThreads <> ";"
       , "    const FK_WIDTH: usize = " <> showT simdWidth <> ";"
       , "    let fk_groups = (" <> rustLaunchExpr k.kernelGroups <> ") as usize;"
       ]
    <> ["    let mut " <> rustIdent n <> ": Vec<" <> rustType t <> "> = vec![" <> zero t <> "; FK_THREADS];" | (n, t) <- locals]
    <> ["    let mut " <> rustIdent a.tgName <> ": Vec<" <> rustType a.tgType <> "> = vec![" <> zero a.tgType <> "; " <> showT a.tgCount <> "];" | a <- k.kernelTgArrays]
    <> ["    let mut " <> rustIdent n <> ": Vec<[f32; 64]> = vec![[0.0f32; 64]; FK_THREADS.div_ceil(FK_WIDTH)];" | n <- fragments]
    <> [ "    let fk_all: Vec<usize> = (0..FK_THREADS).collect();"
       , "    for fk_group in 0..fk_groups {"
       , "        let fk_group = fk_group as u32;"
       , "        let fk_act: &[usize] = &fk_all;"
       ]
    <> evalState (concat <$> traverse (stmt 2) k.kernelBody) (initialEnv k)
    <> [ "    }"
       , "}"
       ]
 where
  k = checkedIR c
  abi = abiOf k
  arguments =
    ["fk_p: &[u8]"]
      <> [ rustIdent b.bufName <> ": " <> (if b.bufAccess == ReadOnly then "&[" else "&mut [") <> rustType b.bufType <> "]"
         | b <- k.kernelBuffers
         ]
      <> [rustIdent t.tableName <> ": &[&[" <> rustType t.tableType <> "]]" | t <- k.kernelTables]
  params = [paramLine f | f <- abi.abiParams]
  paramLine f =
    "let "
      <> rustIdent f.fieldName
      <> ": "
      <> rustType f.fieldType
      <> " = "
      <> decode f.fieldType ("fk_bytes(fk_p, " <> showT f.fieldOffset <> ")")
      <> ";"
  decode t bytes = case t of
    TF32 -> "f32::from_ne_bytes(" <> bytes <> ")"
    TF16 -> "::half::f16::from_bits(u16::from_ne_bytes(" <> bytes <> "))"
    TBF16 -> "::half::bf16::from_bits(u16::from_ne_bytes(" <> bytes <> "))"
    TI32 -> "i32::from_ne_bytes(" <> bytes <> ")"
    TU32 -> "u32::from_ne_bytes(" <> bytes <> ")"
    TI8 -> "i8::from_ne_bytes(" <> bytes <> ")"
    TU8 -> "u8::from_ne_bytes(" <> bytes <> ")"
    TBool -> "u8::from_ne_bytes(" <> bytes <> ") != 0"
  -- Every local, once: names are unique in a checked kernel.
  locals = nub (concatMap bound k.kernelBody)
  -- Every fragment: 64 elements a SIMD group, held as f32.
  fragments = nub (concatMap fragmentsOf k.kernelBody)
  fragmentsOf = \case
    SFragVar n _ -> [n]
    SIf _ _ th el -> concatMap fragmentsOf th <> concatMap fragmentsOf el
    SFor _ _ _ _ _ body -> concatMap fragmentsOf body
    _ -> []
  bound = \case
    SLet n t _ -> [(n, t)]
    SVar n t _ -> [(n, t)]
    SSimd n _ t _ -> [(n, t)]
    SLoadPack ns t _ _ -> [(n, t) | n <- ns]
    SLoadPackVia ns t _ _ _ -> [(n, t) | n <- ns]
    SIf _ _ th el -> concatMap bound th <> concatMap bound el
    SFor _ v _ _ _ body -> (v, TU32) : concatMap bound body
    _ -> []
  simdWidth = 32 :: Int

------------------------------------------------------------------------------
-- Types

data Env = Env
  { envLocals :: Map Name ScalarType
  , envParams :: Map Name ScalarType
  , envBuffers :: Map Name ScalarType
  , envArrays :: Map Name ScalarType
  , envTables :: Map Name ScalarType
  }

initialEnv :: IR.Kernel -> Env
initialEnv k =
  Env
    { envLocals = Map.empty
    , envParams = Map.fromList [(p.paramName, p.paramType) | p <- k.kernelParams]
    , envBuffers = Map.fromList [(b.bufName, b.bufType) | b <- k.kernelBuffers]
    , envArrays = Map.fromList [(a.tgName, a.tgType) | a <- k.kernelTgArrays]
    , envTables = Map.fromList [(t.tableName, t.tableType) | t <- k.kernelTables]
    }

declare :: Name -> ScalarType -> State Env ()
declare n t = modify' (\e -> e {envLocals = Map.insert n t e.envLocals})

typeOf :: Env -> Expr -> ScalarType
typeOf env = \case
  ELit l -> literalType l
  EVar n -> Map.findWithDefault TF32 n env.envLocals
  EParam n -> Map.findWithDefault TU32 n env.envParams
  EBuiltin _ -> TU32
  EUn _ e -> typeOf env e
  EBin op a _
    | op `elem` [Eq, Ne, Lt, Le, Gt, Ge, And, Or] -> TBool
    | otherwise -> typeOf env a
  ESelect _ a _ -> typeOf env a
  EFma a _ _ -> typeOf env a
  EConvert t _ -> t
  ENarrowInt _ t _ -> t
  ELoad n _ -> Map.findWithDefault TF32 n env.envBuffers
  ELoadVia n _ _ -> Map.findWithDefault TF32 n env.envTables
  ETgLoad n _ -> Map.findWithDefault TF32 n env.envArrays

------------------------------------------------------------------------------
-- Statements: each runs for the active lanes, in order

stmt :: Int -> Stmt -> State Env [Text]
stmt depth = \case
  SLet n t e -> do
    s <- forLanes (\e' -> lane n <> " = " <> e' <> ";") e
    declare n t
    pure s
  SVar n t e -> do
    s <- forLanes (\e' -> lane n <> " = " <> e' <> ";") e
    declare n t
    pure s
  SAssign n e -> forLanes (\e' -> lane n <> " = " <> e' <> ";") e
  SStore b i e -> do
    i' <- ex i
    e' <- ex e
    pure (loop ["let fk_value = " <> e' <> ";", rustIdent b <> "[" <> index i' <> "] = fk_value;"])
  STgStore a i e -> do
    i' <- ex i
    e' <- ex e
    pure (loop ["let fk_value = " <> e' <> ";", rustIdent a <> "[" <> index i' <> "] = fk_value;"])
  SLoadPack ns t b i -> do
    i' <- ex i
    mapM_ (`declare` t) ns
    pure . loop $
      ["let fk_at = " <> index i' <> " * " <> showT (length ns) <> ";"]
        <> [lane n <> " = " <> rustIdent b <> "[fk_at + " <> showT j <> "];" | (j, n) <- zip [0 :: Int ..] ns]
  SLoadPackVia ns t b e i -> do
    e' <- ex e
    i' <- ex i
    mapM_ (`declare` t) ns
    pure . loop $
      [ "let fk_array = " <> rustIdent b <> "[" <> index e' <> "];"
      , "let fk_at = " <> index i' <> " * " <> showT (length ns) <> ";"
      ]
        <> [lane n <> " = fk_array[fk_at + " <> showT j <> "];" | (j, n) <- zip [0 :: Int ..] ns]
  SStorePack b i es -> do
    i' <- ex i
    es' <- traverse ex es
    pure . loop $
      ["let fk_at = " <> index i' <> " * " <> showT (length es) <> ";"]
        <> ["let fk_value_" <> showT j <> " = " <> e' <> ";" | (j, e') <- zip [0 :: Int ..] es']
        <> [rustIdent b <> "[fk_at + " <> showT j <> "] = fk_value_" <> showT j <> ";" | j <- [0 .. length es - 1]]
  STgStorePack a i es -> do
    i' <- ex i
    es' <- traverse ex es
    pure . loop $
      ["let fk_at = " <> index i' <> " * " <> showT (length es) <> ";"]
        <> ["let fk_value_" <> showT j <> " = " <> e' <> ";" | (j, e') <- zip [0 :: Int ..] es']
        <> [rustIdent a <> "[fk_at + " <> showT j <> "] = fk_value_" <> showT j <> ";" | j <- [0 .. length es - 1]]
  SIf _ c th el -> do
    c' <- ex c
    th' <- concat <$> traverse (stmt (depth + 2)) th
    el' <- concat <$> traverse (stmt (depth + 2)) el
    pure $
      [ pad <> "{"
      , if null el
          then pad1 <> "let fk_then: Vec<usize> = fk_act.iter().copied().filter(|&fk_l| " <> c' <> ").collect();"
          else pad1 <> "let (fk_then, fk_else): (Vec<usize>, Vec<usize>) = fk_act.iter().copied().partition(|&fk_l| " <> c' <> ");"
      , pad1 <> "if !fk_then.is_empty() {"
      , pad2 <> "let fk_act: &[usize] = &fk_then;"
      ]
        <> th'
        <> [pad1 <> "}"]
        <> ( if null el
               then []
               else
                 [ pad1 <> "if !fk_else.is_empty() {"
                 , pad2 <> "let fk_act: &[usize] = &fk_else;"
                 ]
                   <> el'
                   <> [pad1 <> "}"]
           )
        <> [pad <> "}"]
  SFor _ v start end step body -> do
    start' <- ex start
    declare v TU32
    end' <- ex end
    step' <- ex step
    body' <- concat <$> traverse (stmt (depth + 3)) body
    pure $
      [ pad <> "{"
      , pad1 <> "for &fk_l in fk_act {"
      , pad2 <> lane v <> " = " <> start' <> ";"
      , pad1 <> "}"
      , pad1 <> "let mut fk_live: Vec<usize> = fk_act.to_vec();"
      , pad1 <> "loop {"
      , pad2 <> "fk_live.retain(|&fk_l| " <> lane v <> " < " <> end' <> ");"
      , pad2 <> "if fk_live.is_empty() {"
      , pad3 <> "break;"
      , pad2 <> "}"
      , pad2 <> "{"
      , pad3 <> "let fk_act: &[usize] = &fk_live;"
      ]
        <> body'
        <> [ pad2 <> "}"
           , pad2 <> "for &fk_l in &fk_live {"
           , pad3 <> lane v <> " = " <> lane v <> ".wrapping_add(" <> step' <> ");"
           , pad2 <> "}"
           , pad1 <> "}"
           , pad <> "}"
           ]
  SBarrier -> pure [pad <> "// A barrier: the lanes run in step."]
  SSimdBarrier -> pure [pad <> "// A SIMD-group barrier: the lanes run in step."]
  SSimd n op t e -> do
    e' <- ex e
    declare n t
    let combine a b = case op of
          SimdSum -> arith Add t a b
          SimdMax -> arith Max t a b
          SimdMin -> arith Min t a b
    pure
      [ pad <> "{"
      , pad1 <> "let mut fk_i = 0;"
      , pad1 <> "while fk_i < fk_act.len() {"
      , pad2 <> "let fk_simd = fk_act[fk_i] / FK_WIDTH;"
      , pad2 <> "let mut fk_j = fk_i;"
      , pad2 <> "let mut fk_v: [" <> rustType t <> "; FK_WIDTH] = [Default::default(); FK_WIDTH];"
      , pad2 <> "while fk_j < fk_act.len() && fk_act[fk_j] / FK_WIDTH == fk_simd {"
      , pad3 <> "let fk_l = fk_act[fk_j];"
      , pad3 <> "fk_v[fk_j - fk_i] = " <> e' <> ";"
      , pad3 <> "fk_j += 1;"
      , pad2 <> "}"
      , pad2 <> "// Pairwise, neighbours first, as the IR orders a reduction."
      , pad2 <> "let mut fk_n = fk_j - fk_i;"
      , pad2 <> "while fk_n > 1 {"
      , pad3 <> "for fk_k in 0..fk_n / 2 {"
      , pad4 <> "fk_v[fk_k] = " <> combine "fk_v[2 * fk_k]" "fk_v[2 * fk_k + 1]" <> ";"
      , pad3 <> "}"
      , pad3 <> "if fk_n % 2 == 1 {"
      , pad4 <> "fk_v[fk_n / 2] = fk_v[fk_n - 1];"
      , pad3 <> "}"
      , pad3 <> "fk_n = fk_n.div_ceil(2);"
      , pad2 <> "}"
      , pad2 <> "for &fk_l in &fk_act[fk_i..fk_j] {"
      , pad3 <> lane n <> " = fk_v[0];"
      , pad2 <> "}"
      , pad2 <> "fk_i = fk_j;"
      , pad1 <> "}"
      , pad <> "}"
      ]
  SFragVar n _ -> pure (simdGroups [rustIdent n <> "[fk_simd] = [0.0f32; 64];"])
  SFragLoad n m i st tr -> do
    i' <- ex i
    st' <- ex st
    (memory, t) <- fragMemory m
    let at = if tr then "fk_c * fk_stride + fk_r" else "fk_r * fk_stride + fk_c"
    pure . simdGroups $
      [ "let fk_base = " <> index i' <> ";"
      , "let fk_stride = " <> index st' <> ";"
      , "for fk_r in 0..8 {"
      , "    for fk_c in 0..8 {"
      , "        " <> rustIdent n <> "[fk_simd][fk_r * 8 + fk_c] = " <> widenTo t (memory <> "[fk_base + " <> at <> "]") <> ";"
      , "    }"
      , "}"
      ]
  SFragStore m i st n -> do
    i' <- ex i
    st' <- ex st
    (memory, t) <- fragMemory m
    pure . simdGroups $
      [ "let fk_base = " <> index i' <> ";"
      , "let fk_stride = " <> index st' <> ";"
      , "for fk_r in 0..8 {"
      , "    for fk_c in 0..8 {"
      , "        " <> memory <> "[fk_base + fk_r * fk_stride + fk_c] = " <> narrowTo t (rustIdent n <> "[fk_simd][fk_r * 8 + fk_c]") <> ";"
      , "    }"
      , "}"
      ]
  SFragMma d a b c ->
    pure . simdGroups $
      [ "let mut fk_out = [0.0f32; 64];"
      , "for fk_r in 0..8 {"
      , "    for fk_c in 0..8 {"
      , "        let mut fk_acc = " <> rustIdent c <> "[fk_simd][fk_r * 8 + fk_c];"
      , "        for fk_k in 0..8 {"
      , "            fk_acc = fk_acc + " <> rustIdent a <> "[fk_simd][fk_r * 8 + fk_k] * " <> rustIdent b <> "[fk_simd][fk_k * 8 + fk_c];"
      , "        }"
      , "        fk_out[fk_r * 8 + fk_c] = fk_acc;"
      , "    }"
      , "}"
      , rustIdent d <> "[fk_simd] = fk_out;"
      ]
 where
  pad = T.replicate (4 * depth) " "
  pad1 = T.replicate (4 * (depth + 1)) " "
  pad2 = T.replicate (4 * (depth + 2)) " "
  pad3 = T.replicate (4 * (depth + 3)) " "
  pad4 = T.replicate (4 * (depth + 4)) " "
  ex e = gets (`expr` e)
  forLanes line e = do
    e' <- ex e
    pure (loop [line e'])
  loop body = [pad <> "for &fk_l in fk_act {"] <> map (pad1 <>) body <> [pad <> "}"]
  index i = "(" <> i <> ") as usize"
  -- Once per SIMD group with active lanes, at its first: fk_simd and fk_l.
  simdGroups body =
    [ pad <> "{"
    , pad1 <> "let mut fk_i = 0;"
    , pad1 <> "while fk_i < fk_act.len() {"
    , pad2 <> "let fk_simd = fk_act[fk_i] / FK_WIDTH;"
    , pad2 <> "let fk_l = fk_act[fk_i];"
    , pad2 <> "let mut fk_j = fk_i + 1;"
    , pad2 <> "while fk_j < fk_act.len() && fk_act[fk_j] / FK_WIDTH == fk_simd {"
    , pad3 <> "fk_j += 1;"
    , pad2 <> "}"
    ]
      <> map (pad2 <>) body
      <> [ pad2 <> "fk_i = fk_j;"
         , pad1 <> "}"
         , pad <> "}"
         ]
  -- A fragment's memory, and its element type.
  fragMemory = \case
    FDevice b -> gets (\e -> (rustIdent b, Map.findWithDefault TF32 b e.envBuffers))
    FThreadgroup a -> gets (\e -> (rustIdent a, Map.findWithDefault TF32 a e.envArrays))
  widenTo t v = case t of
    TBF16 -> "fk_widen_bf16(" <> v <> ")"
    TF16 -> "fk_widen_f16(" <> v <> ")"
    _ -> v
  narrowTo t v = case t of
    TBF16 -> "::half::bf16::from_f32(" <> v <> ")"
    TF16 -> "::half::f16::from_f32(" <> v <> ")"
    _ -> v

-- | A local of the current lane.
lane :: Name -> Text
lane n = rustIdent n <> "[fk_l]"

------------------------------------------------------------------------------
-- Expressions, at lane fk_l

expr :: Env -> Expr -> Text
expr env = go
 where
  go = \case
    ELit l -> literal l
    EVar n -> lane n
    EParam n -> rustIdent n
    EBuiltin b -> case b of
      ThreadInGroup -> "(fk_l as u32)"
      GroupInGrid -> "fk_group"
      SimdLane -> "((fk_l % FK_WIDTH) as u32)"
      SimdGroup -> "((fk_l / FK_WIDTH) as u32)"
    EUn op e -> unary op (typeOf env e) (go e)
    EBin op a b -> arith op (typeOf env a) (go a) (go b)
    ESelect c a b -> "fk_select(" <> go c <> ", " <> go a <> ", " <> go b <> ")"
    EFma a b c -> fusedMulAdd (typeOf env a) (go a) (go b) (go c)
    EConvert t e -> convert (typeOf env e) t (go e)
    ENarrowInt o t e -> narrowInt o t (go e)
    ELoad n i -> rustIdent n <> "[" <> idx (go i) <> "]"
    ELoadVia n e i -> rustIdent n <> "[" <> idx (go e) <> "][" <> idx (go i) <> "]"
    ETgLoad n i -> rustIdent n <> "[" <> idx (go i) <> "]"
  idx i = "(" <> i <> ") as usize"

unary :: UnOp -> ScalarType -> Text -> Text
unary op t e = case (op, t) of
  (Not, _) -> "(!" <> e <> ")"
  (Neg, TF16) -> "(-" <> e <> ")"
  (Neg, TF32) -> "(-" <> e <> ")"
  (Neg, _) -> e <> ".wrapping_neg()"
  (Abs, TF32) -> e <> ".abs()"
  (Abs, TF16) -> "::half::f16::from_bits(" <> e <> ".to_bits() & 0x7fff)"
  (Abs, TU32) -> e
  (Abs, _) -> e <> ".wrapping_abs()"
  (Sqrt, TF16) -> half ("fk_widen_f16(" <> e <> ").sqrt()")
  (Sqrt, _) -> e <> ".sqrt()"
  (_, TF16) -> "fk_narrow_f16_f64(" <> inBinary64 ("f64::from(fk_widen_f16(" <> e <> "))") <> ")"
  _ -> "(" <> inBinary64 ("f64::from(" <> e <> ")") <> " as f32)"
 where
  -- In binary64, the result rounded once to the operand's type.
  inBinary64 x = case op of
    Rsqrt -> "(1.0f64 / " <> x <> ".sqrt())"
    Exp -> x <> ".exp()"
    Log -> x <> ".ln()"
    Tanh -> x <> ".tanh()"
    _ -> x

-- | 'IR.EFma': binary32's by Rust's @mul_add@, binary16's through binary64 (@fk_fma_f16@).
fusedMulAdd :: ScalarType -> Text -> Text -> Text -> Text
fusedMulAdd t a b c = case t of
  TF16 -> "fk_fma_f16(" <> a <> ", " <> b <> ", " <> c <> ")"
  _ -> "f32::mul_add(" <> a <> ", " <> b <> ", " <> c <> ")"

arith :: BinOp -> ScalarType -> Text -> Text -> Text
arith op t a b = case t of
  TF32 -> case op of
    Min -> "fk_min(" <> a <> ", " <> b <> ")"
    Max -> "fk_max(" <> a <> ", " <> b <> ")"
    _ -> infixOp
  TF16 -> case op of
    _ | comparison -> infixOp
    Min -> half ("fk_min(fk_widen_f16(" <> a <> "), fk_widen_f16(" <> b <> "))")
    Max -> half ("fk_max(fk_widen_f16(" <> a <> "), fk_widen_f16(" <> b <> "))")
    _ -> half ("fk_widen_f16(" <> a <> ") " <> symbol <> " fk_widen_f16(" <> b <> ")")
  TBool -> infixOp
  _ -> case op of
    Add -> a <> ".wrapping_add(" <> b <> ")"
    Sub -> a <> ".wrapping_sub(" <> b <> ")"
    Mul -> a <> ".wrapping_mul(" <> b <> ")"
    Min -> a <> ".min(" <> b <> ")"
    Max -> a <> ".max(" <> b <> ")"
    _ -> infixOp
 where
  comparison = op `elem` [Eq, Ne, Lt, Le, Gt, Ge]
  infixOp = "(" <> a <> " " <> symbol <> " " <> b <> ")"
  symbol = case op of
    Add -> "+"
    Sub -> "-"
    Mul -> "*"
    Div -> "/"
    Rem -> "%"
    Eq -> "=="
    Ne -> "!="
    Lt -> "<"
    Le -> "<="
    Gt -> ">"
    Ge -> ">="
    And -> "&"
    Or -> "|"
    Min -> "min"
    Max -> "max"

half :: Text -> Text
half x = "::half::f16::from_f32(" <> x <> ")"

convert :: ScalarType -> ScalarType -> Text -> Text
convert from to e = case (from, to) of
  _ | from == to -> e
  (TBF16, TF32) -> "fk_widen_bf16(" <> e <> ")"
  (TF16, TF32) -> "fk_widen_f16(" <> e <> ")"
  (TI8, TI32) -> "i32::from(" <> e <> ")"
  (TU8, TU32) -> "u32::from(" <> e <> ")"
  (TF32, TBF16) -> "::half::bf16::from_f32(" <> e <> ")"
  (TF32, TF16) -> "::half::f16::from_f32(" <> e <> ")"
  -- Integers to binary32 round to nearest, ties to even.
  (_, TF32) -> "(" <> e <> " as f32)"
  _ -> e

-- | An integer narrowed: clamped to the narrower range, or its low bits.
narrowInt :: Overflow -> ScalarType -> Text -> Text
narrowInt o t e = case (o, t) of
  (Clamp, TI8) -> "((" <> e <> ").clamp(-128, 127) as i8)"
  (Clamp, _) -> "((" <> e <> ").min(255) as u8)"
  (Wrap, TI8) -> "((" <> e <> ") as i8)"
  (Wrap, _) -> "((" <> e <> ") as u8)"

literal :: Literal -> Text
literal = \case
  LF32 x -> "f32::from_bits(" <> hex32 (castFloatToWord32 x) <> ")"
  LF16 b -> "::half::f16::from_bits(" <> hex16 (fromIntegral b) <> ")"
  LBF16 b -> "::half::bf16::from_bits(" <> hex16 (fromIntegral b) <> ")"
  LI32 n
    | n < 0 -> "(" <> grouped n <> "i32)"
    | otherwise -> grouped n <> "i32"
  LU32 n -> grouped n <> "u32"
  LI8 n
    | n < 0 -> "(" <> showT n <> "i8)"
    | otherwise -> showT n <> "i8"
  LU8 n -> showT n <> "u8"
  LBool b -> if b then "true" else "false"
 where
  hex16 :: Word32 -> Text
  hex16 w = "0x" <> T.justifyRight 4 '0' (T.pack (showHex w ""))
  hex32 w = "0x" <> T.justifyRight 4 '0' (T.pack (showHex (w `shiftR` 16) "")) <> "_" <> T.justifyRight 4 '0' (T.pack (showHex (w .&. 0xffff) ""))

-- | Digits in groups of three, as clippy reads long literals best.
grouped :: (Show a) => a -> Text
grouped n = case T.uncons (T.pack (show n)) of
  Just ('-', digits) -> "-" <> groups digits
  _ -> groups (T.pack (show n))
 where
  groups = T.reverse . T.intercalate "_" . T.chunksOf 3 . T.reverse

rustType :: ScalarType -> Text
rustType = \case
  TF32 -> "f32"
  TF16 -> "::half::f16"
  TBF16 -> "::half::bf16"
  TI32 -> "i32"
  TU32 -> "u32"
  TI8 -> "i8"
  TU8 -> "u8"
  TBool -> "bool"

zero :: ScalarType -> Text
zero = \case
  TF32 -> "0.0"
  TF16 -> "::half::f16::ZERO"
  TBF16 -> "::half::bf16::ZERO"
  TBool -> "false"
  t -> "0" <> fromMaybe "" (lookup t [(TI32, "i32"), (TU32, "u32"), (TI8, "i8"), (TU8, "u8")])

showT :: (Show a) => a -> Text
showT = T.pack . show
