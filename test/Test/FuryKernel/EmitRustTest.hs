-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

-- | The pieces of generated Rust, golden.
module Test.FuryKernel.EmitRustTest (tests) where

import Data.Text qualified as T
import FuryKernel.Abi (abiOf)
import FuryKernel.Build (kernelIR)
import FuryKernel.Emit.Rust
import FuryKernel.IR (BinOp (..), Expr (..), Literal (..))
import Test.FuryKernel.Kernels (rmsnorm)
import Test.FuryKernel.Targets (T32)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (testCase, (@?=))

tests :: TestTree
tests =
  testGroup
    "Emit.Rust"
    [ testCase "names" $ do
        rustIdent "rows" @?= "rows"
        rustIdent "type" @?= "r#type"
        rustTypeName "rmsnorm_x_h2816" @?= "RmsnormXH2816"
    , testCase "launch sizes in u64" $ do
        let size = EBin Mul (EParam "rows") (ELit (LU32 2816))
        rustLaunchExpr size @?= "u64::from(rows) * 2_816u64"
        rustLaunchTerm size @?= "(u64::from(rows) * 2_816u64)"
        rustLaunchExpr (EBin Add size (ELit (LU32 1))) @?= "(u64::from(rows) * 2_816u64) + 1u64"
        rustLaunchTerm (EParam "rows") @?= "u64::from(rows)"
        rustLaunchTerm (ELit (LU32 738197504)) @?= "738_197_504u64"
        let rows = ELit (LU32 64)
        rustLaunchExpr (EBin Div (EBin Sub (EBin Add (EParam "m") rows) (ELit (LU32 1))) rows) @?= "u64::from(m).div_ceil(64u64)"
        rustLaunchExpr (EBin Div (EBin Add (EParam "m") (EBin Sub rows (ELit (LU32 1)))) rows) @?= "u64::from(m).div_ceil(64u64)"
        rustLaunchExpr (EBin Div (EBin Sub (EBin Add (EParam "m") rows) (ELit (LU32 1))) (ELit (LU32 32))) @?= "((u64::from(m) + 64u64) - 1u64) / 32u64"
    , testCase "a parameter struct and its layout assertions" $
        rustParamsStruct "RmsnormParams" (abiOf (kernelIR (rmsnorm @T32 2816)))
          @?= T.unlines
            [ "#[repr(C)]"
            , "#[derive(Clone, Copy, Debug, PartialEq)]"
            , "pub struct RmsnormParams {"
            , "    pub rows: u32,"
            , "    pub eps: f32,"
            , "}"
            , ""
            , "const _: () = {"
            , "    assert!(::core::mem::size_of::<RmsnormParams>() == 8);"
            , "    assert!(::core::mem::offset_of!(RmsnormParams, rows) == 0);"
            , "    assert!(::core::mem::offset_of!(RmsnormParams, eps) == 4);"
            , "};"
            ]
    ]
