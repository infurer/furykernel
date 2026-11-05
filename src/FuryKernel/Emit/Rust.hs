-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The target-independent pieces of generated Rust: names, scalar types,
launch-size expressions, the checks that hold dimensions to their ranges
before a launch size is computed from them, and parameter structs whose
layout is asserted at compile time. Runtimes' generators put them together
with their own types.

16-bit floats are spelled with the @half@ crate's types, and integer
literals with their digits in groups of three, so that the output passes
clippy's pedantic lints.
-}
module FuryKernel.Emit.Rust (
  rustIdent,
  rustTypeName,
  rustScalarType,
  rustLaunchExpr,
  rustLaunchTerm,
  rustDimChecks,
  dimRangeText,
  rustParamsStruct,
) where

import Data.Char (toUpper)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as T
import FuryKernel.Abi (Abi (..), Field (..))
import FuryKernel.IR (BinOp (..), DimRange (..), Expr (..), Literal (..), Name, Param (..), ParamRole (..))
import FuryKernel.Type (ScalarType (..))

-- | A FuryKernel name as a Rust identifier: raw where it is a Rust keyword.
rustIdent :: Name -> Text
rustIdent n
  | n `Set.member` keywords = "r#" <> n
  | otherwise = n
 where
  keywords =
    Set.fromList . T.words $
      "as async await break const continue dyn else enum extern false fn for gen if impl in let loop match mod move mut pub ref return static struct trait true try type unsafe use where while abstract become box do final macro override priv typeof unsized virtual yield"

-- | @rmsnorm_x_h2816@ as @RmsnormXH2816@.
rustTypeName :: Name -> Text
rustTypeName = T.concat . map capital . filter (not . T.null) . T.splitOn "_"
 where
  capital w = case T.uncons w of
    Just (c, rest) -> T.cons (toUpper c) rest
    Nothing -> w

rustScalarType :: ScalarType -> Text
rustScalarType = \case
  TF32 -> "f32"
  TF16 -> "::half::f16"
  TBF16 -> "::half::bf16"
  TI32 -> "i32"
  TU32 -> "u32"
  TI8 -> "i8"
  TU8 -> "u8"
  TBool -> "bool"

{- | A launch size (a grid or an extent) in u64 arithmetic over u32
variables named after the dimensions. The gate keeps every step of a
launch size in u32, dividing by no zero, for every legal value of its
dimensions ("FuryKernel.Check.Launch"), so for dimensions in their ranges
this agrees with the interpreter's u32 arithmetic and the runtime's. The
whole expression has no parentheses around it; see 'rustLaunchTerm' to
embed one in another.
-}
rustLaunchExpr :: Expr -> Text
rustLaunchExpr = \case
  -- A quotient rounded up, (x + y - 1) / y, as clippy wants it written.
  EBin Div (EBin Sub (EBin Add x y) (ELit (LU32 1))) y' | y == y' -> divCeil x y
  EBin Div (EBin Add x (EBin Sub y (ELit (LU32 1)))) y' | y == y' -> divCeil x y
  EBin op a b -> rustLaunchTerm a <> " " <> binOp op <> " " <> rustLaunchTerm b
  e -> rustLaunchTerm e
 where
  divCeil x y = rustLaunchTerm x <> ".div_ceil(" <> rustLaunchExpr y <> ")"

-- | A launch size as an operand: in parentheses unless it is atomic.
rustLaunchTerm :: Expr -> Text
rustLaunchTerm = \case
  ELit (LU32 n) -> grouped n <> "u64"
  EParam n -> "u64::from(" <> rustIdent n <> ")"
  e@(EBin {}) -> "(" <> rustLaunchExpr e <> ")"
  e -> error ("rustLaunchTerm: not a launch size: " <> show e)

{- | A kernel's dimensions held to their declared ranges and multiples,
the values the gate proves its launch sizes for: an @assert!@ for each
dimension whose range says anything, over a @u32@ variable of its name,
before anything is computed from it. One outside its range panics with
the kernel's name, the dimension, its value and its range.
-}
rustDimChecks :: Name -> [Param] -> [Text]
rustDimChecks kernel ps =
  [ "assert!(" <> T.intercalate " && " conditions <> ", \"" <> kernel <> ": the dimension " <> p.paramName <> " is " <> shown <> ", outside its range " <> dimRangeText r <> ".\"" <> argument <> ");"
  | p <- ps
  , Dim r <- [p.paramRole]
  , let v = rustIdent p.paramName
        conditions = [v <> " >= " <> grouped r.dimMin | r.dimMin > 0] <> [v <> " <= " <> grouped r.dimMax | r.dimMax < 4294967295] <> [v <> " % " <> grouped r.dimMultipleOf <> " == 0" | r.dimMultipleOf > 1]
        -- A raw identifier can't be named in a format string.
        (shown, argument) = if v == p.paramName then ("{" <> v <> "}", "") else ("{}", ", " <> v)
  , not (null conditions)
  ]

-- | A dimension's range as messages and the IR's text give it: @8..64 multiple of 8@, @1..16@.
dimRangeText :: DimRange -> Text
dimRangeText r = T.pack (show r.dimMin) <> ".." <> T.pack (show r.dimMax) <> (if r.dimMultipleOf > 1 then " multiple of " <> T.pack (show r.dimMultipleOf) else "")

-- | Digits in groups of three, as clippy wants long literals written.
grouped :: (Show a) => a -> Text
grouped = T.reverse . T.intercalate "_" . T.chunksOf 3 . T.reverse . T.pack . show

binOp :: BinOp -> Text
binOp = \case
  Add -> "+"
  Sub -> "-"
  Mul -> "*"
  Div -> "/"
  Rem -> "%"
  op -> error ("rustLaunchExpr: not a launch operator: " <> show op)

{- | A @#[repr(C)]@ struct with the kernel's parameters, and assertions
that Rust lays it out as the ABI says.
-}
rustParamsStruct :: Text -> Abi -> Text
rustParamsStruct name abi =
  T.unlines $
    [ "#[repr(C)]"
    , "#[derive(Clone, Copy, Debug, PartialEq)]"
    , "pub struct " <> name <> " {"
    ]
      <> ["    pub " <> rustIdent f.fieldName <> ": " <> rustScalarType f.fieldType <> "," | f <- abi.abiParams]
      <> [ "}"
         , ""
         , "const _: () = {"
         , "    assert!(::core::mem::size_of::<" <> name <> ">() == " <> showT abi.abiParamsSize <> ");"
         ]
      <> [ "    assert!(::core::mem::offset_of!(" <> name <> ", " <> rustIdent f.fieldName <> ") == " <> showT f.fieldOffset <> ");"
         | f <- abi.abiParams
         ]
      <> ["};"]
 where
  showT :: (Show a) => a -> Text
  showT = T.pack . show
