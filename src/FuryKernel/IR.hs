-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The kernel IR: untyped, first-order and structured.

Every pass and every check works on one concrete tree of this type; the
typed surface ("FuryKernel.Build") only constructs it. There are no
closures, no recursion and no goto: control flow is structured if and for,
so that uniformity, hazards and bounds stay decidable by walking the tree.

Names are the author's: locals, parameters, buffers and threadgroup arrays
share one namespace, and the backends print them unchanged.
-}
module FuryKernel.IR (
  -- * Names
  Name,

  -- * Expressions
  Literal (..),
  literalType,
  canonicalNaN,
  UnOp (..),
  BinOp (..),
  Builtin (..),
  Overflow (..),
  Expr (..),

  -- * Statements
  Region (..),
  SimdOp (..),
  FragMemory (..),
  Stmt (..),

  -- * Kernels
  Access (..),
  DimRange (..),
  ParamRole (..),
  Param (..),
  Buffer (..),
  Table (..),
  TgArray (..),
  tgAlign,
  tgBytes,
  tgLayout,
  Kernel (..),

  -- * Rendering
  renderKernel,
  renderExpr,
) where

import Data.Bits ((.&.))
import Data.Int (Int32, Int8)
import Data.List (sortOn)
import Data.Ord (Down (..))
import Data.Text (Text)
import Data.Text qualified as T
import Data.Word (Word16, Word32, Word8)
import FuryKernel.Type (ScalarType (..), leastAlign, scalarSize, scalarTypeName)
import GHC.Float (castFloatToWord32)
import Numeric (showHex)
import Numeric.Natural (Natural)

type Name = Text

data Literal
  = LF32 Float
  | -- | The bits of a binary16.
    LF16 Word16
  | -- | The bits of a bfloat16.
    LBF16 Word16
  | LI32 Int32
  | LU32 Word32
  | LI8 Int8
  | LU8 Word8
  | LBool Bool
  deriving stock (Show)

{- | Literals compare by their bits, as their text does: a NaN equals
itself, and -0 differs from +0.
-}
instance Eq Literal where
  a == b = literalBits a == literalBits b

-- | A literal's type and bits.
literalBits :: Literal -> (Int, Word32)
literalBits = \case
  LF32 x -> (0, castFloatToWord32 x)
  LF16 b -> (1, fromIntegral b)
  LBF16 b -> (2, fromIntegral b)
  LI32 n -> (3, fromIntegral n)
  LU32 n -> (4, n)
  LI8 n -> (5, fromIntegral n)
  LU8 n -> (6, fromIntegral n)
  LBool b -> (7, if b then 1 else 0)

{- | Whether a literal is no NaN, or the one NaN of its type the IR holds:
the quiet NaN with no payload and no sign (binary32 0x7fc00000, binary16
0x7e00, bfloat16 0x7fc0).
-}
canonicalNaN :: Literal -> Bool
canonicalNaN = \case
  LF32 x -> not (isNaN x) || castFloatToWord32 x == 0x7fc00000
  LF16 b -> b .&. 0x7c00 /= 0x7c00 || b .&. 0x03ff == 0 || b == 0x7e00
  LBF16 b -> b .&. 0x7f80 /= 0x7f80 || b .&. 0x007f == 0 || b == 0x7fc0
  _ -> True

literalType :: Literal -> ScalarType
literalType = \case
  LF32 _ -> TF32
  LF16 _ -> TF16
  LBF16 _ -> TBF16
  LI32 _ -> TI32
  LU32 _ -> TU32
  LI8 _ -> TI8
  LU8 _ -> TU8
  LBool _ -> TBool

{- | Unary operators. Of floats, 'Sqrt' is correctly rounded; 'Rsqrt',
'Exp', 'Log' and 'Tanh' are within a bound that each target states. The
interpreter and the CPU target compute those four in binary64 and round the
result once to the operand's type: correctly rounded except, rarely, where
the exact value lies within the binary64 result's error of a midpoint (a
few 2^-29 ulp of binary32, 2^-42 of binary16), so that Linux and macOS give
the same bits but there. binary32 subnormals are as 'BinOp' has them,
'Neg' and 'Abs' counting as arithmetic.
-}
data UnOp
  = {- | A float's sign flipped, a NaN's payload kept (IEEE 754's negate); an
    integer's two's complement, wrapping around.
    -}
    Neg
  | Not
  | {- | A float's sign cleared, a NaN's payload kept (IEEE 754's abs); an
    i32's magnitude, the minimum its own.
    -}
    Abs
  | Sqrt
  | -- | 1 / sqrt x.
    Rsqrt
  | Exp
  | Log
  | Tanh
  deriving stock (Eq, Ord, Show, Enum, Bounded)

{- | Binary operators. Floats round as IEEE 754 does, binary16 after every
operation; a NaN out of arithmetic is only a NaN, its payload unspecified.
On i32 and u32, addition, subtraction and multiplication wrap around, two's
complement; division by zero, and the i32 minimum divided by -1, are
errors.

binary32 subnormals are a target's choice: a target may take a subnormal
binary32 operand as a zero of its sign, in arithmetic, comparisons and
conversions that compute, and may give a zero of the result's sign for a
subnormal binary32 result. An operation that returns one of its operands
('Min', 'Max', 'ESelect', a SIMD-group minimum or maximum) may return it
unflushed, and of operands that compare equal as zeros, any one of them.
Loads, stores and exact widenings keep the bits. binary16 has no such
leave. The interpreter and the CPU target keep IEEE 754's subnormals.
-}
data BinOp
  = Add
  | Sub
  | Mul
  | -- | Floats' quotient; integers' rounds toward zero.
    Div
  | -- | Integer remainder, with the dividend's sign.
    Rem
  | {- | Floats: IEEE 754-2019's minimumNumber, except for choices left to the
    target: of -0 and +0 either may come out, two NaNs give a NaN, either of
    them or another, and binary32 subnormals are as above. A NaN operand,
    quiet or signalling, gives way to the other. The interpreter and the
    CPU target put -0 below +0 and give the quiet NaN with no payload.
    -}
    Min
  | {- | Floats: IEEE 754-2019's maximumNumber, as 'Min' is minimumNumber;
    the interpreter and the CPU target put +0 above -0.
    -}
    Max
  | Eq
  | Ne
  | Lt
  | Le
  | Gt
  | Ge
  | -- | Of bools, both operands evaluated: the second, whatever the first.
    And
  | -- | Of bools, both operands evaluated: the second, whatever the first.
    Or
  deriving stock (Eq, Ord, Show, Enum, Bounded)

-- | Positions every thread knows. One-dimensional threadgroups and grids.
data Builtin
  = -- | The thread's index in its threadgroup.
    ThreadInGroup
  | -- | The threadgroup's index in the grid.
    GroupInGrid
  | -- | The thread's lane in its SIMD group.
    SimdLane
  | -- | The SIMD group's index in its threadgroup.
    SimdGroup
  deriving stock (Eq, Ord, Show, Enum, Bounded)

-- | What narrowing an integer does with a value out of the narrower type's range.
data Overflow
  = -- | It clamps the value to the range's nearer end.
    Clamp
  | -- | It wraps the value around, keeping its low bits (two's complement).
    Wrap
  deriving stock (Eq, Ord, Show, Enum, Bounded)

data Expr
  = ELit Literal
  | -- | A local, bound by a let, a var, a loop or a SIMD-group operation.
    EVar Name
  | -- | A field of the parameter struct: a dimension or a scalar.
    EParam Name
  | EBuiltin Builtin
  | EUn UnOp Expr
  | EBin BinOp Expr Expr
  | -- | Per-element choice: condition, then, else. Both sides are evaluated.
    ESelect Expr Expr Expr
  | {- | @a × b + c@ of floats, IEEE 754's fusedMultiplyAdd: the exact value
    rounded once in the operands' type, f32 or f16 (bf16 and the integers
    have none). A NaN operand gives a NaN, as do @∞ × 0@ and @0 × ∞@ whatever
    @c@, and an infinite product plus the infinity of the other sign. Signed
    zeros follow IEEE 754's sum of the exact product, signed as @a@ times
    @b@, and @c@: an exact zero is -0 only if both are -0, and a nonzero
    value that rounds to zero keeps its sign. binary32 subnormals are as
    'BinOp' has them.
    -}
    EFma Expr Expr Expr
  | {- | Conversion to the given type: widening exactly, narrowing to
    nearest with ties to even, integers to floats rounding to nearest. A
    narrowing may take a binary32 subnormal as a zero of its sign, as
    'BinOp' has it; a widening keeps the bits.
    -}
    EConvert ScalarType Expr
  | {- | An integer narrowed to a narrower integer type, i32 to i8 or u32 to
    u8, a value out of its range clamped or wrapped, as the narrowing says.
    No integer narrows otherwise.
    -}
    ENarrowInt Overflow ScalarType Expr
  | -- | An element of a device buffer.
    ELoad Name Expr
  | -- | An element of the array a table's entry points at: table, entry, index.
    ELoadVia Name Expr Expr
  | -- | An element of a threadgroup array.
    ETgLoad Name Expr
  deriving stock (Eq, Show)

{- | How uniform control is across the threads that reach a statement, as
the builder declared it: across the threadgroup, across each SIMD group,
or per lane. The gate verifies the declarations.
-}
data Region
  = RUniform
  | RSimdUniform
  | RDivergent
  deriving stock (Eq, Ord, Show, Enum, Bounded)

{- | Where a fragment is loaded from or stored to: a device buffer or a
threadgroup array, by name.
-}
data FragMemory
  = FDevice Name
  | FThreadgroup Name
  deriving stock (Eq, Show)

{- | Reductions across the lanes of a SIMD group, by 'Add', 'Max' and 'Min',
each lane getting the same value. The lanes combine pairwise, neighbours
first, then pairs of pairs, each operation rounding in the operands' type:
for a sum, @((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7)) + ...@, as
an xor butterfly over the lanes @l xor 1@, @l xor 2@, ..., @l xor (w / 2)@
computes it. binary32 subnormals are as 'BinOp' has them: a maximum or
minimum of lanes that compare equal as zeros may be any one of them.
-}
data SimdOp
  = SimdSum
  | SimdMax
  | SimdMin
  deriving stock (Eq, Ord, Show, Enum, Bounded)

data Stmt
  = -- | An immutable local.
    SLet Name ScalarType Expr
  | -- | A mutable local and its initial value.
    SVar Name ScalarType Expr
  | SAssign Name Expr
  | -- | Buffer, index, value.
    SStore Name Expr Expr
  | -- | Threadgroup array, index, value.
    STgStore Name Expr Expr
  | {- | Locals bound to a pack: consecutive elements of a buffer, which a
    lane reads at once. The locals, their type, the buffer and the index
    in packs: local @i@ is element @index * n + i@, for @n@ locals.
    -}
    SLoadPack [Name] ScalarType Name Expr
  | -- | Buffer, index in packs, the values: a pack written at once.
    SStorePack Name Expr [Expr]
  | -- | Threadgroup array, index in packs, the values: a pack written at once.
    STgStorePack Name Expr [Expr]
  | {- | Locals bound to a pack read through a table: the locals, their
    type, the table, the entry and the index in packs.
    -}
    SLoadPackVia [Name] ScalarType Name Expr Expr
  | -- | The declared uniformity of the condition, the condition, then, else.
    SIf Region Expr [Stmt] [Stmt]
  | {- | The declared uniformity of the loop's control, the loop variable
    (u32), its start, the exclusive end and the step; the end and the
    step are evaluated anew at every iteration, as in C. A loop whose
    variable stands still (a step of 0) or wraps past u32's greatest is
    one that never ends.
    -}
    SFor Region Name Expr Expr Expr [Stmt]
  | -- | A threadgroup barrier over threadgroup memory.
    SBarrier
  | -- | A SIMD-group barrier over threadgroup memory: its lanes see each other's stores.
    SSimdBarrier
  | -- | A local bound to a SIMD-group reduction of the given type.
    SSimd Name SimdOp ScalarType Expr
  | {- | A fragment: an 8×8 matrix of the given element type that a SIMD
    group holds together, every element zero. Fragments are variables:
    loads and products write them.
    -}
    SFragVar Name ScalarType
  | {- | A fragment loaded from memory: the fragment, the memory, the index
    of its first element, the elements between one of its rows and the
    next, and whether it is the memory's 8×8 block transposed: element
    @(r, c)@ is @index + r * stride + c@, or @index + c * stride + r@.
    -}
    SFragLoad Name FragMemory Expr Expr Bool
  | {- | @d = a · b + c@, of fragments: the result, then @a@, @b@ and @c@;
    the result may be @c@.
    -}
    SFragMma Name Name Name Name
  | -- | A fragment stored, element @(r, c)@ at @index + r * stride + c@.
    SFragStore FragMemory Expr Expr Name
  deriving stock (Eq, Show)

data Access
  = ReadOnly
  | WriteOnly
  | ReadWrite
  deriving stock (Eq, Ord, Show, Enum, Bounded)

-- | The values a dimension may take: a range and a divisor.
data DimRange = DimRange
  { dimMin :: Natural
  , dimMax :: Natural
  , dimMultipleOf :: Natural
  }
  deriving stock (Eq, Show)

data ParamRole
  = -- | A u32 that sizes the launch.
    Dim DimRange
  | Scalar
  deriving stock (Eq, Show)

-- | A field of the kernel's one parameter struct.
data Param = Param
  { paramName :: Name
  , paramType :: ScalarType
  , paramRole :: ParamRole
  }
  deriving stock (Eq, Show)

data Buffer = Buffer
  { bufName :: Name
  , bufAccess :: Access
  , bufType :: ScalarType
  , bufAlign :: Natural
  -- ^ The alignment of the buffer's start, in bytes.
  , bufExtent :: Expr
  -- ^ The number of elements the kernel may touch, over the parameters.
  }
  deriving stock (Eq, Show)

{- | A table of device addresses, which the kernel only reads through: entry
@e@ is the start of an array of @tableExtent@ elements of @tableType@,
aligned to @tableAlign@ bytes. The table itself holds @tableEntries@
64-bit addresses. Experts' weights are reached this way, wherever the
engine keeps them.
-}
data Table = Table
  { tableName :: Name
  , tableType :: ScalarType
  , tableAlign :: Natural
  , tableEntries :: Expr
  -- ^ Entries in the table, over the parameters.
  , tableExtent :: Expr
  -- ^ Elements of each entry's array the kernel may touch.
  }
  deriving stock (Eq, Show)

{- | An array in threadgroup memory, aligned to 'tgAlign' and taking
'tgBytes' of it in the kernel's layout ('tgLayout').
-}
data TgArray = TgArray
  { tgName :: Name
  , tgType :: ScalarType
  , tgCount :: Natural
  }
  deriving stock (Eq, Show)

-- | The alignment of a threadgroup array, in bytes: 'leastAlign' of its type.
tgAlign :: TgArray -> Natural
tgAlign a = leastAlign a.tgType

-- | What a threadgroup array takes: its size, rounded up to its alignment.
tgBytes :: TgArray -> Natural
tgBytes a = (a.tgCount * fromIntegral (scalarSize a.tgType) + tgAlign a - 1) `div` tgAlign a * tgAlign a

{- | A kernel's threadgroup layout: each array's offset in bytes, in the
order they lie, and the total, which targets allocate. The arrays lie in
the order declared, the more aligned first, each taking 'tgBytes'; so each
starts at a multiple of its alignment, and the total is the sum of what
they take, in any order. The gate checks the total against the target's
limit, and manifests and launch tables report it.
-}
tgLayout :: [TgArray] -> ([(Name, Natural)], Natural)
tgLayout arrays = (zip (map (.tgName) placed) (scanl (+) 0 (map tgBytes placed)), sum (map tgBytes arrays))
 where
  placed = sortOn (Down . tgAlign) arrays

data Kernel = Kernel
  { kernelName :: Name
  , kernelThreads :: Natural
  -- ^ Threads per threadgroup, fixed when the kernel is generated.
  , kernelSimdWidth :: Natural
  {- ^ The lanes of a SIMD group on the target it was built for, which the
  body takes as given.
  -}
  , kernelGroups :: Expr
  -- ^ Threadgroups in the grid, over the parameters.
  , kernelParams :: [Param]
  , kernelBuffers :: [Buffer]
  , kernelTables :: [Table]
  -- ^ Bound after the buffers.
  , kernelTgArrays :: [TgArray]
  , kernelBody :: [Stmt]
  }
  deriving stock (Eq, Show)

------------------------------------------------------------------------------
-- Rendering

{- | A deterministic text form, for golden tests, messages and the IR hash:
its first line, @ir 1@, the version of the form. Floats are rendered by
their bits, so that no rendering can round; the gate takes a NaN literal
only as 'canonicalNaN' says.
-}
renderKernel :: Kernel -> Text
renderKernel k =
  T.unlines $
    [ "ir 1"
    , "kernel " <> k.kernelName <> " threads " <> showT k.kernelThreads <> " simd " <> showT k.kernelSimdWidth <> " groups " <> renderExpr k.kernelGroups
    ]
      <> map param k.kernelParams
      <> map buffer k.kernelBuffers
      <> map table k.kernelTables
      <> map tgArray k.kernelTgArrays
      <> ["{"]
      <> concatMap (stmt 1) k.kernelBody
      <> ["}"]
 where
  param p = "  param " <> p.paramName <> " : " <> scalarTypeName p.paramType <> role p.paramRole
  role = \case
    Scalar -> ""
    Dim r -> " dim " <> showT r.dimMin <> ".." <> showT r.dimMax <> " multiple of " <> showT r.dimMultipleOf
  buffer b =
    "  buffer "
      <> b.bufName
      <> " : "
      <> access b.bufAccess
      <> " "
      <> scalarTypeName b.bufType
      <> "["
      <> renderExpr b.bufExtent
      <> "] align "
      <> showT b.bufAlign
  access = \case
    ReadOnly -> "read"
    WriteOnly -> "write"
    ReadWrite -> "read_write"
  table t =
    "  table "
      <> t.tableName
      <> " : "
      <> renderExpr t.tableEntries
      <> " x "
      <> scalarTypeName t.tableType
      <> "["
      <> renderExpr t.tableExtent
      <> "] align "
      <> showT t.tableAlign
  tgArray a = "  threadgroup " <> a.tgName <> " : " <> scalarTypeName a.tgType <> "[" <> showT a.tgCount <> "]"

stmt :: Int -> Stmt -> [Text]
stmt depth = \case
  SLet n t e -> [pad <> "let " <> n <> " : " <> scalarTypeName t <> " = " <> renderExpr e]
  SVar n t e -> [pad <> "var " <> n <> " : " <> scalarTypeName t <> " = " <> renderExpr e]
  SAssign n e -> [pad <> n <> " = " <> renderExpr e]
  SStore b i e -> [pad <> b <> "[" <> renderExpr i <> "] = " <> renderExpr e]
  STgStore a i e -> [pad <> a <> "[" <> renderExpr i <> "] = " <> renderExpr e]
  SLoadPack ns t b i ->
    [ pad
        <> "let ("
        <> T.intercalate ", " ns
        <> ") : "
        <> scalarTypeName t
        <> " = "
        <> b
        <> ".pack"
        <> showT (length ns)
        <> "["
        <> renderExpr i
        <> "]"
    ]
  SStorePack b i es ->
    [pad <> b <> ".pack" <> showT (length es) <> "[" <> renderExpr i <> "] = (" <> T.intercalate ", " (map renderExpr es) <> ")"]
  STgStorePack a i es ->
    [pad <> a <> ".pack" <> showT (length es) <> "[" <> renderExpr i <> "] = (" <> T.intercalate ", " (map renderExpr es) <> ")"]
  SLoadPackVia ns t b e i ->
    [ pad
        <> "let ("
        <> T.intercalate ", " ns
        <> ") : "
        <> scalarTypeName t
        <> " = "
        <> b
        <> "["
        <> renderExpr e
        <> "].pack"
        <> showT (length ns)
        <> "["
        <> renderExpr i
        <> "]"
    ]
  SIf r c th el ->
    [pad <> "if " <> region r <> " " <> renderExpr c <> " {"]
      <> concatMap (stmt (depth + 1)) th
      <> (if null el then [] else [pad <> "} else {"] <> concatMap (stmt (depth + 1)) el)
      <> [pad <> "}"]
  SFor r v start end step body ->
    [ pad
        <> "for "
        <> region r
        <> " "
        <> v
        <> " = "
        <> renderExpr start
        <> " until "
        <> renderExpr end
        <> " step "
        <> renderExpr step
        <> " {"
    ]
      <> concatMap (stmt (depth + 1)) body
      <> [pad <> "}"]
  SBarrier -> [pad <> "barrier"]
  SSimdBarrier -> [pad <> "simd barrier"]
  SSimd n op t e -> [pad <> "let " <> n <> " : " <> scalarTypeName t <> " = " <> simdOp op <> "(" <> renderExpr e <> ")"]
  SFragVar n t -> [pad <> "fragment " <> n <> " : " <> scalarTypeName t <> "[8x8] = 0"]
  SFragLoad n m i st tr ->
    [pad <> n <> " = " <> memory m <> ".fragment" <> (if tr then "_t" else "") <> "[" <> renderExpr i <> ", stride " <> renderExpr st <> "]"]
  SFragMma d a b c -> [pad <> d <> " = " <> a <> " * " <> b <> " + " <> c]
  SFragStore m i st n ->
    [pad <> memory m <> ".fragment[" <> renderExpr i <> ", stride " <> renderExpr st <> "] = " <> n]
 where
  memory = \case
    FDevice b -> b
    FThreadgroup a -> "threadgroup " <> a
  pad = T.replicate (2 * depth) " "
  region = \case
    RUniform -> "uniform"
    RSimdUniform -> "simd_uniform"
    RDivergent -> "divergent"
  simdOp = \case
    SimdSum -> "simd_sum"
    SimdMax -> "simd_max"
    SimdMin -> "simd_min"

{- | Fully parenthesised. On checked IR, where every name is declared once,
the text fixes the tree; on IR the gate refuses, a buffer's and a
threadgroup array's loads print alike, as do their stores and packs.
-}
renderExpr :: Expr -> Text
renderExpr = \case
  ELit l -> literal l
  EVar n -> n
  EParam n -> "$" <> n
  EBuiltin b -> case b of
    ThreadInGroup -> "@thread"
    GroupInGrid -> "@group"
    SimdLane -> "@lane"
    SimdGroup -> "@simd_group"
  EUn op e -> T.toLower (showT op) <> "(" <> renderExpr e <> ")"
  EBin op a b -> "(" <> renderExpr a <> " " <> binOp op <> " " <> renderExpr b <> ")"
  ESelect c a b -> "select(" <> renderExpr c <> ", " <> renderExpr a <> ", " <> renderExpr b <> ")"
  EFma a b c -> "fma(" <> renderExpr a <> ", " <> renderExpr b <> ", " <> renderExpr c <> ")"
  EConvert t e -> scalarTypeName t <> "(" <> renderExpr e <> ")"
  ENarrowInt o t e -> scalarTypeName t <> (case o of Clamp -> "_clamped("; Wrap -> "_wrapped(") <> renderExpr e <> ")"
  ELoad n i -> n <> "[" <> renderExpr i <> "]"
  ELoadVia n e i -> n <> "[" <> renderExpr e <> "][" <> renderExpr i <> "]"
  ETgLoad n i -> n <> "[" <> renderExpr i <> "]"
 where
  binOp = \case
    Add -> "+"
    Sub -> "-"
    Mul -> "*"
    Div -> "/"
    Rem -> "%"
    Min -> "min"
    Max -> "max"
    Eq -> "=="
    Ne -> "!="
    Lt -> "<"
    Le -> "<="
    Gt -> ">"
    Ge -> ">="
    And -> "&&"
    Or -> "||"

literal :: Literal -> Text
literal = \case
  LF32 x -> "f32:0x" <> hex (castFloatToWord32 x)
  LF16 b -> "f16:0x" <> hex (fromIntegral b :: Word32)
  LBF16 b -> "bf16:0x" <> hex (fromIntegral b :: Word32)
  LI32 n -> showT n <> "i"
  LU32 n -> showT n <> "u"
  LI8 n -> showT n <> "i8"
  LU8 n -> showT n <> "u8"
  LBool b -> if b then "true" else "false"
 where
  hex w = T.pack (showHex w "")

showT :: (Show a) => a -> Text
showT = T.pack . show
