-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE RoleAnnotations #-}
{-# LANGUAGE TypeData #-}
{-# LANGUAGE UndecidableInstances #-}

{- | The typed surface: what kernels are written in.

A kernel has three phases, one monad each:

* 'Sig': the interface. Dimensions with their ranges, scalar parameters
  (both in one parameter struct), buffers with their access, alignment and
  extent, the threads per threadgroup and the size of the grid. Buffers
  carry the target, so that the body checks packs against it.
* 'Scope': threadgroup memory, whose total in the kernel's layout
  ("FuryKernel.IR".'FuryKernel.IR.tgLayout') the gate checks.
* 'K': the body, indexed by its 'Region': 'Uniform' where every thread of
  the threadgroup arrives, 'SimdUniform' where every lane of a SIMD group
  does, 'Divergent' under a per-lane branch. Barriers need 'Uniform',
  SIMD-group operations need 'Converged'.

Expressions are 'Exp', a typed wrapper around the IR. Storage types (bf16,
i8, u8) have no arithmetic: a sum of two bf16 values does not compile.
Sharing is explicit ('let_'), and every local is named by the author, so
generated code reads like the kernel.

A lane reads or writes consecutive elements at once as a 'Pack', indexed
in packs: the access is aligned whenever the buffer is, and 'PackOk'
checks the width against the target and the buffer's alignment.

The gate ("FuryKernel.Check") re-checks everything the types promise, on
the IR, and what they don't: the uniformity of conditions, the totals.
-}
module FuryKernel.Build (
  -- * Kernels
  Kernel,
  kernelIR,
  kernelProblems,
  kernelCaps,
  kernel,

  -- * Expressions
  Exp,
  expIR,
  true_,
  false_,
  (.==),
  (./=),
  (.<),
  (.<=),
  (.>),
  (.>=),
  (.&&),
  (.||),
  notE,
  select,
  minE,
  maxE,
  sqrtE,
  rsqrtE,
  expE,
  logE,
  tanhE,
  fmaE,
  quotE,
  remE,
  widen,
  narrow,
  narrowClamped,
  narrowWrapped,
  toF32,
  NumScalar,
  SimdScalar,
  IntScalar,
  FloatScalar,
  Comparable,
  Widen,
  Narrow,
  NarrowInt,

  -- * The interface
  Sig,
  Access (..),
  Buf,
  Tab,
  dim,
  param,
  buffer,
  table,
  threads,
  groups,

  -- * Threadgroup memory
  Scope,
  TgArr,
  tgArray,

  -- * Bodies
  Region (..),
  K,
  Converged,
  AllThreads,
  Meet,
  Readable,
  Writable,
  Var,
  let_,
  var,
  get,
  (.=),
  load,
  loadVia,
  store,
  Pack,
  unpack,
  loadPack,
  loadPackVia,
  storePack,
  tgLoad,
  tgStore,
  barrier,
  simdSum,
  simdMax,
  whenU,
  whenSimd,
  whenLane,
  forU,
  forStep,
  strided,
  threadIndex,
  groupIndex,
  simdLane,
  simdGroupIndex,
  threadCount,
  simdGroupCount,
  simdWidth,
  simdGroups,
  note,
  f32,
) where

import Control.Monad (when)
import Control.Monad.Trans.State.Strict (State, execState, gets, modify', state)
import Data.Foldable (find, toList)
import Data.Int (Int32, Int8)
import Data.Kind (Constraint, Type)
import Data.Maybe (fromMaybe, isJust)
import Data.Proxy (Proxy (..))
import Data.Sequence (Seq)
import Data.Sequence qualified as Seq
import Data.Set (Set)
import Data.Set qualified as Set
import Data.Text qualified as T
import Data.Word (Word32, Word8)
import FuryKernel.Constraint (AlignOk, PackOk, TgArrayOk, ThreadsOk)
import FuryKernel.Finding (Finding (..), Rule (..))
import FuryKernel.IR (Name)
import FuryKernel.IR qualified as IR
import FuryKernel.Numerics (rationalF16)
import FuryKernel.Target (Target, TargetCaps (..), targetCaps)
import FuryKernel.Type (BF16, F16 (..), IsScalar (..), ScalarType (..))
import GHC.Float (castWord32ToFloat)
import GHC.TypeError (ErrorMessage (..), Unsatisfiable)
import GHC.TypeNats (KnownNat, Nat, natVal)
import Numeric.Natural (Natural)

------------------------------------------------------------------------------
-- Building

data BuildState = BuildState
  { names :: Set Name
  , params :: Seq IR.Param
  , buffers :: Seq IR.Buffer
  , tables :: Seq IR.Table
  , tgArrays :: Seq IR.TgArray
  , threadsPer :: Maybe Natural
  , gridGroups :: Maybe IR.Expr
  , stmts :: Seq IR.Stmt
  , caps :: TargetCaps
  , problems :: Seq Finding
  }

emit :: IR.Stmt -> State BuildState ()
emit s = modify' (\b -> b {stmts = b.stmts Seq.|> s})

problem :: Rule -> T.Text -> State BuildState ()
problem r msg = modify' (\b -> b {problems = b.problems Seq.|> Finding r msg})

-- | Claims an interface name exactly as given: the ABI exposes it.
claim :: Name -> State BuildState ()
claim n = do
  taken <- gets (Set.member n . (.names))
  when taken $ problem Names ("The name " <> n <> " is declared twice.")
  modify' (\b -> b {names = Set.insert n b.names})

-- | A local's name: the author's, or with a suffix if already taken.
fresh :: Name -> State BuildState Name
fresh base = state $ \b ->
  let candidates = base : [base <> "_" <> T.pack (show i) | i <- [1 :: Int ..]]
      -- The list is infinite and the set finite: there is always one.
      n = fromMaybe base (find (`Set.notMember` b.names) candidates)
   in (n, b {names = Set.insert n b.names})

-- | Runs a block and returns its statements, leaving the current block as it was.
block :: K s () -> State BuildState [IR.Stmt]
block (K m) = do
  saved <- gets (.stmts)
  modify' (\b -> b {stmts = Seq.empty})
  m
  inner <- gets (.stmts)
  modify' (\b -> b {stmts = saved})
  pure (toList inner)

------------------------------------------------------------------------------
-- Kernels

{- | A kernel for target @t@, before the gate: its IR, what the builder
noticed itself, and the caps of the target it was built for. Its fields
are read by functions only, so no record update changes its target.
-}
type role Kernel nominal

data Kernel t = Kernel IR.Kernel [Finding] TargetCaps

kernelIR :: Kernel t -> IR.Kernel
kernelIR (Kernel ir _ _) = ir

-- | Names declared twice, a missing thread count or grid.
kernelProblems :: Kernel t -> [Finding]
kernelProblems (Kernel _ problems _) = problems

-- | The caps of the target the kernel was built for.
kernelCaps :: Kernel t -> TargetCaps
kernelCaps (Kernel _ _ caps) = caps

kernel :: forall t. (Target t) => Name -> Sig t (Scope t (K Uniform ())) -> Kernel t
kernel name (Sig sig) =
  Kernel
    IR.Kernel
      { IR.kernelName = name
      , IR.kernelThreads = fromMaybe 0 final.threadsPer
      , IR.kernelSimdWidth = (targetCaps @t).capSimdWidth
      , IR.kernelGroups = fromMaybe (IR.ELit (IR.LU32 0)) final.gridGroups
      , IR.kernelParams = toList final.params
      , IR.kernelBuffers = toList final.buffers
      , IR.kernelTables = toList final.tables
      , IR.kernelTgArrays = toList final.tgArrays
      , IR.kernelBody = toList final.stmts
      }
    ( toList final.problems
        <> [Finding Threads "No thread count: declare one with threads." | null final.threadsPer]
        <> [Finding Threads "No grid size: declare one with groups." | null final.gridGroups]
    )
    (targetCaps @t)
 where
  final = execState run initial
  run = do
    claim name
    Scope scope <- sig
    K body <- scope
    body
  initial =
    BuildState
      { names = Set.empty
      , params = Seq.empty
      , buffers = Seq.empty
      , tables = Seq.empty
      , tgArrays = Seq.empty
      , threadsPer = Nothing
      , gridGroups = Nothing
      , stmts = Seq.empty
      , caps = targetCaps @t
      , problems = Seq.empty
      }

------------------------------------------------------------------------------
-- Expressions

-- | An expression of scalar type @a@.
type role Exp nominal

newtype Exp a = Exp IR.Expr

expIR :: Exp a -> IR.Expr
expIR (Exp e) = e

un :: IR.UnOp -> Exp a -> Exp b
un op (Exp a) = Exp (IR.EUn op a)

bin :: IR.BinOp -> Exp a -> Exp a -> Exp b
bin op (Exp a) (Exp b) = Exp (IR.EBin op a b)

true_, false_ :: Exp Bool
true_ = Exp (IR.ELit (IR.LBool True))
false_ = Exp (IR.ELit (IR.LBool False))

-- | Compute types: arithmetic and integer literals.
type NumScalar :: Type -> Constraint
class (Comparable a) => NumScalar a where
  integerLiteral :: Integer -> IR.Literal

instance NumScalar Float where
  integerLiteral = IR.LF32 . fromInteger

instance NumScalar F16 where
  integerLiteral n = let F16 b = rationalF16 (fromInteger n) in IR.LF16 b

instance NumScalar Int32 where
  integerLiteral = IR.LI32 . inRange "i32"

instance NumScalar Word32 where
  integerLiteral = IR.LU32 . inRange "u32"

{- | An integer literal as its type, if it is in the type's range: i32's
least, a negated literal past its greatest, is @-2147483647 - 1@ (or
@-2147483648@ under NegativeLiterals).
-}
inRange :: forall a. (Bounded a, Integral a) => String -> Integer -> a
inRange name n
  | n < toInteger (minBound @a) || n > toInteger (maxBound @a) =
      error ("FuryKernel.Build: the literal " <> show n <> " is outside " <> name <> "'s range, " <> show (toInteger (minBound @a)) <> " to " <> show (toInteger (maxBound @a)) <> ".")
  | otherwise = fromInteger n

instance
  ( Unsatisfiable
      ( 'Text "bf16 is a storage type, with no arithmetic: widen it to Float,"
          ':$$: 'Text "compute, and narrow the result back."
      )
  ) =>
  NumScalar BF16

instance
  ( Unsatisfiable
      ('Text "i8 is a storage type, with no arithmetic: widen it to Int32 first.")
  ) =>
  NumScalar Int8

instance
  ( Unsatisfiable
      ('Text "u8 is a storage type, with no arithmetic: widen it to Word32 first.")
  ) =>
  NumScalar Word8

instance
  (Unsatisfiable ('Text "Bool has no arithmetic: use .&&, .|| and notE.")) =>
  NumScalar Bool

instance (NumScalar a) => Num (Exp a) where
  (+) = bin IR.Add
  (-) = bin IR.Sub
  (*) = bin IR.Mul
  negate = un IR.Neg
  abs = un IR.Abs
  signum x = select (x .> 0) 1 (select (x .< 0) (-1) 0)
  fromInteger = Exp . IR.ELit . integerLiteral @a

-- | The floating compute types.
type FloatScalar :: Type -> Constraint
class (NumScalar a) => FloatScalar a where
  rationalLiteral :: Rational -> IR.Literal

instance FloatScalar Float where
  rationalLiteral = IR.LF32 . fromRational

-- | A binary16 literal is its value rounded once, to nearest, ties to even.
instance FloatScalar F16 where
  rationalLiteral r = let F16 b = rationalF16 r in IR.LF16 b

instance
  ( Unsatisfiable
      ( 'Text "bf16 is a storage type, with no arithmetic: widen it to Float,"
          ':$$: 'Text "compute, and narrow the result back."
      )
  ) =>
  FloatScalar BF16

instance
  ( Unsatisfiable
      ('Text "i8 is a storage type, with no arithmetic: widen it to Int32 first.")
  ) =>
  FloatScalar Int8

instance
  ( Unsatisfiable
      ('Text "u8 is a storage type, with no arithmetic: widen it to Word32 first.")
  ) =>
  FloatScalar Word8

instance
  (Unsatisfiable ('Text "Bool has no arithmetic: use .&&, .|| and notE.")) =>
  FloatScalar Bool

instance
  ( Unsatisfiable
      ('Text "i32 is an integer type, with no floating-point operations.")
  ) =>
  FloatScalar Int32

instance
  ( Unsatisfiable
      ('Text "u32 is an integer type, with no floating-point operations: convert it with toF32 first.")
  ) =>
  FloatScalar Word32

instance (FloatScalar a) => Fractional (Exp a) where
  (/) = bin IR.Div
  fromRational = Exp . IR.ELit . rationalLiteral @a

-- | Types with an order: the compute types.
type Comparable :: Type -> Constraint
class (IsScalar a) => Comparable a

instance Comparable Float

instance Comparable F16

instance Comparable Int32

instance Comparable Word32

instance
  ( Unsatisfiable
      ('Text "bf16 is a storage type, with no comparisons: widen it to Float first.")
  ) =>
  Comparable BF16

instance
  ( Unsatisfiable
      ('Text "i8 is a storage type, with no comparisons: widen it to Int32 first.")
  ) =>
  Comparable Int8

instance
  ( Unsatisfiable
      ('Text "u8 is a storage type, with no comparisons: widen it to Word32 first.")
  ) =>
  Comparable Word8

instance
  ( Unsatisfiable
      ('Text "Bool has no order: combine bools with .&&, .|| and notE.")
  ) =>
  Comparable Bool

infix 4 .==, ./=, .<, .<=, .>, .>=

(.==), (./=), (.<), (.<=), (.>), (.>=) :: (Comparable a) => Exp a -> Exp a -> Exp Bool
(.==) = bin IR.Eq
(./=) = bin IR.Ne
(.<) = bin IR.Lt
(.<=) = bin IR.Le
(.>) = bin IR.Gt
(.>=) = bin IR.Ge

infixr 3 .&&

infixr 2 .||

(.&&), (.||) :: Exp Bool -> Exp Bool -> Exp Bool
(.&&) = bin IR.And
(.||) = bin IR.Or

notE :: Exp Bool -> Exp Bool
notE = un IR.Not

-- | Per-element choice. Both sides are evaluated.
select :: Exp Bool -> Exp a -> Exp a -> Exp a
select (Exp c) (Exp a) (Exp b) = Exp (IR.ESelect c a b)

-- | The lesser and the greater, of floats as IEEE 754-2019's minimumNumber and maximumNumber ('IR.Min').
minE, maxE :: (Comparable a) => Exp a -> Exp a -> Exp a
minE = bin IR.Min
maxE = bin IR.Max

sqrtE, rsqrtE, expE, logE, tanhE :: (FloatScalar a) => Exp a -> Exp a
sqrtE = un IR.Sqrt
rsqrtE = un IR.Rsqrt
expE = un IR.Exp
logE = un IR.Log
tanhE = un IR.Tanh

{- | @fmaE a b c@ is @a * b + c@ rounded once ('IR.EFma'). Targets fuse
nothing else: @a * b + c@ written so rounds twice.
-}
fmaE :: (FloatScalar a) => Exp a -> Exp a -> Exp a -> Exp a
fmaE (Exp a) (Exp b) (Exp c) = Exp (IR.EFma a b c)

-- | The integer compute types.
type IntScalar :: Type -> Constraint
class (NumScalar a) => IntScalar a

instance IntScalar Int32

instance IntScalar Word32

{- | Integer division, rounding toward zero, and its remainder. A zero
divisor is undefined on the targets, and an error in the interpreter.
-}
quotE, remE :: (IntScalar a) => Exp a -> Exp a -> Exp a
quotE = bin IR.Div
remE = bin IR.Rem

-- | Exact conversions to a wider type.
type Widen :: Type -> Type -> Constraint
class (IsScalar a, IsScalar b) => Widen a b

instance Widen BF16 Float

instance Widen F16 Float

instance Widen Int8 Int32

instance Widen Word8 Word32

-- | Conversions to a narrower type, rounding to nearest with ties to even.
type Narrow :: Type -> Type -> Constraint
class (IsScalar a, IsScalar b) => Narrow a b

instance Narrow Float BF16

instance Narrow Float F16

instance
  ( Unsatisfiable
      ( 'Text "An i32 narrows to i8 with narrowClamped or narrowWrapped,"
          ':$$: 'Text "which say what a value out of its range becomes."
      )
  ) =>
  Narrow Int32 Int8

instance
  ( Unsatisfiable
      ( 'Text "A u32 narrows to u8 with narrowClamped or narrowWrapped,"
          ':$$: 'Text "which say what a value out of its range becomes."
      )
  ) =>
  Narrow Word32 Word8

-- | Integer narrowings, each saying what a value out of the range becomes.
type NarrowInt :: Type -> Type -> Constraint
class (IsScalar a, IsScalar b) => NarrowInt a b

instance NarrowInt Int32 Int8

instance NarrowInt Word32 Word8

-- | @widen \@Float x@.
widen :: forall b a. (Widen a b) => Exp a -> Exp b
widen (Exp e) = Exp (IR.EConvert (scalarType @b) e)

-- | @narrow \@BF16 x@.
narrow :: forall b a. (Narrow a b) => Exp a -> Exp b
narrow (Exp e) = Exp (IR.EConvert (scalarType @b) e)

-- | @narrowClamped \@Int8 x@: @x@, or the narrower range's nearer end.
narrowClamped :: forall b a. (NarrowInt a b) => Exp a -> Exp b
narrowClamped (Exp e) = Exp (IR.ENarrowInt IR.Clamp (scalarType @b) e)

-- | @narrowWrapped \@Int8 x@: @x@'s low bits, two's complement.
narrowWrapped :: forall b a. (NarrowInt a b) => Exp a -> Exp b
narrowWrapped (Exp e) = Exp (IR.ENarrowInt IR.Wrap (scalarType @b) e)

-- | An integer as a float, rounding to nearest.
toF32 :: Exp Word32 -> Exp Float
toF32 (Exp e) = Exp (IR.EConvert TF32 e)

------------------------------------------------------------------------------
-- The interface

-- | The interface of a kernel for target @t@.
type role Sig nominal nominal

newtype Sig t a = Sig (State BuildState a)
  deriving newtype (Functor, Applicative, Monad)

type data Access = ReadOnly | WriteOnly | ReadWrite

type KnownAccess :: Access -> Constraint
class KnownAccess acc where
  accessOf :: IR.Access

instance KnownAccess ReadOnly where
  accessOf = IR.ReadOnly

instance KnownAccess WriteOnly where
  accessOf = IR.WriteOnly

instance KnownAccess ReadWrite where
  accessOf = IR.ReadWrite

{- | A device buffer of @a@ of a kernel for target @t@, with access @acc@,
whose start is aligned to @align@ bytes.
-}
type Buf :: Type -> Access -> Nat -> Type -> Type

type role Buf nominal nominal nominal nominal

newtype Buf t acc align a = Buf Name

{- | A table of device addresses for target @t@: each entry the start of an
array of @a@ aligned to @align@ bytes. Kernels only read through tables.
-}
type Tab :: Type -> Nat -> Type -> Type

type role Tab nominal nominal nominal

newtype Tab t align a = Tab Name

-- | A u32 dimension of the launch, in the parameter struct.
dim :: Name -> IR.DimRange -> Sig t (Exp Word32)
dim n r = Sig $ do
  claim n
  modify' (\b -> b {params = b.params Seq.|> IR.Param n TU32 (IR.Dim r)})
  pure (Exp (IR.EParam n))

-- | A scalar parameter, in the parameter struct.
param :: forall a t. (IsScalar a) => Name -> Sig t (Exp a)
param n = Sig $ do
  claim n
  modify' (\b -> b {params = b.params Seq.|> IR.Param n (scalarType @a) IR.Scalar})
  pure (Exp (IR.EParam n))

{- | @buffer \@ReadOnly \@16 \@BF16 "x" extent@: a buffer of @extent@
elements, aligned to a power of two of at least 16 bytes and the element
size ('AlignOk').
-}
buffer ::
  forall acc align a t.
  (KnownAccess acc, KnownNat align, IsScalar a, AlignOk a align) =>
  Name ->
  Exp Word32 ->
  Sig t (Buf t acc align a)
buffer n (Exp extent) = Sig $ do
  claim n
  let b' = IR.Buffer n (accessOf @acc) (scalarType @a) (natVal (Proxy @align)) extent
  modify' (\b -> b {buffers = b.buffers Seq.|> b'})
  pure (Buf n)

{- | @table \@16 \@BF16 "experts" entries extent@: a table of @entries@
addresses, each of an array of which the kernel may touch @extent@
elements, aligned as a buffer is ('AlignOk'). Tables bind after the
buffers.
-}
table ::
  forall align a t.
  (KnownNat align, IsScalar a, AlignOk a align) =>
  Name ->
  Exp Word32 ->
  Exp Word32 ->
  Sig t (Tab t align a)
table n (Exp entries) (Exp extent) = Sig $ do
  claim n
  let tb = IR.Table n (scalarType @a) (natVal (Proxy @align)) entries extent
  modify' (\b -> b {tables = b.tables Seq.|> tb})
  pure (Tab n)

-- | @threads \@256@: threads per threadgroup, checked against @t@ statically.
threads :: forall n t. (KnownNat n, ThreadsOk t n) => Sig t ()
threads = Sig $ do
  already <- gets (.threadsPer)
  when (isJust already) $ problem Threads "The thread count is declared twice."
  modify' (\b -> b {threadsPer = Just (natVal (Proxy @n))})

-- | Threadgroups in the grid.
groups :: Exp Word32 -> Sig t ()
groups (Exp e) = Sig $ do
  already <- gets (.gridGroups)
  when (isJust already) $ problem Threads "The grid size is declared twice."
  modify' (\b -> b {gridGroups = Just e})

------------------------------------------------------------------------------
-- Threadgroup memory

-- | Threadgroup memory of a kernel for target @t@.
type role Scope nominal nominal

newtype Scope t a = Scope (State BuildState a)
  deriving newtype (Functor, Applicative, Monad)

-- | A threadgroup array of @n@ elements of @a@.
type TgArr :: Nat -> Type -> Type

type role TgArr nominal nominal

newtype TgArr n a = TgArr Name

-- | @tgArray \@8 \@Float "partial"@, checked against @t@ statically.
tgArray :: forall n a t. (KnownNat n, IsScalar a, TgArrayOk t n a) => Name -> Scope t (TgArr n a)
tgArray n = Scope $ do
  claim n
  let a = IR.TgArray n (scalarType @a) (natVal (Proxy @n))
  modify' (\b -> b {tgArrays = b.tgArrays Seq.|> a})
  pure (TgArr n)

------------------------------------------------------------------------------
-- Bodies

-- | How uniform control is where the code runs.
type data Region = Uniform | SimdUniform | Divergent

-- | The less uniform of two regions.
type Meet :: Region -> Region -> Region
type family Meet r s where
  Meet Divergent _ = Divergent
  Meet _ Divergent = Divergent
  Meet SimdUniform _ = SimdUniform
  Meet _ SimdUniform = SimdUniform
  Meet Uniform Uniform = Uniform

-- | Code in region @r@ of a kernel body.
type K :: Region -> Type -> Type

type role K nominal nominal

newtype K r a = K (State BuildState a)
  deriving newtype (Functor, Applicative, Monad)

-- | Regions where every lane of a SIMD group is active.
type Converged :: Region -> Constraint
class Converged r

instance Converged Uniform

instance Converged SimdUniform

instance
  ( Unsatisfiable
      ( 'Text "SIMD-group operations need every lane of the group, and this code"
          ':$$: 'Text "runs under a per-lane branch or loop (whenLane, strided, forStep)."
          ':$$: 'Text "Move the operation out of it, or choose per lane with select."
      )
  ) =>
  Converged Divergent

type Readable :: Access -> Constraint
class Readable acc

instance Readable ReadOnly

instance Readable ReadWrite

instance
  (Unsatisfiable ('Text "This buffer is write-only: declare it ReadWrite to load from it.")) =>
  Readable WriteOnly

type Writable :: Access -> Constraint
class Writable acc

instance Writable WriteOnly

instance Writable ReadWrite

instance
  (Unsatisfiable ('Text "This buffer is read-only: declare it ReadWrite to store to it.")) =>
  Writable ReadOnly

-- | A mutable local.
type role Var nominal

newtype Var a = Var Name

-- | Binds an expression to a named local, so that it is computed once.
let_ :: forall a r. (IsScalar a) => Name -> Exp a -> K r (Exp a)
let_ n (Exp e) = K $ do
  n' <- fresh n
  emit (IR.SLet n' (scalarType @a) e)
  pure (Exp (IR.EVar n'))

var :: forall a r. (IsScalar a) => Name -> Exp a -> K r (Var a)
var n (Exp e) = K $ do
  n' <- fresh n
  emit (IR.SVar n' (scalarType @a) e)
  pure (Var n')

-- | The variable's value where the expression is used.
get :: Var a -> Exp a
get (Var n) = Exp (IR.EVar n)

infix 1 .=

(.=) :: Var a -> Exp a -> K r ()
Var n .= Exp e = K (emit (IR.SAssign n e))

load :: (Readable acc) => Buf t acc align a -> Exp Word32 -> Exp a
load (Buf n) (Exp i) = Exp (IR.ELoad n i)

-- | Element @i@ of the array that entry @e@ of the table points at.
loadVia :: Tab t align a -> Exp Word32 -> Exp Word32 -> Exp a
loadVia (Tab n) (Exp e) (Exp i) = Exp (IR.ELoadVia n e i)

store :: (Writable acc) => Buf t acc align a -> Exp Word32 -> Exp a -> K r ()
store (Buf n) (Exp i) (Exp e) = K (emit (IR.SStore n i e))

-- | @n@ consecutive elements of @a@, which one lane loads at once.
type Pack :: Nat -> Type -> Type

type role Pack nominal nominal

newtype Pack n a = Pack [Exp a]

-- | The elements, in order.
unpack :: Pack n a -> [Exp a]
unpack (Pack es) = es

{- | @loadPack \@8 "wv" w j@: elements @8 j@ to @8 j + 7@ of @w@, read at
once and bound to the locals @wv_0@ to @wv_7@.
-}
loadPack ::
  forall n t acc align a r.
  (Readable acc, KnownNat n, IsScalar a, PackOk t n a align) =>
  Name ->
  Buf t acc align a ->
  Exp Word32 ->
  K r (Pack n a)
loadPack base (Buf b) (Exp j) = K $ do
  names <- traverse (\i -> fresh (base <> "_" <> T.pack (show i))) [0 .. natVal (Proxy @n) - 1]
  emit (IR.SLoadPack names (scalarType @a) b j)
  pure (Pack (map (Exp . IR.EVar) names))

{- | @loadPackVia \@8 "wv" w e j@: elements @8 j@ to @8 j + 7@ of the array
that entry @e@ of the table points at, read at once.
-}
loadPackVia ::
  forall n t align a r.
  (KnownNat n, IsScalar a, PackOk t n a align) =>
  Name ->
  Tab t align a ->
  Exp Word32 ->
  Exp Word32 ->
  K r (Pack n a)
loadPackVia base (Tab b) (Exp e) (Exp j) = K $ do
  names <- traverse (\i -> fresh (base <> "_" <> T.pack (show i))) [0 .. natVal (Proxy @n) - 1]
  emit (IR.SLoadPackVia names (scalarType @a) b e j)
  pure (Pack (map (Exp . IR.EVar) names))

{- | @storePack \@4 y j [a, b, c, d]@: elements @4 j@ to @4 j + 3@ of @y@,
written at once.
-}
storePack ::
  forall n t acc align a r.
  (Writable acc, KnownNat n, PackOk t n a align) =>
  Buf t acc align a ->
  Exp Word32 ->
  [Exp a] ->
  K r ()
storePack (Buf b) (Exp j) es = K $ do
  let n = natVal (Proxy @n)
  when (fromIntegral (length es) /= n) $
    problem Typing ("A pack of " <> T.pack (show n) <> " stored to " <> b <> " with " <> T.pack (show (length es)) <> " values.")
  emit (IR.SStorePack b j (map expIR es))

tgLoad :: TgArr n a -> Exp Word32 -> Exp a
tgLoad (TgArr n) (Exp i) = Exp (IR.ETgLoad n i)

tgStore :: TgArr n a -> Exp Word32 -> Exp a -> K r ()
tgStore (TgArr n) (Exp i) (Exp e) = K (emit (IR.STgStore n i e))

-- | Every thread of the threadgroup waits here, so only in 'Uniform' code.
barrier :: (AllThreads r) => K r ()
barrier = K (emit IR.SBarrier)

-- | Regions every thread of the threadgroup reaches.
type AllThreads :: Region -> Constraint
class AllThreads r

instance AllThreads Uniform

instance
  ( Unsatisfiable
      ( 'Text "A barrier needs every thread of the threadgroup, and this code runs"
          ':$$: 'Text "under a branch that is only uniform per SIMD group (whenSimd)."
          ':$$: 'Text "Move the barrier out of the branch."
      )
  ) =>
  AllThreads SimdUniform

instance
  ( Unsatisfiable
      ( 'Text "A barrier needs every thread of the threadgroup, and this code runs"
          ':$$: 'Text "under a per-lane branch or loop (whenLane, strided, forStep)."
          ':$$: 'Text "Move the barrier out of it."
      )
  ) =>
  AllThreads Divergent

-- | The types SIMD groups reduce.
type SimdScalar :: Type -> Constraint
class (Comparable a) => SimdScalar a

instance SimdScalar Float

instance SimdScalar F16

instance SimdScalar Int32

instance SimdScalar Word32

instance
  ( Unsatisfiable
      ('Text "SIMD groups reduce f32, f16, i32 and u32: widen a storage type first.")
  ) =>
  SimdScalar BF16

instance
  ( Unsatisfiable
      ('Text "SIMD groups reduce f32, f16, i32 and u32: widen a storage type first.")
  ) =>
  SimdScalar Int8

instance
  ( Unsatisfiable
      ('Text "SIMD groups reduce f32, f16, i32 and u32: widen a storage type first.")
  ) =>
  SimdScalar Word8

instance
  ( Unsatisfiable
      ('Text "SIMD groups reduce f32, f16, i32 and u32, and no bools.")
  ) =>
  SimdScalar Bool

-- | The sum over the lanes of the SIMD group, in every lane.
simdSum :: forall a r. (Converged r, SimdScalar a) => Name -> Exp a -> K r (Exp a)
simdSum = simdOp IR.SimdSum

-- | The maximum over the lanes of the SIMD group, in every lane.
simdMax :: forall a r. (Converged r, SimdScalar a) => Name -> Exp a -> K r (Exp a)
simdMax = simdOp IR.SimdMax

simdOp :: forall a r. (SimdScalar a) => IR.SimdOp -> Name -> Exp a -> K r (Exp a)
simdOp op n (Exp e) = K $ do
  n' <- fresh n
  emit (IR.SSimd n' op (scalarType @a) e)
  pure (Exp (IR.EVar n'))

-- | A branch on a condition uniform across the threadgroup.
whenU :: Exp Bool -> K r () -> K r ()
whenU (Exp c) body = K $ do
  inner <- block body
  emit (IR.SIf IR.RUniform c inner [])

-- | A branch on a condition uniform across each SIMD group.
whenSimd :: Exp Bool -> K (Meet r SimdUniform) () -> K r ()
whenSimd (Exp c) body = K $ do
  inner <- block body
  emit (IR.SIf IR.RSimdUniform c inner [])

-- | A branch per lane.
whenLane :: Exp Bool -> K Divergent () -> K r ()
whenLane (Exp c) body = K $ do
  inner <- block body
  emit (IR.SIf IR.RDivergent c inner [])

-- | @for (i = lo; i < hi; i++)@ with bounds uniform across the threadgroup.
forU :: Name -> Exp Word32 -> Exp Word32 -> (Exp Word32 -> K r ()) -> K r ()
forU n (Exp lo) (Exp hi) body = K $ do
  v <- fresh n
  inner <- block (body (Exp (IR.EVar v)))
  emit (IR.SFor IR.RUniform v lo hi (IR.ELit (IR.LU32 1)) inner)

{- | @for (i = start; i < end; i += step)@ with any bounds: lanes may leave
at different iterations, so the body is 'Divergent'.
-}
forStep :: Name -> Exp Word32 -> Exp Word32 -> Exp Word32 -> (Exp Word32 -> K Divergent ()) -> K r ()
forStep n (Exp start) (Exp end) (Exp step) body = K $ do
  v <- fresh n
  inner <- block (body (Exp (IR.EVar v)))
  emit (IR.SFor IR.RDivergent v start end step inner)

{- | The threads of the threadgroup take turns over @[0, n)@: thread @i@
visits @i@, @i + threads@, and so on. Lanes may stop at different
iterations, so the body is 'Divergent'.
-}
strided :: Name -> Exp Word32 -> (Exp Word32 -> K Divergent ()) -> K r ()
strided n (Exp end) body = K $ do
  v <- fresh n
  step <- gets (fromMaybe 0 . (.threadsPer))
  inner <- block (body (Exp (IR.EVar v)))
  emit (IR.SFor IR.RDivergent v (IR.EBuiltin IR.ThreadInGroup) end (IR.ELit (IR.LU32 (fromIntegral step))) inner)

threadIndex, groupIndex, simdLane, simdGroupIndex :: Exp Word32
threadIndex = Exp (IR.EBuiltin IR.ThreadInGroup)
groupIndex = Exp (IR.EBuiltin IR.GroupInGrid)
simdLane = Exp (IR.EBuiltin IR.SimdLane)
simdGroupIndex = Exp (IR.EBuiltin IR.SimdGroup)

-- | Threads per threadgroup, as a literal.
threadCount :: K r (Exp Word32)
threadCount = K (gets (Exp . IR.ELit . IR.LU32 . fromIntegral . fromMaybe 0 . (.threadsPer)))

-- | SIMD groups per threadgroup, as a literal.
simdGroupCount :: K r (Exp Word32)
simdGroupCount = K $ do
  n <- gets (fromMaybe 0 . (.threadsPer))
  w <- gets (.caps.capSimdWidth)
  pure (Exp (IR.ELit (IR.LU32 (fromIntegral (n `div` max 1 w)))))

-- | Lanes per SIMD group on the target, as a literal.
simdWidth :: K r (Exp Word32)
simdWidth = K (gets (Exp . IR.ELit . IR.LU32 . fromIntegral . (.caps.capSimdWidth)))

{- | A problem the gate reports, for libraries that check their own
preconditions while they build, by a rule of "FuryKernel.Finding".
-}
note :: Rule -> T.Text -> K r ()
note r msg = K (problem r msg)

-- | SIMD groups per threadgroup, as a number, for checks while building.
simdGroups :: K r Natural
simdGroups = K $ do
  n <- gets (fromMaybe 0 . (.threadsPer))
  w <- gets (.caps.capSimdWidth)
  pure (n `div` max 1 w)

{- | A binary32 literal from its value, infinities and NaNs included: every
NaN is the canonical one, the quiet NaN 0x7fc00000.
-}
f32 :: Float -> Exp Float
f32 x = Exp (IR.ELit (IR.LF32 (if isNaN x then castWord32ToFloat 0x7fc00000 else x)))
