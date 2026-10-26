-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE AllowAmbiguousTypes #-}

{- | A SIMT interpreter for checked kernels: the reference that kernels'
specifications and, through them, the targets are tested against.

Threadgroups run one after another. Within one, every thread executes the
same statement together, on a vector of lanes with an active mask: a branch
splits the mask, a loop keeps the lanes whose condition still holds, and
the lanes reconverge after the construct, as on the GPU. SIMD-group
reductions combine each group's lanes pairwise, in the IR's order.

What the model makes explicit:

* arithmetic as the IR defines it ("FuryKernel.IR"'s 'BinOp' and 'UnOp'):
  binary32 with IEEE rounding, binary16 rounded after every operation
  (exact, since binary32 holds every binary16 product and quotient to
  within one rounding), narrowing to nearest with ties to even, minimum
  and maximum as IEEE 754's minimumNumber and maximumNumber, integer
  addition, subtraction, multiplication and negation wrapping around;
* the stores of one statement land in an order drawn from the seed, so a
  kernel whose lanes race to one address gives seed-dependent results; a
  lane's pack lands whole;
* errors instead of undefined behaviour: accesses past what the kernel
  declares (a buffer's extent, a threadgroup array's size), however much
  the launch gives, dimensions outside their declared range, reads of
  threadgroup memory never written, integer division by zero and the one
  quotient that overflows, loops that never end, and barriers or
  SIMD-group operations reached by part of a group.

Between barriers, the lanes run in lockstep here, so a read that races
with another lane's write still sees it; the hazard check of a later stage
rejects such kernels instead.
-}
module FuryKernel.Interp (
  Value (..),
  valueType,
  Launch,
  launch,
  withSeed,
  RunError (..),
  run,
) where

import Control.Monad (forM, forM_, unless, when)
import Control.Monad.ST (ST, runST)
import Control.Monad.Trans.Class (lift)
import Control.Monad.Trans.Except (ExceptT, runExceptT, throwE)
import Data.Array (Array, elems)
import Data.Array.ST (STArray, getBounds, newArray, newListArray, readArray, writeArray)
import Data.Array.Unsafe (unsafeFreeze)
import Data.Bits (shiftR, xor, (.&.), (.|.))
import Data.Int (Int32, Int8)
import Data.IntSet qualified as IntSet
import Data.List (sortOn, transpose)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.STRef (STRef, modifySTRef', newSTRef, readSTRef)
import Data.Text (Text)
import Data.Text qualified as T
import Data.Word (Word32, Word64, Word8)
import FuryKernel.Check (Checked, checkedIR)
import FuryKernel.IR hiding (Kernel (..))
import FuryKernel.IR qualified as IR
import FuryKernel.Numerics (narrowBF16, narrowF16, narrowF16Double, rationalF16, widenBF16, widenF16)
import FuryKernel.Target (Target, TargetCaps (..), targetCaps)
import FuryKernel.Type (BF16 (..), F16 (..), ScalarType (..))
import GHC.Float (castFloatToWord32, castWord32ToFloat, double2Float, float2Double)

data Value
  = VF32 Float
  | VF16 F16
  | VBF16 BF16
  | VI32 Int32
  | VU32 Word32
  | VI8 Int8
  | VU8 Word8
  | VBool Bool
  deriving stock (Eq, Show)

valueType :: Value -> ScalarType
valueType = \case
  VF32 _ -> TF32
  VF16 _ -> TF16
  VBF16 _ -> TBF16
  VI32 _ -> TI32
  VU32 _ -> TU32
  VI8 _ -> TI8
  VU8 _ -> TU8
  VBool _ -> TBool

{- | What a launch is given: the parameter struct and every buffer. Made by
'launch' and the setters after it, which what a launch gains later leaves
as they are.
-}
data Launch = Launch
  { launchParams :: Map Name Value
  , launchBuffers :: Map Name [Value]
  , launchSeed :: Word64
  }

-- | A launch of these parameters and buffers: seed 0.
launch :: Map Name Value -> Map Name [Value] -> Launch
launch params buffers = Launch params buffers 0

-- | The launch, the order of its stores drawn from this seed.
withSeed :: Word64 -> Launch -> Launch
withSeed seed l = l {launchSeed = seed}

{- | What stops a run. More may come: match those you expect and leave the
rest to a catch-all.
-}
data RunError
  = MissingParam Name
  | MissingBuffer Name
  | -- | A parameter or buffer element of the wrong type.
    WrongType Name ScalarType ScalarType
  | -- | A buffer shorter than its declared extent: name, extent, length.
    ShortBuffer Name Int Int
  | {- | An access past what the kernel declares: name, index, and the
    buffer's extent or the threadgroup array's size.
    -}
    OutOfBounds Name Integer Int
  | -- | A dimension outside its declared range or off its multiple: name, value, range.
    DimOutOfRange Name Word32 DimRange
  | UninitializedRead Name Int
  | DivideByZero
  | -- | The one signed quotient that overflows: the minimum by -1.
    IntegerOverflow
  | {- | A loop that never ends: a lane's variable stood still (a step of 0)
    or wrapped past u32's greatest, or the loop ran 2^24 iterations.
    -}
    NonTermination Name
  | DivergentBarrier
  | DivergentSimd Name
  | Internal Text
  deriving stock (Eq, Show)

-- | Runs a kernel; returns every buffer as it is afterwards.
run :: forall t. (Target t) => Checked t -> Launch -> Either RunError (Map Name [Value])
run checked supplied = runST (runExceptT (execute (targetCaps @t) (checkedIR checked) supplied))

type M s = ExceptT RunError (ST s)

data Env s = Env
  { envCaps :: TargetCaps
  , envKernel :: IR.Kernel
  , envParams :: Map Name Value
  , envBuffers :: Map Name (STArray s Int Value)
  , envSeed :: Word64
  , envStoreCount :: STRef s Word64
  }

-- | Per threadgroup: the locals as lane vectors, the threadgroup arrays.
data Group s = Group
  { groupIndex :: Word32
  , groupLocals :: STRef s (Map Name (STArray s Int Value))
  , groupTg :: Map Name (STArray s Int (Maybe Value))
  }

execute :: forall s. TargetCaps -> IR.Kernel -> Launch -> M s (Map Name [Value])
execute caps k supplied = do
  params <- forM k.kernelParams $ \p -> case Map.lookup p.paramName supplied.launchParams of
    Nothing -> throwE (MissingParam p.paramName)
    Just v
      | valueType v /= p.paramType -> throwE (WrongType p.paramName p.paramType (valueType v))
      | otherwise -> case (p.paramRole, v) of
          (Dim r, VU32 n)
            | toInteger n < toInteger r.dimMin
                || toInteger n > toInteger r.dimMax
                || (r.dimMultipleOf > 0 && toInteger n `mod` toInteger r.dimMultipleOf /= 0) ->
                throwE (DimOutOfRange p.paramName n r)
          _ -> pure (p.paramName, v)
  let paramMap = Map.fromList params
  -- Each buffer as far as its declared extent, the rest of what the launch
  -- gives kept aside, untouched.
  buffers <- forM k.kernelBuffers $ \b -> case Map.lookup b.bufName supplied.launchBuffers of
    Nothing -> throwE (MissingBuffer b.bufName)
    Just vs -> do
      forM_ vs $ \v -> when (valueType v /= b.bufType) $ throwE (WrongType b.bufName b.bufType (valueType v))
      extent <- fromIntegral <$> hostU32 paramMap b.bufExtent
      when (length vs < extent) $ throwE (ShortBuffer b.bufName extent (length vs))
      arr <- lift (newListArray (0, extent - 1) (take extent vs))
      pure (b.bufName, (arr, drop extent vs))
  counter <- lift (newSTRef 0)
  let env = Env caps k paramMap (Map.fromList [(n, arr) | (n, (arr, _)) <- buffers]) supplied.launchSeed counter
  groups <- hostU32 paramMap k.kernelGroups
  forM_ (if groups == 0 then [] else [0 .. groups - 1]) $ \g -> do
    locals <- lift (newSTRef Map.empty)
    tg <- forM k.kernelTgArrays $ \a -> do
      arr <- lift (newArray (0, fromIntegral a.tgCount - 1) Nothing)
      pure (a.tgName, arr)
    let grp = Group g locals (Map.fromList tg)
    block env grp (allLanes k) k.kernelBody
  forM (Map.fromList buffers) $ \(arr, rest) -> (<> rest) . elems <$> lift (freeze' arr)
 where
  freeze' :: STArray s Int Value -> ST s (Array Int Value)
  freeze' = unsafeFreeze

allLanes :: IR.Kernel -> [Int]
allLanes k = [0 .. fromIntegral k.kernelThreads - 1]

{- | Launch sizes: u32 expressions over the parameters, as the host computes
them. 'execute' checks the dimensions against their ranges first, and the
gate keeps every step of a launch size in u32 for those
("FuryKernel.Check.Launch"), so nothing here wraps, and the values agree
with the u64 arithmetic of "FuryKernel.Emit.Rust".
-}
hostU32 :: Map Name Value -> Expr -> M s Word32
hostU32 params = \case
  ELit (LU32 n) -> pure n
  EParam n -> case Map.lookup n params of
    Just (VU32 v) -> pure v
    _ -> throwE (MissingParam n)
  EBin op a b -> do
    x <- hostU32 params a
    y <- hostU32 params b
    case op of
      Add -> pure (x + y)
      Sub -> pure (x - y)
      Mul -> pure (x * y)
      Div -> if y == 0 then throwE DivideByZero else pure (x `div` y)
      Rem -> if y == 0 then throwE DivideByZero else pure (x `mod` y)
      _ -> throwE (Internal "launch size operator")
  _ -> throwE (Internal "launch size expression")

block :: Env s -> Group s -> [Int] -> [Stmt] -> M s ()
block env grp active = mapM_ (stmt env grp active)

stmt :: forall s. Env s -> Group s -> [Int] -> Stmt -> M s ()
stmt env grp active s = unless (null active) $ case s of
  SLet n _ e -> bindLocal n e
  SVar n _ e -> bindLocal n e
  SAssign n e -> do
    vs <- forM active $ \l -> (l,) <$> eval env grp l e
    arr <- local n
    forM_ vs $ \(l, v) -> lift (writeArray arr l $! v)
  SStore b i e -> do
    writes <- forM active $ \l -> (,) <$> (eval env grp l i >>= index) <*> eval env grp l e
    arr <- maybe (throwE (MissingBuffer b)) pure (Map.lookup b env.envBuffers)
    ordered <- shuffle writes
    forM_ ordered $ \(ix, v) -> do
      checkBounds b arr ix
      lift (writeArray arr (fromIntegral ix) $! v)
  STgStore a i e -> do
    writes <- forM active $ \l -> (,) <$> (eval env grp l i >>= index) <*> eval env grp l e
    arr <- maybe (throwE (Internal ("threadgroup array " <> a))) pure (Map.lookup a grp.groupTg)
    ordered <- shuffle writes
    forM_ ordered $ \(ix, v) -> do
      checkBounds a arr ix
      lift (writeArray arr (fromIntegral ix) $! Just $! v)
  SLoadPack ns _ b i -> do
    let width = toInteger (length ns)
    arr <- maybe (throwE (MissingBuffer b)) pure (Map.lookup b env.envBuffers)
    packs <- forM active $ \l -> do
      j <- eval env grp l i >>= index
      forM [j * width .. j * width + width - 1] $ \ix -> do
        checkBounds b arr ix
        lift (readArray arr (fromIntegral ix))
    forM_ (zip ns (transpose packs)) $ \(n, vs) -> do
      local' <- fresh n
      forM_ (zip active vs) $ \(l, v) -> lift (writeArray local' l $! v)
  SStorePack b i es -> do
    let width = toInteger (length es)
    writes <- forM active $ \l -> do
      j <- eval env grp l i >>= index
      vs <- forM es (eval env grp l)
      pure (zip [j * width ..] vs)
    arr <- maybe (throwE (MissingBuffer b)) pure (Map.lookup b env.envBuffers)
    ordered <- shuffle writes
    forM_ (concat ordered) $ \(ix, v) -> do
      checkBounds b arr ix
      lift (writeArray arr (fromIntegral ix) $! v)
  SIf _ c th el -> do
    conds <- forM active $ \l -> (l,) <$> (eval env grp l c >>= bool)
    block env grp [l | (l, True) <- conds] th
    block env grp [l | (l, False) <- conds] el
  SFor _ v start end step body -> do
    initial <- forM active $ \l -> (l,) <$> eval env grp l start
    arr <- fresh v
    forM_ initial $ \(l, x) -> lift (writeArray arr l $! x)
    let go :: Int -> [Int] -> M s ()
        go iterations lanes = do
          when (iterations > maxIterations) $ throwE (NonTermination v)
          continuing <- fmap concat . forM lanes $ \l -> do
            i <- lift (readArray arr l) >>= u32
            e <- eval env grp l end >>= u32
            pure [l | i < e]
          unless (null continuing) $ do
            block env grp continuing body
            forM_ continuing $ \l -> do
              i <- lift (readArray arr l) >>= u32
              d <- eval env grp l step >>= u32
              when (d == 0 || i + d < i) $ throwE (NonTermination v)
              lift (writeArray arr l $! VU32 (i + d))
            go (iterations + 1) continuing
    go 0 active
  SBarrier ->
    unless (active == allLanes env.envKernel) $ throwE DivergentBarrier
  SSimd n op _ e -> do
    let w = fromIntegral env.envCaps.capSimdWidth
        threads = fromIntegral env.envKernel.kernelThreads
        groupsOf = [[g * w .. min threads ((g + 1) * w) - 1] | g <- [0 .. (threads + w - 1) `div` w - 1]]
        activeSet = IntSet.fromList active
    arr <- fresh n
    forM_ groupsOf $ \lanes -> do
      let present = filter (`IntSet.member` activeSet) lanes
      unless (null present) $ do
        unless (present == lanes) $ throwE (DivergentSimd n)
        vs <- forM lanes (\l -> eval env grp l e)
        r <- reduce op vs
        forM_ lanes $ \l -> lift (writeArray arr l $! r)
 where
  bindLocal n e = do
    vs <- forM active $ \l -> (l,) <$> eval env grp l e
    arr <- fresh n
    forM_ vs $ \(l, v) -> lift (writeArray arr l $! v)
  -- A local's lane vector, created on first binding; a loop rebinds it.
  fresh n = do
    m <- lift (readSTRef grp.groupLocals)
    case Map.lookup n m of
      Just arr -> pure arr
      Nothing -> do
        arr <- lift (newArray (0, fromIntegral env.envKernel.kernelThreads - 1) (VBool False))
        lift (modifySTRef' grp.groupLocals (Map.insert n arr))
        pure arr
  local n = do
    m <- lift (readSTRef grp.groupLocals)
    maybe (throwE (Internal ("local " <> n))) pure (Map.lookup n m)
  index v = toInteger <$> u32 v
  -- The stores of one statement, in an order drawn from the seed.
  shuffle :: [w] -> M s [w]
  shuffle writes = do
    count <- lift (readSTRef env.envStoreCount)
    lift (modifySTRef' env.envStoreCount (+ 1))
    pure (map snd (sortOn fst [(mix (env.envSeed `xor` mix count `xor` fromIntegral j), w) | (j, w) <- zip [0 :: Int ..] writes]))

-- | Iterations after which a loop counts as one that never ends.
maxIterations :: Int
maxIterations = 2 ^ (24 :: Int)

-- | SplitMix's finaliser: a bijection on 64 bits that scatters well.
mix :: Word64 -> Word64
mix z0 =
  let z1 = (z0 `xor` (z0 `shiftR` 30)) * 0xBF58476D1CE4E5B9
      z2 = (z1 `xor` (z1 `shiftR` 27)) * 0x94D049BB133111EB
   in z2 `xor` (z2 `shiftR` 31)

checkBounds :: Name -> STArray s Int a -> Integer -> M s ()
checkBounds n arr ix = do
  (_, hi) <- lift (getBounds arr)
  when (ix < 0 || ix > toInteger hi) $ throwE (OutOfBounds n ix (hi + 1))

-- | A SIMD group's lanes combined pairwise, neighbours first, as the IR orders them.
reduce :: SimdOp -> [Value] -> M s Value
reduce op vs = case vs of
  [] -> throwE (Internal "an empty SIMD group")
  [v] -> pure v
  _ -> pairs vs >>= reduce op
 where
  -- An odd lane out, which no checked kernel's groups have, goes on as it is.
  pairs (a : b : rest) = (:) <$> combine a b <*> pairs rest
  pairs rest = pure rest
  combine a b = case op of
    SimdSum -> arith Add a b
    SimdMax -> arith Max a b
    SimdMin -> arith Min a b

------------------------------------------------------------------------------
-- Expressions

eval :: Env s -> Group s -> Int -> Expr -> M s Value
eval env grp lane = go
 where
  w = fromIntegral env.envCaps.capSimdWidth :: Int
  go = \case
    ELit l -> pure (literal l)
    EVar n -> do
      m <- lift (readSTRef grp.groupLocals)
      case Map.lookup n m of
        Just arr -> lift (readArray arr lane)
        Nothing -> throwE (Internal ("local " <> n))
    EParam n -> maybe (throwE (MissingParam n)) pure (Map.lookup n env.envParams)
    EBuiltin b -> pure . VU32 . fromIntegral $ case b of
      ThreadInGroup -> lane
      GroupInGrid -> fromIntegral grp.groupIndex
      SimdLane -> lane `mod` w
      SimdGroup -> lane `div` w
    EUn op e -> go e >>= unary op
    EBin op a b -> do
      x <- go a
      y <- go b
      arith op x y
    ESelect c a b -> do
      cv <- go c >>= bool
      x <- go a
      y <- go b
      pure (if cv then x else y)
    EFma a b c -> do
      x <- go a
      y <- go b
      z <- go c
      fma x y z
    EConvert t e -> go e >>= convert t
    ENarrowInt o t e -> go e >>= narrowInt o t
    ELoad n i -> do
      ix <- go i >>= u32
      arr <- maybe (throwE (MissingBuffer n)) pure (Map.lookup n env.envBuffers)
      checkBounds n arr (toInteger ix)
      lift (readArray arr (fromIntegral ix))
    ETgLoad n i -> do
      ix <- go i >>= u32
      arr <- maybe (throwE (Internal ("threadgroup array " <> n))) pure (Map.lookup n grp.groupTg)
      checkBounds n arr (toInteger ix)
      lift (readArray arr (fromIntegral ix)) >>= maybe (throwE (UninitializedRead n (fromIntegral ix))) pure

literal :: Literal -> Value
literal = \case
  LF32 x -> VF32 x
  LF16 b -> VF16 (F16 b)
  LBF16 b -> VBF16 (BF16 b)
  LI32 n -> VI32 n
  LU32 n -> VU32 n
  LI8 n -> VI8 n
  LU8 n -> VU8 n
  LBool b -> VBool b

u32 :: Value -> M s Word32
u32 = \case
  VU32 n -> pure n
  v -> throwE (Internal ("not a u32: " <> showT v))

bool :: Value -> M s Bool
bool = \case
  VBool b -> pure b
  v -> throwE (Internal ("not a bool: " <> showT v))

showT :: (Show a) => a -> Text
showT = T.pack . show

unary :: UnOp -> Value -> M s Value
unary op v = case (op, v) of
  (Not, VBool b) -> pure (VBool (not b))
  -- A float's sign flipped or cleared, its payload kept if a NaN.
  (Neg, VF32 x) -> pure (VF32 (castWord32ToFloat (castFloatToWord32 x `xor` 0x80000000)))
  (Neg, VF16 (F16 h)) -> pure (VF16 (F16 (h `xor` 0x8000)))
  (Neg, VI32 n) -> pure (VI32 (negate n))
  (Neg, VU32 n) -> pure (VU32 (negate n))
  (Abs, VF32 x) -> pure (VF32 (castWord32ToFloat (castFloatToWord32 x .&. 0x7FFFFFFF)))
  (Abs, VF16 (F16 h)) -> pure (VF16 (F16 (h .&. 0x7FFF)))
  (Abs, VI32 n) -> pure (VI32 (abs n))
  (Abs, VU32 n) -> pure (VU32 n)
  (Sqrt, VF32 x) -> pure (VF32 (sqrt x))
  -- Correctly rounded through binary32, whose 24 bits are twice binary16's
  -- 11 and two more.
  (Sqrt, VF16 h) -> pure (f16 (sqrt (widenF16 h)))
  (_, VF32 x) -> VF32 . double2Float <$> inBinary64 (float2Double x)
  (_, VF16 h) -> VF16 . narrowF16Double <$> inBinary64 (float2Double (widenF16 h))
  _ -> throwE (Internal ("unary " <> showT op <> " of " <> showT v))
 where
  -- In binary64, the result rounded once to the operand's type.
  inBinary64 x = case op of
    Rsqrt -> pure (1 / sqrt x)
    Exp -> pure (exp x)
    Log -> pure (log x)
    Tanh -> pure (tanh x)
    _ -> throwE (Internal ("unary " <> showT op))

f16 :: Float -> Value
f16 = VF16 . narrowF16

-- | 'IR.EFma' of three floats of one type.
fma :: Value -> Value -> Value -> M s Value
fma a b c = case (a, b, c) of
  (VF32 x, VF32 y, VF32 z) -> pure (VF32 (fused fromRational x y z))
  (VF16 x, VF16 y, VF16 z) -> pure (f16 (fused (widenF16 . rationalF16) (widenF16 x) (widenF16 y) (widenF16 z)))
  _ -> throwE (Internal ("fma of " <> showT a <> ", " <> showT b <> " and " <> showT c))

{- | IEEE 754's fusedMultiplyAdd of values of a type that binary32 holds
exactly: its special cases by their rules, and otherwise the exact
@x * y + z@, a 'Rational', which @nearest@ rounds once to the type.
'Rational' has no infinities, NaNs or -0, so those never reach it. A NaN
result is the quiet NaN with no payload.
-}
fused :: (Rational -> Float) -> Float -> Float -> Float -> Float
fused nearest x y z
  | isNaN x || isNaN y || isNaN z = nan
  | isInfinite x || isInfinite y =
      if x == 0 || y == 0 || (isInfinite z && signum z /= signum infiniteProduct) then nan else infiniteProduct
  | isInfinite z = z
  | exact /= 0 = nearest exact
  -- An exact zero: -0 only if the product and z both are.
  | (x == 0 || y == 0) && negativeProduct && isNegativeZero z = -0
  | otherwise = 0
 where
  nan = castWord32ToFloat 0x7FC00000
  negative v = v < 0 || isNegativeZero v
  negativeProduct = negative x /= negative y
  infiniteProduct = if negativeProduct then -(1 / 0) else 1 / 0
  exact = toRational x * toRational y + toRational z

arith :: forall s. BinOp -> Value -> Value -> M s Value
arith op a b = case (a, b) of
  (VF32 x, VF32 y) -> floatOp VF32 x y
  (VF16 x, VF16 y) -> floatOp f16 (widenF16 x) (widenF16 y)
  (VI32 x, VI32 y) -> intOp VI32 x y
  (VU32 x, VU32 y) -> intOp VU32 x y
  (VBool x, VBool y) -> case op of
    And -> pure (VBool (x && y))
    Or -> pure (VBool (x || y))
    Eq -> pure (VBool (x == y))
    Ne -> pure (VBool (x /= y))
    _ -> bad
  _ -> bad
 where
  bad = throwE (Internal ("binary " <> showT op <> " of " <> showT a <> " and " <> showT b))
  floatOp :: (Float -> Value) -> Float -> Float -> M s Value
  floatOp wrap x y = case op of
    Add -> pure (wrap (x + y))
    Sub -> pure (wrap (x - y))
    Mul -> pure (wrap (x * y))
    Div -> pure (wrap (x / y))
    Min -> pure (wrap (minimumNumber x y))
    Max -> pure (wrap (maximumNumber x y))
    _ -> compare' x y
  intOp :: (Integral i, Bounded i) => (i -> Value) -> i -> i -> M s Value
  intOp wrap x y = case op of
    Add -> pure (wrap (x + y))
    Sub -> pure (wrap (x - y))
    Mul -> pure (wrap (x * y))
    Div
      | y == 0 -> throwE DivideByZero
      | overflows x y -> throwE IntegerOverflow
      | otherwise -> pure (wrap (x `quot` y))
    Rem
      | y == 0 -> throwE DivideByZero
      | overflows x y -> throwE IntegerOverflow
      | otherwise -> pure (wrap (x `rem` y))
    Min -> pure (wrap (min x y))
    Max -> pure (wrap (max x y))
    _ -> compare' x y
  overflows :: (Integral i, Bounded i) => i -> i -> Bool
  overflows x y = toInteger y == -1 && x == minBound
  compare' :: (Ord x) => x -> x -> M s Value
  compare' x y = case op of
    Eq -> pure (VBool (x == y))
    Ne -> pure (VBool (x /= y))
    Lt -> pure (VBool (x < y))
    Le -> pure (VBool (x <= y))
    Gt -> pure (VBool (x > y))
    Ge -> pure (VBool (x >= y))
    _ -> bad

{- | IEEE 754-2019's minimumNumber and maximumNumber: a NaN operand, quiet
or signalling, gives way to the other, two NaNs give the quiet NaN
0x7FC00000, and -0 is below +0. Equal operands differ at most in their
sign, which the bits' or (the minimum) or and (the maximum) settle.
-}
minimumNumber, maximumNumber :: Float -> Float -> Float
minimumNumber x y
  | isNaN x = if isNaN y then castWord32ToFloat 0x7FC00000 else y
  | isNaN y || x < y = x
  | y < x = y
  | otherwise = castWord32ToFloat (castFloatToWord32 x .|. castFloatToWord32 y)
maximumNumber x y
  | isNaN x = if isNaN y then castWord32ToFloat 0x7FC00000 else y
  | isNaN y || x > y = x
  | y > x = y
  | otherwise = castWord32ToFloat (castFloatToWord32 x .&. castFloatToWord32 y)

convert :: ScalarType -> Value -> M s Value
convert t v = case (v, t) of
  (_, _) | valueType v == t -> pure v
  (VBF16 b, TF32) -> pure (VF32 (widenBF16 b))
  (VF16 h, TF32) -> pure (VF32 (widenF16 h))
  (VI8 n, TI32) -> pure (VI32 (fromIntegral n))
  (VU8 n, TU32) -> pure (VU32 (fromIntegral n))
  (VF32 x, TBF16) -> pure (VBF16 (narrowBF16 x))
  (VF32 x, TF16) -> pure (VF16 (narrowF16 x))
  (VU32 n, TF32) -> pure (VF32 (fromIntegral n))
  (VI32 n, TF32) -> pure (VF32 (fromIntegral n))
  _ -> throwE (Internal ("conversion of " <> showT v <> " to " <> showT t))

-- | An integer narrowed: clamped to the narrower range, or its low bits.
narrowInt :: Overflow -> ScalarType -> Value -> M s Value
narrowInt o t v = case (v, t) of
  (VI32 n, TI8) -> pure (VI8 (fromIntegral (case o of Clamp -> max (-128) (min 127 n); Wrap -> n)))
  (VU32 n, TU8) -> pure (VU8 (fromIntegral (case o of Clamp -> min 255 n; Wrap -> n)))
  _ -> throwE (Internal ("integer narrowing of " <> showT v <> " to " <> showT t))
