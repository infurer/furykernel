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
  declares (a buffer's extent, a table's entries and each entry's extent,
  a threadgroup array's size), however much the launch gives, dimensions
  outside their declared range, reads of threadgroup memory never
  written, integer division by zero and the one quotient that overflows,
  loops that never end, and barriers or SIMD-group operations reached by
  part of a group;
* races, as errors: every element remembers the threads that touched it,
  and two threads touching one element unordered, one of them storing,
  stop the run. A threadgroup barrier orders the threadgroup memory of
  what came before it against what comes after, a SIMD-group barrier does
  so for its group's lanes, and nothing orders device memory between
  threads. A fragment operation is its SIMD group's lanes together.

The threadgroups run one after another, in an order drawn from the seed;
within one, the lanes run in lockstep, every lane evaluating a statement
before its stores land, so it is the race detector, not the order, that
finds what could differ on a target.
-}
module FuryKernel.Interp (
  Value (..),
  valueType,
  Launch,
  launch,
  withSeed,
  withTables,
  RunError (..),
  Who (..),
  run,
  runIR,
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

{- | What a launch is given: the parameter struct, every buffer, the seed of
the stores' order, and per table the arrays its entries point at. Made by
'launch' and the setters after it, which what a launch gains later leaves
as they are.
-}
data Launch = Launch
  { launchParams :: Map Name Value
  , launchBuffers :: Map Name [Value]
  , launchSeed :: Word64
  , launchTables :: Map Name [[Value]]
  }

-- | A launch of these parameters and buffers: seed 0, no tables.
launch :: Map Name Value -> Map Name [Value] -> Launch
launch params buffers = Launch params buffers 0 Map.empty

-- | The launch, the order of its stores drawn from this seed.
withSeed :: Word64 -> Launch -> Launch
withSeed seed l = l {launchSeed = seed}

-- | The launch, per table the arrays its entries point at.
withTables :: Map Name [[Value]] -> Launch -> Launch
withTables tables l = l {launchTables = tables}

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
    buffer's or the table entry's extent, the table's entries, or the
    threadgroup array's size.
    -}
    OutOfBounds Name Integer Int
  | -- | A dimension outside its declared range or off its multiple: name, value, range.
    DimOutOfRange Name Word32 DimRange
  | -- | A table with fewer entries than the kernel declares: name, entries, given.
    ShortTable Name Int Int
  | UninitializedRead Name Int
  | DivideByZero
  | -- | A launch size of IR the gate refuses that leaves u32 at a step: the step, and its value.
    LaunchSizeOutOfRange Text Integer
  | -- | The one signed quotient that overflows: the minimum by -1.
    IntegerOverflow
  | {- | A loop that never ends: a lane's variable stood still (a step of 0)
    or wrapped past u32's greatest, or the loop ran 2^24 iterations.
    -}
    NonTermination Name
  | DivergentBarrier
  | DivergentSimd Name
  | {- | Two threads touched one element unordered, one of them storing: the
    memory, the element, and who touched it first and how, then second.
    -}
    Race Name Integer Who Text Who Text
  | Internal Text
  deriving stock (Eq, Show)

-- | Who touched memory.
data Who
  = -- | A thread: its threadgroup and its index in it.
    Thread Word32 Int
  | {- | A SIMD group's lanes together, in a fragment operation: the
    threadgroup and the group's index in it.
    -}
    Lanes Word32 Int
  deriving stock (Eq, Show)

-- | Runs a kernel; returns every buffer as it is afterwards.
run :: forall t. (Target t) => Checked t -> Launch -> Either RunError (Map Name [Value])
run checked = runIR (targetCaps @t) (checkedIR checked)

{- | Runs the IR of a kernel the gate need not have passed, on a target
with these limits, such as one the hazard check refuses: the
interpreter's own errors still hold.
-}
runIR :: TargetCaps -> IR.Kernel -> Launch -> Either RunError (Map Name [Value])
runIR caps k supplied = runST (runExceptT (execute caps k supplied))

type M s = ExceptT RunError (ST s)

data Env s = Env
  { envCaps :: TargetCaps
  , envKernel :: IR.Kernel
  , envParams :: Map Name Value
  , envBuffers :: Map Name (STArray s Int Value)
  , envTables :: Map Name [STArray s Int Value]
  , envSeed :: Word64
  , envStoreCount :: STRef s Word64
  , envMarks :: Map Name (STArray s Int Marks)
  -- ^ Who touched each element of the buffers the kernel stores to.
  }

-- | Per threadgroup: the locals as lane vectors, the threadgroup arrays.
data Group s = Group
  { groupIndex :: Word32
  , groupLocals :: STRef s (Map Name (STArray s Int Value))
  , groupTg :: Map Name (STArray s Int (Maybe Value))
  , groupMarks :: Map Name (STArray s Int Marks)
  , groupEpoch :: STRef s Int
  -- ^ The threadgroup barriers so far.
  , groupSimdEpochs :: STArray s Int Int
  -- ^ Per SIMD group, its barriers and the threadgroup's so far.
  }

-- | A touch of an element: who, after which barriers, and how.
data Mark = Mark
  { markWho :: Who
  , markEpoch :: Int
  , markSimd :: Int
  , markWhat :: Text
  }

-- | An element's last store, and the loads since.
data Marks = Marks (Maybe Mark) [Mark]

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
  -- gives kept aside, untouched; each table as far as its declared entries,
  -- each entry as far as its declared extent.
  buffers <- forM k.kernelBuffers $ \b -> case Map.lookup b.bufName supplied.launchBuffers of
    Nothing -> throwE (MissingBuffer b.bufName)
    Just vs -> do
      forM_ vs $ \v -> when (valueType v /= b.bufType) $ throwE (WrongType b.bufName b.bufType (valueType v))
      extent <- fromIntegral <$> hostU32 paramMap b.bufExtent
      when (length vs < extent) $ throwE (ShortBuffer b.bufName extent (length vs))
      arr <- lift (newListArray (0, extent - 1) (take extent vs))
      pure (b.bufName, (arr, drop extent vs))
  tables <- forM k.kernelTables $ \t -> case Map.lookup t.tableName supplied.launchTables of
    Nothing -> throwE (MissingBuffer t.tableName)
    Just entries -> do
      count <- fromIntegral <$> hostU32 paramMap t.tableEntries
      extent <- fromIntegral <$> hostU32 paramMap t.tableExtent
      when (length entries < count) $ throwE (ShortTable t.tableName count (length entries))
      arrays <- forM (take count entries) $ \vs -> do
        forM_ vs $ \v -> when (valueType v /= t.tableType) $ throwE (WrongType t.tableName t.tableType (valueType v))
        when (length vs < extent) $ throwE (ShortBuffer t.tableName extent (length vs))
        lift (newListArray (0, extent - 1) (take extent vs))
      pure (t.tableName, arrays)
  counter <- lift (newSTRef 0)
  marks <- forM [(b.bufName, arr) | b <- k.kernelBuffers, b.bufAccess /= ReadOnly, (n, (arr, _)) <- buffers, n == b.bufName] $ \(n, arr) -> do
    (lo, hi) <- lift (getBounds arr)
    (n,) <$> lift (newArray (lo, hi) (Marks Nothing []))
  let env = Env caps k paramMap (Map.fromList [(n, arr) | (n, (arr, _)) <- buffers]) (Map.fromList tables) supplied.launchSeed counter (Map.fromList marks)
      w = fromIntegral caps.capSimdWidth
      simds = (fromIntegral k.kernelThreads + w - 1) `div` w
  groups <- hostU32 paramMap k.kernelGroups
  forM_ (sortOn (\g -> mix (supplied.launchSeed `xor` mix (fromIntegral g + 0x9E3779B97F4A7C15))) (if groups == 0 then [] else [0 .. groups - 1])) $ \g -> do
    locals <- lift (newSTRef Map.empty)
    tg <- forM k.kernelTgArrays $ \a -> do
      arr <- lift (newArray (0, fromIntegral a.tgCount - 1) Nothing)
      touched <- lift (newArray (0, fromIntegral a.tgCount - 1) (Marks Nothing []))
      pure ((a.tgName, arr), (a.tgName, touched))
    epoch <- lift (newSTRef 0)
    simdEpochs <- lift (newArray (0, max 0 (simds - 1)) 0)
    let grp = Group g locals (Map.fromList (map fst tg)) (Map.fromList (map snd tg)) epoch simdEpochs
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
("FuryKernel.Check.Launch"), so the values agree with the u64 arithmetic
of "FuryKernel.Emit.Rust"; on IR the gate refuses, a step that leaves u32
is 'LaunchSizeOutOfRange'.
-}
hostU32 :: Map Name Value -> Expr -> M s Word32
hostU32 params = \case
  ELit (LU32 n) -> pure n
  EParam n -> case Map.lookup n params of
    Just (VU32 v) -> pure v
    _ -> throwE (MissingParam n)
  EBin op a b -> do
    x <- toInteger <$> hostU32 params a
    y <- toInteger <$> hostU32 params b
    v <- case op of
      Add -> pure (x + y)
      Sub -> pure (x - y)
      Mul -> pure (x * y)
      Div -> if y == 0 then throwE DivideByZero else pure (x `div` y)
      Rem -> if y == 0 then throwE DivideByZero else pure (x `mod` y)
      _ -> throwE (Internal "launch size operator")
    if v < 0 || v > toInteger (maxBound :: Word32)
      then throwE (LaunchSizeOutOfRange (renderExpr (EBin op a b)) v)
      else pure (fromInteger v)
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
    writes <- forM active $ \l -> (l,,) <$> (eval env grp l i >>= index) <*> eval env grp l e
    arr <- maybe (throwE (MissingBuffer b)) pure (Map.lookup b env.envBuffers)
    landing <- shuffle writes
    forM_ landing $ \(l, ix, v) -> do
      checkBounds b arr ix
      touch env grp True b ix (Thread grp.groupIndex l) True ("the store to " <> b <> "[" <> renderExpr i <> "]")
      lift (writeArray arr (fromIntegral ix) $! v)
  STgStore a i e -> do
    writes <- forM active $ \l -> (l,,) <$> (eval env grp l i >>= index) <*> eval env grp l e
    arr <- maybe (throwE (Internal ("threadgroup array " <> a))) pure (Map.lookup a grp.groupTg)
    landing <- shuffle writes
    forM_ landing $ \(l, ix, v) -> do
      checkBounds a arr ix
      touch env grp False a ix (Thread grp.groupIndex l) True ("the store to " <> a <> "[" <> renderExpr i <> "]")
      lift (writeArray arr (fromIntegral ix) $! Just $! v)
  SLoadPack ns _ b i -> do
    let width = toInteger (length ns)
    arr <- maybe (throwE (MissingBuffer b)) pure (Map.lookup b env.envBuffers)
    packs <- forM active $ \l -> do
      j <- eval env grp l i >>= index
      forM [j * width .. j * width + width - 1] $ \ix -> do
        checkBounds b arr ix
        touch env grp True b ix (Thread grp.groupIndex l) False ("the pack loaded from " <> b <> " at " <> renderExpr i)
        lift (readArray arr (fromIntegral ix))
    forM_ (zip ns (transpose packs)) $ \(n, vs) -> do
      local' <- fresh n
      forM_ (zip active vs) $ \(l, v) -> lift (writeArray local' l $! v)
  SLoadPackVia ns _ t e i -> do
    let width = toInteger (length ns)
    packs <- forM active $ \l -> do
      arr <- tableEntry env t =<< eval env grp l e
      j <- eval env grp l i >>= index
      forM [j * width .. j * width + width - 1] $ \ix -> do
        checkBounds t arr ix
        lift (readArray arr (fromIntegral ix))
    forM_ (zip ns (transpose packs)) $ \(n, vs) -> do
      local' <- fresh n
      forM_ (zip active vs) $ \(l, v) -> lift (writeArray local' l $! v)
  SStorePack b i es -> do
    let width = toInteger (length es)
    writes <- forM active $ \l -> do
      j <- eval env grp l i >>= index
      vs <- forM es (eval env grp l)
      pure [(l, ix, v) | (ix, v) <- zip [j * width ..] vs]
    arr <- maybe (throwE (MissingBuffer b)) pure (Map.lookup b env.envBuffers)
    landing <- shuffle writes
    forM_ (concat landing) $ \(l, ix, v) -> do
      checkBounds b arr ix
      touch env grp True b ix (Thread grp.groupIndex l) True ("the pack stored to " <> b <> " at " <> renderExpr i)
      lift (writeArray arr (fromIntegral ix) $! v)
  STgStorePack a i es -> do
    let width = toInteger (length es)
    writes <- forM active $ \l -> do
      j <- eval env grp l i >>= index
      vs <- forM es (eval env grp l)
      pure [(l, ix, v) | (ix, v) <- zip [j * width ..] vs]
    arr <- maybe (throwE (Internal ("threadgroup array " <> a))) pure (Map.lookup a grp.groupTg)
    landing <- shuffle writes
    forM_ (concat landing) $ \(l, ix, v) -> do
      checkBounds a arr ix
      touch env grp False a ix (Thread grp.groupIndex l) True ("the pack stored to " <> a <> " at " <> renderExpr i)
      lift (writeArray arr (fromIntegral ix) $! Just $! v)
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
  SBarrier -> do
    unless (active == allLanes env.envKernel) $ throwE DivergentBarrier
    lift (modifySTRef' grp.groupEpoch (+ 1))
    (lo, hi) <- lift (getBounds grp.groupSimdEpochs)
    forM_ [lo .. hi] $ \g -> lift (readArray grp.groupSimdEpochs g >>= writeArray grp.groupSimdEpochs g . (+ 1))
  SSimdBarrier -> do
    let w = fromIntegral env.envCaps.capSimdWidth
        threads = fromIntegral env.envKernel.kernelThreads
        groupsOf = [[g * w .. min threads ((g + 1) * w) - 1] | g <- [0 .. (threads + w - 1) `div` w - 1]]
        activeSet = IntSet.fromList active
    forM_ (zip [0 ..] groupsOf) $ \(g, lanes) -> do
      let present = filter (`IntSet.member` activeSet) lanes
      unless (null present || present == lanes) $ throwE (DivergentSimd "simdgroup barrier")
      unless (null present) $ lift (readArray grp.groupSimdEpochs g >>= writeArray grp.groupSimdEpochs g . (+ 1))
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
  SFragVar n t -> simdGroups n $ \g _ -> do
    arr <- fragment n
    forM_ [0 .. 63] $ \e -> lift (writeArray arr (g * 64 + e) $! zero t)
  SFragLoad n m i st tr -> simdGroups n $ \g lane -> do
    base <- eval env grp lane i >>= index
    stride <- eval env grp lane st >>= index
    arr <- fragment n
    forM_ [(r, c) | r <- [0 .. 7], c <- [0 .. 7]] $ \(r, c) -> do
      let ix = base + (if tr then c * stride + r else r * stride + c)
      v <- fragMemoryRead m ix
      touchFragment m ix g False ("the fragment loaded from " <> memoryName m <> " at " <> renderExpr i)
      lift (writeArray arr (g * 64 + fromIntegral (r * 8 + c)) $! v)
  SFragStore m i st n -> simdGroups n $ \g lane -> do
    base <- eval env grp lane i >>= index
    stride <- eval env grp lane st >>= index
    arr <- fragment n
    forM_ [(r, c) | r <- [0 .. 7], c <- [0 .. 7]] $ \(r, c) -> do
      v <- lift (readArray arr (g * 64 + fromIntegral (r * 8 + c)))
      fragMemoryWrite m (base + r * stride + c) v
      touchFragment m (base + r * stride + c) g True ("the fragment stored to " <> memoryName m <> " at " <> renderExpr i)
  SFragScaleRows f a i -> simdGroups f $ \g lane -> do
    base <- eval env grp lane i >>= index
    arr <- maybe (throwE (Internal ("threadgroup array " <> a))) pure (Map.lookup a grp.groupTg)
    ff <- fragment f
    forM_ [0 .. 7] $ \r -> do
      checkBounds a arr (base + r)
      touch env grp False a (base + r) (Lanes grp.groupIndex g) False ("the row scales read from " <> a <> " at " <> renderExpr i)
      scale <- lift (readArray arr (fromIntegral (base + r))) >>= maybe (throwE (UninitializedRead a (fromIntegral (base + r)))) pure >>= asFloat
      forM_ [0 .. 7] $ \col -> do
        v <- lift (readArray ff (g * 64 + fromIntegral (r * 8 + col))) >>= asFloat
        lift (writeArray ff (g * 64 + fromIntegral (r * 8 + col)) $! VF32 (v * scale))
  SFragConvert d src -> simdGroups d $ \g _ -> do
    fs <- fragment src
    fd <- fragment d
    forM_ [0 .. 63] $ \e -> do
      v <- lift (readArray fs (g * 64 + e)) >>= convert TF32
      lift (writeArray fd (g * 64 + e) $! v)
  SFragMma d a b c -> simdGroups d $ \g _ -> do
    fa <- fragment a
    fb <- fragment b
    fc <- fragment c
    let at arr r k = lift (readArray arr (g * 64 + r * 8 + k)) >>= asFloat
    results <- forM [(r, col) | r <- [0 .. 7], col <- [0 .. 7]] $ \(r, col) -> do
      start <- at fc r col
      products <- forM [0 .. 7] $ \k -> (*) <$> at fa r k <*> at fb k col
      pure (r * 8 + col, foldl (+) start products)
    fd <- fragment d
    forM_ results $ \(e, v) -> lift (writeArray fd (g * 64 + e) $! VF32 v)
 where
  touchFragment m ix g store what = case m of
    FDevice b -> touch env grp True b ix (Lanes grp.groupIndex g) store what
    FThreadgroup a -> touch env grp False a ix (Lanes grp.groupIndex g) store what
  memoryName = \case
    FDevice b -> b
    FThreadgroup a -> a
  -- A fragment operation, once per SIMD group with any lane active: every
  -- lane of it must be, and the group's first lane gives the addresses.
  simdGroups :: Name -> (Int -> Int -> M s ()) -> M s ()
  simdGroups n body = do
    let w = fromIntegral env.envCaps.capSimdWidth
        threads = fromIntegral env.envKernel.kernelThreads
        activeSet = IntSet.fromList active
    forM_ [0 .. (threads + w - 1) `div` w - 1] $ \g -> do
      let lanes = [g * w .. min threads ((g + 1) * w) - 1]
          present = filter (`IntSet.member` activeSet) lanes
      unless (null present) $ do
        unless (present == lanes) $ throwE (DivergentSimd n)
        body g (g * w)
  -- A fragment's elements, 64 a SIMD group, made on first use.
  fragment n = do
    m <- lift (readSTRef grp.groupLocals)
    case Map.lookup n m of
      Just arr -> pure arr
      Nothing -> do
        let w = fromIntegral env.envCaps.capSimdWidth
            groups = (fromIntegral env.envKernel.kernelThreads + w - 1) `div` w
        arr <- lift (newArray (0, groups * 64 - 1) (VF32 0))
        lift (modifySTRef' grp.groupLocals (Map.insert n arr))
        pure arr
  zero = \case
    TBF16 -> VBF16 (narrowBF16 0)
    TF16 -> VF16 (narrowF16 0)
    _ -> VF32 0
  asFloat = \case
    VF32 x -> pure x
    VBF16 b -> pure (widenBF16 b)
    VF16 h -> pure (widenF16 h)
    v -> throwE (Internal ("a fragment element " <> T.pack (show v)))
  fragMemoryRead m ix = case m of
    FDevice b -> do
      arr <- maybe (throwE (MissingBuffer b)) pure (Map.lookup b env.envBuffers)
      checkBounds b arr ix
      lift (readArray arr (fromIntegral ix))
    FThreadgroup a -> do
      arr <- maybe (throwE (Internal ("threadgroup array " <> a))) pure (Map.lookup a grp.groupTg)
      checkBounds a arr ix
      lift (readArray arr (fromIntegral ix)) >>= maybe (throwE (UninitializedRead a (fromIntegral ix))) pure
  fragMemoryWrite m ix v = case m of
    FDevice b -> do
      arr <- maybe (throwE (MissingBuffer b)) pure (Map.lookup b env.envBuffers)
      checkBounds b arr ix
      lift (writeArray arr (fromIntegral ix) $! v)
    FThreadgroup a -> do
      arr <- maybe (throwE (Internal ("threadgroup array " <> a))) pure (Map.lookup a grp.groupTg)
      checkBounds a arr ix
      lift (writeArray arr (fromIntegral ix) $! Just $! v)
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

{- | A touch of element @ix@ of a threadgroup array or, for @device@, of a
buffer the kernel stores to: a race with an earlier touch it isn't ordered
after, one of them a store, is an error.
-}
touch :: Env s -> Group s -> Bool -> Name -> Integer -> Who -> Bool -> Text -> M s ()
touch env grp device name ix who store what =
  case Map.lookup name (if device then env.envMarks else grp.groupMarks) of
    Nothing -> pure ()
    Just arr -> do
      epoch <- if device then pure 0 else lift (readSTRef grp.groupEpoch)
      simd <- if device then pure 0 else lift (readArray grp.groupSimdEpochs (snd (simdOf who)))
      let now = Mark who epoch simd what
          unordered m = not (ordered m now)
      Marks lastStore loads <- lift (readArray arr (fromIntegral ix))
      case [m | Just m <- [lastStore], unordered m] <> (if store then filter unordered loads else []) of
        m : _ -> throwE (Race name ix m.markWho m.markWhat who what)
        [] -> pure ()
      -- A load before the last barrier is ordered before any touch to
      -- come, one superseded by a later one of the same thread is no
      -- longer needed, and on device memory one other thread's suffices.
      lift . writeArray arr (fromIntegral ix) $!
        if store
          then Marks (Just now) []
          else Marks lastStore (now : (if device then take 1 else id) [l | l <- loads, l.markEpoch >= epoch, not (same l.markWho who)])
 where
  w = fromIntegral env.envCaps.capSimdWidth
  simdOf = \case
    Thread g t -> (g, t `div` w)
    Lanes g s -> (g, s)
  same a b = case (a, b) of
    (Thread g t, Thread g' t') -> g == g' && t == t'
    _ -> False
  -- One thread's touches in its own order; on threadgroup memory, a
  -- barrier's, and among a SIMD group's lanes, a SIMD-group barrier's.
  ordered a b
    | same a.markWho b.markWho = True
    | device = False
    | a.markEpoch < b.markEpoch = True
    | otherwise = simdOf a.markWho == simdOf b.markWho && a.markSimd < b.markSimd

-- | The array an entry of a table points at.
tableEntry :: Env s -> Name -> Value -> M s (STArray s Int Value)
tableEntry env t e = do
  entry <- u32 e
  arrays <- maybe (throwE (MissingBuffer t)) pure (Map.lookup t env.envTables)
  case drop (fromIntegral entry) arrays of
    arr : _ -> pure arr
    [] -> throwE (OutOfBounds t (toInteger entry) (length arrays))

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
      touch env grp True n (toInteger ix) (Thread grp.groupIndex lane) False ("the load of " <> n <> "[" <> renderExpr i <> "]")
      lift (readArray arr (fromIntegral ix))
    ELoadVia n e i -> do
      arr <- tableEntry env n =<< go e
      ix <- go i >>= u32
      checkBounds n arr (toInteger ix)
      lift (readArray arr (fromIntegral ix))
    ETgLoad n i -> do
      ix <- go i >>= u32
      arr <- maybe (throwE (Internal ("threadgroup array " <> n))) pure (Map.lookup n grp.groupTg)
      checkBounds n arr (toInteger ix)
      touch env grp False n (toInteger ix) (Thread grp.groupIndex lane) False ("the load of " <> n <> "[" <> renderExpr i <> "]")
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
