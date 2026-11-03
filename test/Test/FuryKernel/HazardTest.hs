-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE AllowAmbiguousTypes #-}

{- | Races: the hazard check refuses them, and the interpreter stops on
them; the library's kernels pass both, and lose a barrier only where the
interpreter, finding the race that makes, is not the only one to notice.
-}
module Test.FuryKernel.HazardTest (tests) where

import Control.Monad (void, when)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Word (Word32)
import FuryKernel.Build
import FuryKernel.Check (checkIR)
import FuryKernel.Constraint (TgArrayOk)
import FuryKernel.Finding (Finding (..), Rule (..))
import FuryKernel.IR (DimRange (..), Name, Stmt (..))
import FuryKernel.IR qualified as IR
import FuryKernel.Interp
import FuryKernel.Reduce (combineSplits, tgReduceSum)
import FuryKernel.Target (Target, targetCaps)
import FuryKernel.Tile (GemmTile (..), gemmTile)
import FuryKernel.Type (BF16 (..))
import GHC.TypeNats (KnownNat)
import Test.FuryKernel.Targets (T32)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (Assertion, assertBool, assertFailure, testCase, (@?=))

tests :: TestTree
tests =
  testGroup
    "hazards"
    [ testCase "a store and another thread's load between the same barriers" $ do
        refused rawNoBarrier ["Threads may race on the threadgroup array t: the store to t[@thread] and the load of t[(63u - @thread)] can touch one element between the same barriers."]
        raced rawNoBarrier [] [("y", zeros 64)] "t"
    , testCase "a store and another thread's load between the same barriers, the load an fma's operand" $ do
        refused fmaNoBarrier ["Threads may race on the threadgroup array t: the store to t[@thread] and the load of t[(63u - @thread)] can touch one element between the same barriers."]
        raced fmaNoBarrier [] [("y", replicate 64 (VF32 0))] "t"
    , testCase "a loop's next stores before the loads of the last, without a barrier" $ do
        refused warNoBarrier ["Threads may race on the threadgroup array t: the store to t[@thread] and the load of t[(63u - @thread)] can touch one element between the same barriers."]
        raced warNoBarrier [] [("x", map VU32 [0 .. 191]), ("y", zeros 192)] "t"
    , testCase "a threadgroup loads what another stores to device memory" $ do
        refused chain ["Threads may race on the buffer y: the load of y[@group] and the store to y[(@group + 1u)] can touch one element in two threads of the launch."]
        raced chain [] [("y", zeros 5)] "y"
    , testCase "two SIMD groups store to one element, in two statements" $ do
        refused storesOfTwo ["Threads may race on the buffer y: the store to y[0u] and the store to y[0u] can touch one element in two threads of the launch."]
        raced storesOfTwo [] [("y", zeros 1)] "y"
    , testCase "every threadgroup stores to one element" $ do
        refused storesOfAll ["Threads may race on the buffer y: the store to y[0u] in two threads can touch one element in two threads of the launch."]
        raced storesOfAll [] [("y", zeros 1)] "y"
    , testCase "two SIMD groups store fragments to one place" $ do
        refused fragmentsOfTwo ["Threads may race on the buffer c: the fragment stored to c at 0u in two threads can touch one element in two threads of the launch."]
        raced fragmentsOfTwo [] [("a", map VF32 [0 .. 127]), ("c", replicate 64 (VF32 0))] "c"
    , testCase "a fragment stored over itself, its rows four apart" $ do
        refused overlapping ["Threads may race on the buffer c: the fragment stored to c at 0u in two threads can touch one element in two threads of the launch."]
        raced overlapping [] [("a", map VF32 [0 .. 63]), ("c", replicate 64 (VF32 0))] "c"
    , testCase "a barrier orders no device memory" $ do
        refused deviceAcross ["Threads may race on the buffer y: the store to y[@thread] and the load of y[(63u - @thread)] can touch one element in two threads of the launch."]
        raced deviceAcross [] [("y", zeros 64), ("z", zeros 64)] "y"
    , testCase "lanes of a SIMD group hand values over through threadgroup memory only after a SIMD-group barrier" $ do
        refused (handOver False) ["Threads may race on the threadgroup array t: the store to t[@thread] and the load of t[((@thread + 1u) - (2u * (@thread % 2u)))] can touch one element between the same barriers."]
        raced (handOver False) [] [("y", zeros 64)] "t"
        refused (handOver True) []
        fmap (Map.lookup "y") (runIR (targetCaps @T32) (kernelIR (handOver True)) (launched [] [("y", zeros 64)] 0))
          @?= Right (Just [VU32 (fromIntegral (l `xorU` 1)) | l <- [0 .. 63 :: Int]])
    , testCase "a buffer declared apart: threadgroups' stores at loaded places pass, the launch's data keeps them apart" $ do
        refused (scatter False) ["Threads may race on the buffer y: the store to y[places[@group]] in two threads can touch one element in two threads of the launch."]
        refused (scatter True) []
        fmap (Map.lookup "y") (runIR (targetCaps @T32) (kernelIR (scatter True)) (launched [] [("places", map VU32 [3, 0, 2, 1]), ("y", zeros 4)] 0))
          @?= Right (Just (map VU32 [1, 3, 2, 0]))
        raced (scatter True) [] [("places", map VU32 [3, 0, 3, 1]), ("y", zeros 4)] "y"
    , testCase "a buffer declared apart: the threads of one threadgroup are still checked" $
        refused (scatterTwice True) ["Threads may race on the buffer y: the store to y[places[@group]] and the store to y[places[@group]] can touch one element in two threads of the launch."]
    , testCase "passes over a stage, each of blocks that load it, then a store over it, pass, and race without their barriers" $
        mapM_
          ( \loaded -> do
              refused (staged loaded Waited) []
              mapM_
                ( \seed ->
                    fmap (Map.lookup "y") (runIR (targetCaps @T32) (kernelIR (staged loaded Waited)) (launched [] stagedBuffers seed))
                      @?= Right (Just [VU32 (64 - i) | i <- [0 .. 63]])
                )
                [0 .. 7]
              racesWithout (staged loaded Waited) [] stagedBuffers
          )
          [False, True]
    , testCase "passes over a stage race without the barrier after the store, without the blocks', or with the store in a block" $
        mapM_
          ( \shape -> do
              refused (staged True shape) ["Threads may race on the threadgroup array stage: the load of stage[(63u - @thread)] and the store to stage[@thread] can touch one element between the same barriers."]
              raced (staged True shape) [] stagedBuffers "stage"
          )
          [Unwaited, Unblocked, Inside]
    , testCase "a pass's store before the next pass's load, the barriers between them only before the store" $ do
        refused restaged ["Threads may race on the threadgroup array stage: the load of stage[(63u - @thread)] and the store to stage[@thread] can touch one element between the same barriers."]
        raced restaged [] stagedBuffers "stage"
    , testCase "a store and a load kept apart by a value per pass, a local a search finds, a branch's barrier or a minimum pass" $
        mapM_
          ( \(k, data') -> do
              refused k []
              mapM_ (\(bs, seed) -> void (runIR (targetCaps @T32) (kernelIR k) (launched [] bs seed)) @?= Right ()) [(bs, seed) | bs <- data', seed <- [0 .. 7]]
          )
          [ (turns False, [stagedBuffers, takenBuffers])
          , (searched False, [stagedBuffers, takenBuffers])
          , (branched False, [stagedBuffers, takenBuffers])
          , (clamped False, [[("y", zeros 128)]])
          ]
    , testCase "their near misses race" $ do
        mapM_
          ( \k -> do
              refused k ["Threads may race on the threadgroup array stage: the store to stage[@thread] and the load of stage[(63u - @thread)] can touch one element between the same barriers."]
              raced k [] takenBuffers "stage"
          )
          [turns True, searched True, branched True]
        refused (clamped True) ["Threads may race on the threadgroup array t: the store to t[((((@simd_group * 0u) + (@simd_group min 1u)) * 32u) + @lane)] in two threads can touch one element between the same barriers."]
        raced (clamped True) [] [("y", zeros 128)] "t"
    , testCase "the library's reductions and tiles pass, and race without their barriers" $ do
        refused reduced []
        refused combined []
        refused tiled []
        racesWithout reduced [] [("y", [VF32 0])]
        racesWithout combined [] [("p", map VF32 [1 .. 8]), ("y", replicate 2 (VF32 0))]
        racesWithout tiled tileParams tileBuffers
    , testCase "a tile's caller stores what its runs read, then a barrier: none comes before the tile's first step" $ do
        refused (stagedTile True) []
        assertBool "the hazard check passes the caller without its barrier" (any ((== Hazards) . (.findingRule)) (checkIR (targetCaps @T32) (kernelIR (stagedTile False))))
        raced (stagedTile False) tileParams tileBuffers "a_rows"
    , testCase "a shift by one through threadgroup memory, no barrier between its loads and stores: refused and raced" $ do
        assertBool "the hazard check passes the shift" (any ((== Hazards) . (.findingRule)) (checkIR (targetCaps @T32) (kernelIR shifted)))
        raced shifted [] [("y", replicate 65 (VF32 0))] "t"
    , testCase "a barrier the interpreter finds a race without, the hazard check finds one without" $ do
        soundWithout reduced [] [("y", [VF32 0])]
        soundWithout combined [] [("p", map VF32 [1 .. 8]), ("y", replicate 2 (VF32 0))]
        soundWithout tiled tileParams tileBuffers
        soundWithout (staged False Waited) [] stagedBuffers
        soundWithout (staged True Waited) [] stagedBuffers
    ]
 where
  xorU a b = if even a then a + b else a - b

------------------------------------------------------------------------------
-- Checks

-- | The hazards the gate finds, by their messages.
refused :: forall t. (Target t) => Kernel t -> [Text] -> Assertion
refused k want = [f.findingMessage | f <- checkIR (targetCaps @t) (kernelIR k), f.findingRule == Hazards] @?= want

launched :: [(Name, Value)] -> [(Name, [Value])] -> Word32 -> Launch
launched ps bs seed = withSeed (fromIntegral seed) (launch (Map.fromList ps) (Map.fromList bs))

-- | The interpreter stops on a race on the named memory, whatever the seed.
raced :: forall t. (Target t) => Kernel t -> [(Name, Value)] -> [(Name, [Value])] -> Name -> Assertion
raced k ps bs memory =
  mapM_
    ( \seed -> case runIR (targetCaps @t) (kernelIR k) (launched ps bs seed) of
        Left (Race m _ a _ b _) -> do
          m @?= memory
          -- A fragment operation's lanes may be any of their group's.
          assertBool "a race of one thread with itself" (a /= b || isLanes a)
        other -> assertFailure ("no race, seed " <> show seed <> ": " <> show other)
    )
    [0 .. 7]

-- | Without any barrier, both the gate and the interpreter find a race.
racesWithout :: forall t. (Target t) => Kernel t -> [(Name, Value)] -> [(Name, [Value])] -> Assertion
racesWithout k ps bs = do
  let bare = (kernelIR k) {IR.kernelBody = concatMap strip (kernelIR k).kernelBody}
  assertBool "the hazard check passes the kernel without barriers" (any ((== Hazards) . (.findingRule)) (checkIR (targetCaps @t) bare))
  case runIR (targetCaps @t) bare (launched ps bs 0) of
    Left Race {} -> pure ()
    other -> assertFailure ("no race without barriers: " <> show (fmap Map.keys other))
 where
  strip = \case
    SBarrier -> []
    SSimdBarrier -> []
    SIf r c th el -> [SIf r c (concatMap strip th) (concatMap strip el)]
    SFor r v a b c body -> [SFor r v a b c (concatMap strip body)]
    s -> [s]

-- | Each barrier gone: where the interpreter finds a race, the gate does.
soundWithout :: forall t. (Target t) => Kernel t -> [(Name, Value)] -> [(Name, [Value])] -> Assertion
soundWithout k ps bs =
  mapM_
    ( \i -> do
        let mutant = (kernelIR k) {IR.kernelBody = snd (dropBarrier i 0 (kernelIR k).kernelBody)}
            found = any ((== Hazards) . (.findingRule)) (checkIR (targetCaps @t) mutant)
        case runIR (targetCaps @t) mutant (launched ps bs 0) of
          Left Race {} -> assertBool ("barrier " <> show i <> " gone: the interpreter finds a race, the gate none") found
          _ -> pure ()
    )
    [0 .. barriers (kernelIR k).kernelBody - 1]
 where
  barriers = sum . map count
  count = \case
    SBarrier -> 1
    SSimdBarrier -> 1
    SIf _ _ th el -> barriers th + barriers el
    SFor _ _ _ _ _ body -> barriers body
    _ -> 0 :: Int
  -- The statements with barrier @i@ (counting from @n@) gone, and the
  -- barriers they hold.
  dropBarrier :: Int -> Int -> [Stmt] -> (Int, [Stmt])
  dropBarrier _ n [] = (n, [])
  dropBarrier i n (s : rest) =
    let (n', s') = case s of
          SBarrier -> (n + 1, [SBarrier | n /= i])
          SSimdBarrier -> (n + 1, [SSimdBarrier | n /= i])
          SIf r c th el ->
            let (n1, th') = dropBarrier i n th
                (n2, el') = dropBarrier i n1 el
             in (n2, [SIf r c th' el'])
          SFor r v a b c body -> let (n1, body') = dropBarrier i n body in (n1, [SFor r v a b c body'])
          _ -> (n, [s])
        (n'', rest') = dropBarrier i n' rest
     in (n'', s' <> rest')

isLanes :: Who -> Bool
isLanes = \case
  Lanes {} -> True
  Thread {} -> False

-- | Each thread moves its element of a threadgroup array one up, no barrier between the loads and the stores.
shifted :: Kernel T32
shifted = kernel "shifted" $ do
  y <- buffer @WriteOnly @16 @Float "y" 65
  threads @64
  groups 1
  pure $ do
    t <- tgArray @65 @Float "t"
    pure $ do
      tgStore t threadIndex (toF32 threadIndex)
      barrier
      tgStore t (threadIndex + 1) (tgLoad t threadIndex)
      barrier
      strided "i" 65 $ \i -> store y i (tgLoad t i)

zeros :: Int -> [Value]
zeros n = replicate n (VU32 0)

------------------------------------------------------------------------------
-- Kernels that race

-- | Every thread stores its index, then without a barrier loads another's.
rawNoBarrier :: Kernel T32
rawNoBarrier = kernel "raw" $ do
  y <- buffer @WriteOnly @16 @Word32 "y" 64
  threads @64
  groups 1
  pure $ do
    t <- tgArray @64 @Word32 "t"
    pure $ do
      tgStore t threadIndex threadIndex
      store y threadIndex (tgLoad t (63 - threadIndex))

-- | 'rawNoBarrier' with the load an operand of an fma.
fmaNoBarrier :: Kernel T32
fmaNoBarrier = kernel "raw_fma" $ do
  y <- buffer @WriteOnly @16 @Float "y" 64
  threads @64
  groups 1
  pure $ do
    t <- tgArray @64 @Float "t"
    pure $ do
      tgStore t threadIndex 1
      store y threadIndex (fmaE 2 (tgLoad t (63 - threadIndex)) 1)

-- | A loop that stages, waits, and loads another's, but stages the next before all have loaded.
warNoBarrier :: Kernel T32
warNoBarrier = kernel "war" $ do
  x <- buffer @ReadOnly @16 @Word32 "x" 192
  y <- buffer @WriteOnly @16 @Word32 "y" 192
  threads @64
  groups 1
  pure $ do
    t <- tgArray @64 @Word32 "t"
    pure $ forU "s" 0 3 $ \s -> do
      tgStore t threadIndex (load x (s * 64 + threadIndex))
      barrier
      store y (s * 64 + threadIndex) (tgLoad t (63 - threadIndex))

-- | Threadgroup g loads what g - 1 stores.
chain :: Kernel T32
chain = kernel "chain" $ do
  y <- buffer @ReadWrite @16 @Word32 "y" 5
  threads @32
  groups 4
  pure . pure $ whenLane (threadIndex .== 0) (store y (groupIndex + 1) (load y groupIndex + 1))

-- | Lanes of two SIMD groups store to one element.
storesOfTwo :: Kernel T32
storesOfTwo = kernel "stores_of_two" $ do
  y <- buffer @WriteOnly @16 @Word32 "y" 1
  threads @64
  groups 1
  pure . pure $ do
    whenLane (threadIndex .== 0) (store y 0 1)
    whenLane (threadIndex .== 32) (store y 0 2)

-- | Every threadgroup stores to one element.
storesOfAll :: Kernel T32
storesOfAll = kernel "stores_of_all" $ do
  y <- buffer @WriteOnly @16 @Word32 "y" 1
  threads @32
  groups 4
  pure . pure $ whenLane (threadIndex .== 0) (store y 0 groupIndex)

-- | Two SIMD groups store their fragments to one place.
fragmentsOfTwo :: Kernel T32
fragmentsOfTwo = kernel "fragments_of_two" $ do
  a <- buffer @ReadOnly @16 @Float "a" 128
  c <- buffer @WriteOnly @16 @Float "c" 64
  threads @64
  groups 1
  pure . pure $ do
    f <- fragment @Float "f"
    fragLoad f a (simdGroupIndex * 64) 8
    fragStore c 0 8 f

-- | A fragment stored with its rows four elements apart, over itself.
overlapping :: Kernel T32
overlapping = kernel "overlapping" $ do
  a <- buffer @ReadOnly @16 @Float "a" 64
  c <- buffer @WriteOnly @16 @Float "c" 64
  threads @32
  groups 1
  pure . pure $ do
    f <- fragment @Float "f"
    fragLoad f a 0 8
    fragStore c 0 4 f

-- | Device memory one SIMD group stores and another loads after a barrier.
deviceAcross :: Kernel T32
deviceAcross = kernel "device_across" $ do
  y <- buffer @ReadWrite @16 @Word32 "y" 64
  z <- buffer @WriteOnly @16 @Word32 "z" 64
  threads @64
  groups 1
  pure . pure $ do
    store y threadIndex (threadIndex * 10)
    barrier
    store z threadIndex (load y (63 - threadIndex))

-- | Each lane hands its index to its neighbour through threadgroup memory.
handOver :: Bool -> Kernel T32
handOver waits = kernel "hand_over" $ do
  y <- buffer @WriteOnly @16 @Word32 "y" 64
  threads @64
  groups 1
  pure $ do
    t <- tgArray @64 @Word32 "t"
    pure $ do
      tgStore t threadIndex threadIndex
      when waits simdBarrier
      store y threadIndex (tgLoad t (threadIndex + 1 - 2 * (threadIndex `remE` 2)))

-- | Each threadgroup's first thread stores its index at the place loaded for it.
scatter :: Bool -> Kernel T32
scatter declared = kernel "scatter" $ do
  places <- buffer @ReadOnly @16 @Word32 "places" 4
  y <- buffer @WriteOnly @16 @Word32 "y" 4
  when declared (apart y)
  threads @32
  groups 4
  pure . pure $ whenLane (threadIndex .== 0) (store y (load places groupIndex) groupIndex)

-- | 'scatter' by two threads of each threadgroup.
scatterTwice :: Bool -> Kernel T32
scatterTwice declared = kernel "scatter_twice" $ do
  places <- buffer @ReadOnly @16 @Word32 "places" 4
  y <- buffer @WriteOnly @16 @Word32 "y" 4
  when declared (apart y)
  threads @32
  groups 4
  pure . pure $ do
    whenLane (threadIndex .== 0) (store y (load places groupIndex) groupIndex)
    whenLane (threadIndex .== 1) (store y (load places groupIndex) groupIndex)

-- | How a pass of 'staged' ends its blocks and itself.
data Shape
  = {- | A barrier ends each block; after the blocks, a store over the
    stage and a barrier.
    -}
    Waited
  | -- | No barrier after the store.
    Unwaited
  | -- | No barrier in the blocks.
    Unblocked
  | -- | The store in each block, before its barrier.
    Inside
  deriving stock (Eq)

{- | Two passes over a staged array, each of blocks that load it, the blocks
of a pass three or loaded: a uniform inner loop of a barrier each, then a
store over the stage.
-}
staged :: Bool -> Shape -> Kernel T32
staged loaded shape = kernel "staged" $ do
  counts <- buffer @ReadOnly @16 @Word32 "counts" 4
  y <- buffer @WriteOnly @16 @Word32 "y" 64
  threads @64
  groups 1
  pure $ do
    stage <- tgArray @64 @Word32 "stage"
    pure $ do
      tgStore stage threadIndex threadIndex
      barrier
      forU "pass" 0 2 $ \p -> do
        forU "block" 0 (if loaded then load counts p else 3) $ \_ -> do
          store y threadIndex (tgLoad stage (63 - threadIndex))
          when (shape == Inside) (tgStore stage threadIndex (threadIndex + 1))
          when (shape /= Unblocked) barrier
        when (shape /= Inside) (tgStore stage threadIndex (threadIndex + 1))
        when (shape /= Unwaited) barrier

{- | Passes that load the stage, wait, run blocks of a barrier each, then
store over the stage: the next pass loads it with no barrier after the
store.
-}
restaged :: Kernel T32
restaged = kernel "restaged" $ do
  counts <- buffer @ReadOnly @16 @Word32 "counts" 4
  y <- buffer @WriteOnly @16 @Word32 "y" 64
  threads @64
  groups 1
  pure $ do
    stage <- tgArray @64 @Word32 "stage"
    pure $ do
      tgStore stage threadIndex threadIndex
      barrier
      forU "pass" 0 2 $ \p -> do
        v <- let_ "v" (tgLoad stage (63 - threadIndex))
        barrier
        forU "block" 0 (load counts p) (const barrier)
        store y threadIndex v
        tgStore stage threadIndex (threadIndex + 1)

-- | The first pass of none of its blocks.
stagedBuffers :: [(Name, [Value])]
stagedBuffers = [("counts", map VU32 [0, 2, 1, 1]), ("y", zeros 64)]

-- | The first pass of one block, the second of none.
takenBuffers :: [(Name, [Value])]
takenBuffers = [("counts", map VU32 [1, 0, 1, 1]), ("y", zeros 64)]

{- | Passes that store over the stage or load it as a value of the pass picks,
or, a near miss, both in the first.
-}
turns :: Bool -> Kernel T32
turns near = kernel "turns" $ do
  counts <- buffer @ReadOnly @16 @Word32 "counts" 4
  y <- buffer @WriteOnly @16 @Word32 "y" 64
  threads @64
  groups 1
  pure $ do
    stage <- tgArray @64 @Word32 "stage"
    pure $ do
      tgStore stage threadIndex 0
      barrier
      forU "pass" 0 2 $ \p -> do
        turn <- let_ @Word32 "turn" (select (p .== 0) 1 2)
        whenU (turn .== 1) (tgStore stage threadIndex (threadIndex + 1))
        whenU (turn .== (if near then 1 else 2)) (store y threadIndex (tgLoad stage (63 - threadIndex)))
        forU "block" 0 (load counts p) (const barrier)
        barrier

{- | Passes that store over the stage or load it as a local found by a
search picks, or, a near miss, both where it's 1.
-}
searched :: Bool -> Kernel T32
searched near = kernel "searched" $ do
  counts <- buffer @ReadOnly @16 @Word32 "counts" 4
  y <- buffer @WriteOnly @16 @Word32 "y" 64
  threads @64
  groups 1
  pure $ do
    stage <- tgArray @64 @Word32 "stage"
    pure $ do
      tgStore stage threadIndex 0
      barrier
      forU "pass" 0 2 $ \p -> do
        lo <- var "lo" 0
        forU "search" 0 3 $ \_ -> lo .= select (load counts p .> get lo) (get lo + 1) (get lo)
        found <- let_ "found" (get lo)
        whenU (found .< 2) (tgStore stage threadIndex (threadIndex + 1))
        whenU (if near then found .< 3 else found .>= 2) (store y threadIndex (tgLoad stage (63 - threadIndex)))
        forU "block" 0 (load counts p) (const barrier)
        barrier

{- | A store over the stage in a branch that ends with a barrier, a load
after the branch; or, a near miss, the barrier before the store.
-}
branched :: Bool -> Kernel T32
branched near = kernel "branched" $ do
  counts <- buffer @ReadOnly @16 @Word32 "counts" 4
  y <- buffer @WriteOnly @16 @Word32 "y" 64
  threads @64
  groups 1
  pure $ do
    stage <- tgArray @64 @Word32 "stage"
    pure $ do
      tgStore stage threadIndex 0
      barrier
      whenU (load counts 0 .== 1) $
        if near then barrier >> tgStore stage threadIndex 1 else tgStore stage threadIndex 1 >> barrier
      store y threadIndex (tgLoad stage (63 - threadIndex))

{- | SIMD groups storing at places a minimum keeps apart; or, a near miss,
groups 1 to 3 at one place.
-}
clamped :: Bool -> Kernel T32
clamped near = kernel "clamped" $ do
  y <- buffer @WriteOnly @16 @Word32 "y" 128
  threads @128
  groups 1
  pure $ do
    t <- tgArray @160 @Word32 "t"
    pure $ do
      let place = (simdGroupIndex * (if near then 0 else 1) + minE simdGroupIndex 1) * 32 + simdLane
      tgStore t place threadIndex
      barrier
      store y threadIndex (tgLoad t place)

------------------------------------------------------------------------------
-- The library's

-- | The sum of a row of eight over 64 threads, through a threadgroup reduction.
reduced :: Kernel T32
reduced = kernel "reduced" $ do
  y <- buffer @WriteOnly @16 @Float "y" 1
  threads @64
  groups 1
  pure $ do
    scratch <- tgArray @32 @Float "scratch"
    pure $ do
      s <- tgReduceSum "total" scratch (toF32 threadIndex)
      whenLane (threadIndex .== 0) (store y 0 s)

-- | Two outputs of four splits each, combined.
combined :: Kernel T32
combined = kernel "combined" $ do
  p <- buffer @ReadOnly @16 @Float "p" 8
  y <- buffer @WriteOnly @16 @Float "y" 2
  threads @32
  groups 1
  pure $ do
    partials <- tgArray @8 @Float "partials"
    pure $ do
      whenLane (threadIndex .< 8) (tgStore partials threadIndex (load p threadIndex))
      combineSplits "total" partials 4 2 (store y)

-- | A 16 × 16 tile of C = A · Bᵀ, k in steps of 8, by two SIMD groups.
tiled :: Kernel T32
tiled = tile @128 @128 @128 (GemmTile 16 16 8 2)

-- | 'tiled', A's runs read from a threadgroup array the caller fills first, with a barrier between or none.
stagedTile :: Bool -> Kernel T32
stagedTile waits = kernel "staged_tile" $ do
  k <- dim "k" (DimRange 8 64 8)
  a <- buffer @ReadOnly @16 @BF16 "a" (16 * k)
  b <- buffer @ReadOnly @16 @BF16 "b" (16 * k)
  c <- buffer @WriteOnly @16 @Float "c" 256
  threads @64
  groups 1
  pure $ do
    aRows <- tgArray @1024 @Float "a_rows"
    aSlice <- tgArray @128 @Float "a_slice"
    bSlice <- tgArray @128 @Float "b_slice"
    cTile <- tgArray @128 @Float "c_tile"
    pure $ do
      strided "i" (16 * k) $ \i -> tgStore aRows i (widen @Float (load a i))
      when waits barrier
      gemmTile
        "t"
        (GemmTile 16 16 8 2)
        aSlice
        bSlice
        cTile
        k
        (\r p -> pure [tgLoad aRows (r * k + p + fromIntegral x) | x <- [0 .. 3 :: Int]])
        (\col p -> pure [widen @Float (load b (col * k + p + fromIntegral x)) | x <- [0 .. 7 :: Int]])
        (\r col v -> store c (r * 16 + col) v)

tile :: forall na nb nc. (KnownNat na, KnownNat nb, KnownNat nc, TgArrayOk T32 na Float, TgArrayOk T32 nb Float, TgArrayOk T32 nc Float) => GemmTile -> Kernel T32
tile t = kernel "tile" $ do
  k <- dim "k" (DimRange 8 64 8)
  a <- buffer @ReadOnly @16 @BF16 "a" (16 * k)
  b <- buffer @ReadOnly @16 @BF16 "b" (16 * k)
  c <- buffer @WriteOnly @16 @Float "c" 256
  threads @64
  groups 1
  pure $ do
    aSlice <- tgArray @na @Float "a_slice"
    bSlice <- tgArray @nb @Float "b_slice"
    cTile <- tgArray @nc @Float "c_tile"
    pure $
      gemmTile
        "t"
        t
        aSlice
        bSlice
        cTile
        k
        (\r p -> pure [widen @Float (load a (r * k + p + fromIntegral x)) | x <- [0 .. 3 :: Int]])
        (\col p -> pure [widen @Float (load b (col * k + p + fromIntegral x)) | x <- [0 .. 7 :: Int]])
        (\r col v -> store c (r * 16 + col) v)

tileParams :: [(Name, Value)]
tileParams = [("k", VU32 16)]

tileBuffers :: [(Name, [Value])]
tileBuffers =
  [ ("a", [VBF16 (BF16 0x3f80) | _ <- [1 .. 256 :: Int]])
  , ("b", [VBF16 (BF16 0x3f80) | _ <- [1 .. 256 :: Int]])
  , ("c", replicate 256 (VF32 0))
  ]
