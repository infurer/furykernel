-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The gate, rule by rule, on IR built by hand: it holds for any IR, not
just what the typed surface builds.
-}
module Test.FuryKernel.CheckTest (tests) where

import Data.Bifunctor (first)
import Data.Either (fromLeft, isRight)
import Data.Foldable (for_)
import Data.List (sortOn)
import Data.Text (Text)
import Data.Text qualified as T
import FuryKernel.Build qualified as B
import FuryKernel.Check (Checked)
import FuryKernel.Check qualified as C
import FuryKernel.Finding (Finding (..), Location (..), Rule (..), locatedAt, renderFinding)
import FuryKernel.IR
import FuryKernel.Target (Target, TargetCaps (..), targetCaps)
import FuryKernel.Type (ScalarType (..), scalarSize)
import GHC.Float (castWord32ToFloat)
import Hedgehog (assert, forAll, property, (===))
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Test.FuryKernel.Kernels (scale)
import Test.FuryKernel.Targets (T32, T64, T8)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (assertBool, testCase, (@?=))
import Test.Tasty.Hedgehog (testProperty)
import Unsafe.Coerce (unsafeCoerce)

tests :: TestTree
tests =
  testGroup
    "Check"
    [ testCase "a sound kernel passes" $
        assertBool "scale" (isRight (check (scale @T32)))
    , testCase "uniformity: a per-thread condition declared uniform" $
        findings [SIf RUniform (u32 ThreadInGroup .<. lu 5) [] []]
          @?= [ Finding
                  Uniformity
                  "The condition (@thread < 5u) is declared uniform across the threadgroup but is divergent across lanes."
              ]
    , testCase "uniformity: the typed surface cannot see it, the gate can" $
        either (map (.findingRule)) (const []) (check lying) @?= [Uniformity]
    , testCase "uniformity: a SIMD-group condition is honest under whenSimd" $
        findings [SIf RSimdUniform (u32 SimdGroup .==. lu 0) [SSimd "s" SimdSum TF32 (f32 1)] []] @?= []
    , testCase "uniformity: divergence carried by a var, around a loop" $
        findings
          [ SVar "x" TU32 (ELit (LU32 0))
          , SFor
              RUniform
              "i"
              (ELit (LU32 0))
              (ELit (LU32 4))
              (ELit (LU32 1))
              [SIf RDivergent (u32 SimdLane .==. lu 0) [SAssign "x" (ELit (LU32 1))] []]
          , SIf RUniform (EVar "x" .==. lu 0) [] []
          ]
          @?= [ Finding
                  Uniformity
                  "The condition (x == 0u) is declared uniform across the threadgroup but is divergent across lanes."
              ]
    , testCase "uniformity: divergence carried around a loop, var to var, found only by iterating" $
        findings
          [ SVar "x" TU32 (lu 0)
          , SVar "y" TU32 (lu 0)
          , SVar "z" TU32 (lu 0)
          , SFor RUniform "i" (lu 0) (lu 4) (lu 1) [SAssign "x" (EVar "y"), SAssign "y" (EVar "z"), SAssign "z" (u32 ThreadInGroup)]
          , SIf RUniform (EVar "x" .==. lu 0) [SBarrier] []
          ]
          @?= [ Finding Uniformity "The condition (x == 0u) is declared uniform across the threadgroup but is divergent across lanes."
              , Finding Regions "A barrier where not every thread of the threadgroup arrives."
              ]
    , testCase "uniformity: a loop's bounds per thread, declared uniform" $
        findings [SFor RUniform "i" (lu 0) (u32 ThreadInGroup) (lu 1) []]
          @?= [Finding Uniformity "The bounds of loop i is declared uniform across the threadgroup but is divergent across lanes."]
    , testCase "uniformity: the SIMD group's index is uniform only across the group" $
        findings [SIf RUniform (u32 SimdGroup .==. lu 0) [SBarrier] []]
          @?= [ Finding Uniformity "The condition (@simd_group == 0u) is declared uniform across the threadgroup but is uniform across each SIMD group."
              , Finding Regions "A barrier where not every thread of the threadgroup arrives."
              ]
    , testCase "uniformity: threadgroup memory is per thread to whoever loads it" $
        checkIR (targetCaps @T32) ((base [STgStore "t" (u32 ThreadInGroup) (u32 ThreadInGroup), SIf RUniform (ETgLoad "t" (lu 0) .==. lu 0) [SBarrier] []]) {kernelTgArrays = [TgArray "t" TU32 64]})
          @?= [ Finding Uniformity "The condition (t[0u] == 0u) is declared uniform across the threadgroup but is divergent across lanes."
              , Finding Regions "A barrier where not every thread of the threadgroup arrives."
              ]
    , testCase "uniformity: a threadgroup load alone in its barrier epoch is as uniform as its index" $ do
        epochOf [STgStore "t" (u32 ThreadInGroup) (f32 1), SBarrier, SLet "v" TF32 (ETgLoad "t" (lu 0)), SIf RUniform (f32 0 .<. EVar "v") [SBarrier] [], SBarrier] @?= []
        -- Stored before a barrier at the top of a loop's pass, loaded after it,
        -- a barrier before the next pass stores again.
        epochOf [SFor RUniform "i" (lu 0) (lu 4) (lu 1) [SIf RDivergent (u32 ThreadInGroup .==. lu 0) [STgStore "t" (lu 0) (f32 2)] [], SBarrier, SLet "v" TF32 (ETgLoad "t" (lu 0)), SIf RUniform (f32 0 .<. EVar "v") [] [], SBarrier]] @?= []
    , testCase "uniformity: a threadgroup load divergent where a store to its array can run in its epoch, or its index differs" $ do
        let divergent = ["The condition (f32:0x0 < v) is declared uniform across the threadgroup but is divergent across lanes."]
            load = SLet "v" TF32 (ETgLoad "t" (lu 0))
            branch = SIf RUniform (f32 0 .<. EVar "v") [] []
            byOne i = SIf RDivergent (u32 ThreadInGroup .==. lu 1) [STgStore "t" (lu i) (f32 2)] []
        -- A store to its element after it, before the next barrier.
        epochOf [SBarrier, load, byOne 0, branch] @?= divergent
        -- A store to another element: the rule is the array's.
        epochOf [SBarrier, load, byOne 1, branch] @?= divergent
        -- A store under an if that never runs: the rule is the statements'.
        epochOf [SBarrier, load, SIf RUniform (ELit (LBool False)) [STgStore "t" (lu 0) (f32 2)] [], branch] @?= divergent
        -- A divergent index, with no store.
        epochOf [SBarrier, SLet "v" TF32 (ETgLoad "t" (u32 ThreadInGroup)), branch] @?= divergent
        -- In a loop: a store after the load, before the next pass's barrier;
        -- and one after the barrier, which the next pass's load meets.
        epochOf [SFor RUniform "i" (lu 0) (lu 4) (lu 1) [SBarrier, load, branch, byOne 0]] @?= divergent
        epochOf [SFor RUniform "i" (lu 0) (lu 4) (lu 1) [load, branch, SBarrier, byOne 0]] @?= divergent
    , testCase "uniformity: a fragment stored to a threadgroup array in a load's epoch makes the load divergent" $
        epochOf [SFragVar "f" TF32, SBarrier, SLet "v" TF32 (ETgLoad "t" (lu 0)), SFragStore (FThreadgroup "t") (lu 0) (lu 8) "f", SIf RUniform (f32 0 .<. EVar "v") [] []]
          @?= ["The condition (f32:0x0 < v) is declared uniform across the threadgroup but is divergent across lanes."]
    , testCase "uniformity: a SIMD-group operation's result is uniform only across the group" $
        findings [SSimd "s" SimdSum TU32 (u32 SimdGroup), SFor RUniform "i" (lu 0) (EVar "s") (lu 1) [SBarrier]]
          @?= [ Finding Uniformity "The bounds of loop i is declared uniform across the threadgroup but is uniform across each SIMD group."
              , Finding Regions "A barrier where not every thread of the threadgroup arrives."
              ]
    , testCase "regions: a uniform branch inside a per-lane one is per lane" $
        findings [SIf RDivergent (u32 SimdLane .==. lu 0) [SIf RUniform (ELit (LBool True)) [SBarrier] []] []]
          @?= [Finding Regions "A barrier where not every thread of the threadgroup arrives."]
    , testCase "regions: a strided loop's body is per lane" $
        findings [SFor RDivergent "i" (u32 ThreadInGroup) (lu 100) (lu 64) [SBarrier]]
          @?= [Finding Regions "A barrier where not every thread of the threadgroup arrives."]
    , testCase "typing: no conversion from bool, nor from f32 to i32" $
        findings [SLet "a" TF32 (EConvert TF32 (ELit (LBool True))), SLet "b" TI32 (EConvert TI32 (f32 1.5))]
          @?= [Finding Typing "No conversion from bool to f32.", Finding Typing "No conversion from f32 to i32."]
    , testCase "typing: an integer narrows clamped or wrapped, i32 to i8 and u32 to u8, never by conversion" $
        findings
          [ SLet "a" TI8 (ENarrowInt Clamp TI8 (ELit (LI32 300)))
          , SLet "b" TU8 (ENarrowInt Wrap TU8 (ELit (LU32 300)))
          , SLet "c" TI8 (EConvert TI8 (ELit (LI32 3)))
          , SLet "d" TI8 (ENarrowInt Wrap TI8 (ELit (LU32 3)))
          ]
          @?= [Finding Typing "No conversion from i32 to i8.", Finding Typing "No integer narrowing from u32 to i8."]
    , testCase "a kernel built for one target is refused by another's gate" $
        fromLeft [] (check (unsafeCoerce (scale @T64) :: B.Kernel T8))
          @?= [Finding Target "The kernel was built for T64, whose SIMD width and limits its IR holds; it is checked only for that target, not for T8."]
    , testCase "IR is checked only on the SIMD width it was built for" $
        checkIR (targetCaps @T64) (B.kernelIR (scale @T32))
          @?= [Finding Target "The kernel was built for SIMD groups of 32 lanes, which its body takes as given; T64's have 64."]
    , testCase "a NaN literal is its type's canonical NaN" $ do
        findings [SLet "a" TF32 (ELit (LF32 (castWord32ToFloat 0x7fc00000))), SLet "b" TF16 (ELit (LF16 0x7e00)), SLet "c" TBF16 (ELit (LBF16 0x7fc0))] @?= []
        map (.findingMessage) (findings [SLet "a" TF32 (ELit (LF32 (castWord32ToFloat 0xffc00000))), SLet "b" TF16 (ELit (LF16 0x7e01)), SLet "c" TBF16 (ELit (LBF16 0x7f81))])
          @?= [ "The literal f32:0xffc00000 is a NaN other than its type's canonical one, the quiet NaN with no payload or sign."
              , "The literal f16:0x7e01 is a NaN other than its type's canonical one, the quiet NaN with no payload or sign."
              , "The literal bf16:0x7f81 is a NaN other than its type's canonical one, the quiet NaN with no payload or sign."
              ]
    , testCase "a finding renders with its place, where it has one" $ do
        renderFinding (Finding Typing "No parameter n.") @?= "Typing: No parameter n."
        renderFinding (locatedAt (Location "scale" ["statement 2", "operand 1"]) (Finding Typing "No parameter n."))
          @?= "Typing: No parameter n. (in scale, statement 2, operand 1)"
    , testCase "regions: barriers need every thread, SIMD-group operations every lane" $
        map
          (.findingRule)
          ( findings
              [ SIf RDivergent (u32 SimdLane .==. lu 0) [SBarrier, SSimd "s" SimdSum TF32 (f32 1), SSimdBarrier] []
              , SIf RSimdUniform (u32 SimdGroup .==. lu 0) [SBarrier, SSimdBarrier] []
              ]
          )
          @?= [Regions, Regions, Regions, Regions]
    , testCase "typing" $
        map
          (.findingMessage)
          ( findings
              [ SStore "out" (ELit (LU32 0)) (ELit (LU32 1))
              , SLet "a" TF32 (EVar "nowhere")
              , SAssign "a" (f32 1)
              , SLet "b" TF32 (ELoad "out" (f32 0))
              , SLet "c" TBF16 (EBin Add (ELit (LBF16 0)) (ELit (LBF16 0)))
              ]
          )
          @?= [ "The value stored to out is u32, not f32."
              , "The local nowhere is used outside its scope or before it is bound."
              , "An assignment to a, which is not a var."
              , "The index into buffer out is f32, not u32."
              , "Add of bf16."
              ]
    , testCase "packs: width and alignment against the target" $ do
        let packed buffers body = checkIR (targetCaps @T32) (base body) {kernelBuffers = buffers}
            src = Buffer "src" ReadOnly TF32 16 (ELit (LU32 64)) False
            out8 = Buffer "out" ReadWrite TF32 8 (ELit (LU32 64)) False
        packed [src] [SLoadPack ["a", "b", "c", "d"] TF32 "src" (lu 1)] @?= []
        map
          (.findingMessage)
          (packed [src, out8] [SLoadPack ["a", "b", "c", "d"] TF32 "out" (lu 1), SStorePack "out" (lu 0) (map f32 [1, 2, 3, 4])])
          @?= [ "Buffer out: an alignment of 8 B for elements of 4 B; the least is 16 B."
              , "Pack of 4 elements of 4 B = 16 B from a buffer aligned to 8 B: align it to a multiple of 16 B."
              ]
        map (.findingRule) (packed [src] [SLoadPack (map (T.pack . ("v" <>) . show) [0 .. 7 :: Int]) TF32 "src" (lu 0)])
          @?= [Packs, Packs]
    , testCase "packs: to threadgroup arrays, as wide as a lane loads, of the array's type" $ do
        let tg body = checkIR (targetCaps @T32) (base body) {kernelTgArrays = [TgArray "t" TF32 64]}
        tg [SIf RDivergent (EBin Eq (EBuiltin ThreadInGroup) (lu 0)) [STgStorePack "t" (lu 1) (map f32 [1, 2, 3, 4]), STgStorePack "t" (lu 3) (map f32 [1, 2])] []] @?= []
        map (.findingRule) (tg [STgStorePack "t" (lu 0) (map f32 [1 .. 8])]) @?= [Packs, Packs]
        map (.findingMessage) (tg [STgStorePack "t" (lu 0) [f32 1, ELit (LU32 2)], STgStorePack "u" (lu 0) [f32 1, f32 2]])
          @?= ["A value stored to t is u32, not f32.", "No threadgroup array u."]
    , testCase "fragments widen only into f32" $
        map
          (.findingMessage)
          (findings [SFragVar "f" TF32, SFragVar "h" TBF16, SFragConvert "f" "h", SFragConvert "h" "f"])
          @?= ["The fragment h converted into is not of f32."]
    , testCase "packs: of no elements, asked no alignment" $ do
        let src = Buffer "src" ReadOnly TF32 16 (ELit (LU32 64)) False
        [f.findingMessage | f <- checkIR (targetCaps @T32) (base [SLoadPack [] TF32 "src" (lu 0)]) {kernelBuffers = [src]}, f.findingRule == Packs]
          @?= ["A pack needs at least two elements.", "Pack of 0 elements of 4 B = 0 B, which is not a power of two.", "Pack of 0 elements of 4 B = 0 B from a buffer aligned to 16 B: a pack of no bytes has nothing to align."]
    , testCase "packs: typing" $
        map
          (.findingMessage)
          ( checkIR
              (targetCaps @T32)
              (base [SLoadPack ["a", "b"] TU32 "src" (lu 0), SStorePack "src" (lu 0) [f32 1, ELit (LU32 2)]])
                { kernelBuffers = [Buffer "src" ReadOnly TF32 16 (ELit (LU32 64)) False]
                }
          )
          @?= [ "A pack of u32 loaded from src, a buffer of f32."
              , "A store to the read-only buffer src."
              , "A value stored to src is u32, not f32."
              ]
    , testCase "packs: as uniform as their index, from a buffer nobody writes" $ do
        let uniformIf = SIf RUniform (EBin Lt (EVar "a") (f32 1)) [] []
            withSource acc = (base [SLoadPack ["a", "b"] TF32 "src" (lu 0), uniformIf]) {kernelBuffers = [Buffer "src" acc TF32 16 (ELit (LU32 64)) False]}
        checkIR (targetCaps @T32) (withSource ReadOnly) @?= []
        map (.findingRule) (checkIR (targetCaps @T32) (withSource ReadWrite)) @?= [Uniformity]
    , testCase "fragments: an index the same in every lane of the SIMD group" $
        findings [SFragVar "f" TF32, SFragLoad "f" (FDevice "out") (u32 SimdLane) (lu 8) False]
          @?= [Finding Uniformity "The fragment f's index or stride differs across the lanes of a SIMD group."]
    , testCase "a product's result is its accumulator or another fragment, never one it multiplies" $
        findings [SFragVar "a" TF32, SFragVar "b" TF32, SFragMma "a" "a" "b" "a"]
          @?= [Finding Typing "The product into a multiplies a too: only its accumulator may be its result."]
    , testCase "fragments: a product of either input's type the target doesn't take" $
        [ f.findingMessage
        | f <- checkIR ((targetCaps @T32) {capFragments = [TF16]}) (base [SFragVar "a" TF16, SFragVar "b" TBF16, SFragVar "c" TF32, SFragMma "c" "a" "b" "c"])
        , f.findingRule == Fragments
        ]
          @?= ["T32 takes no fragments of bf16 into a product."]
    , testCase "fragments: only where every lane of the SIMD group is active" $
        findings [SIf RDivergent (u32 SimdLane .==. lu 0) [SFragVar "f" TF32] []]
          @?= [Finding Regions "The fragment operation on f where not every lane of the group is active."]
    , testCase "names" $
        map
          (.findingMessage)
          (findings [SLet "float" TF32 (f32 0), SLet "fk_x" TF32 (f32 0), SLet "_x" TF32 (f32 0), SLet "a__b" TF32 (f32 0), SLet "float4" TF32 (f32 0), SLet "out" TF32 (f32 0)])
          @?= [ "The name float is reserved in a target language."
              , "The name fk_x starts with fk_ in some case, which generated code reserves."
              , "The name _x starts with an underscore, which C++ reserves."
              , "The name a__b holds two underscores in a row, which C++ reserves."
              , "The name float4 is reserved in a target language."
              , "The name out is declared 2 times."
              ]
    , testCase "resources" $ do
        map (.findingRule) (checkIR (targetCaps @T32) (base []) {kernelThreads = 1000}) @?= [Threads]
        map
          (.findingRule)
          (checkIR (targetCaps @T32) (base []) {kernelTgArrays = [TgArray "a" TF32 6000, TgArray "b" TF32 4000]})
          @?= [ThreadgroupMemory]
        map
          (.findingRule)
          ( checkIR
              (targetCaps @T32)
              (base []) {kernelBuffers = [Buffer ("b" <> showT i) ReadOnly TF32 16 (ELit (LU32 1)) False | i <- [1 .. 32 :: Int]]}
          )
          @?= [Buffers]
    , testCase "alignments: a power of two, at least 16 B and the element size" $ do
        let aligned buffers tables = map (.findingMessage) (checkIR (targetCaps @T32) (base []) {kernelBuffers = buffers, kernelTables = tables})
            buf n t align = Buffer n ReadOnly t align (ELit (LU32 64)) False
            tab n t align = Table n t align (ELit (LU32 4)) (ELit (LU32 64))
        aligned [buf "a" TF32 16, buf "b" TU8 32, buf "c" TBF16 4096] [tab "t" TBF16 16] @?= []
        aligned [buf "a" TF32 0, buf "b" TF32 8, buf "c" TBF16 24, buf "d" TU8 1] [tab "t" TBF16 4]
          @?= [ "Buffer a: an alignment of 0 B, which is not a power of two."
              , "Buffer a: an alignment of 0 B for elements of 4 B; the least is 16 B."
              , "Buffer b: an alignment of 8 B for elements of 4 B; the least is 16 B."
              , "Buffer c: an alignment of 24 B, which is not a power of two."
              , "Buffer d: an alignment of 1 B for elements of 1 B; the least is 16 B."
              , "Table t: an alignment of 4 B for elements of 2 B; the least is 16 B."
              ]
    , testCase "threadgroup memory: each array 16-aligned, the total within the limit" $ do
        let arrays as = checkIR (targetCaps @T32) (base []) {kernelTgArrays = [TgArray n TF32 c | (n, c) <- as]}
            over parts total =
              [ Finding ThreadgroupMemory $
                  "Threadgroup memory on T32, each array taking its size rounded up to its alignment: "
                    <> parts
                    <> " = "
                    <> total
                    <> " B; the limit is 32768 B."
              ]
        -- Exactly the limit, with and without padding, and one alignment unit more.
        arrays [("a", 4092), ("b", 4100)] @?= []
        arrays [("a", 4093), ("b", 4096)] @?= []
        arrays [("a", 4092), ("b", 4104)] @?= over "a 16368 B + b 16416 B" "32784"
        arrays [("a", 4093), ("b", 4097)] @?= over "a 16384 B + b 16400 B" "32784"
        arrays [("a", 4093), ("b", 4099)] @?= over "a 16384 B + b 16400 B" "32784"
        tgLayout [TgArray "a" TF32 4093, TgArray "b" TF32 4099] @?= ([("a", 0), ("b", 16384)], 32784)
        tgLayout [TgArray "a" TU8 1, TgArray "b" TF16 3, TgArray "c" TF32 5] @?= ([("a", 0), ("b", 16), ("c", 32)], 64)
        tgLayout [] @?= ([], 0)
    , testProperty "threadgroup layout: each array aligned, apart from the others, within the total" $ property $ do
        arrays <- forAll $ do
          specs <- Gen.list (Range.linear 0 6) ((,) <$> Gen.enumBounded <*> Gen.integral (Range.linear 1 5000))
          pure [TgArray ("a" <> showT i) t c | (i, (t, c)) <- zip [0 :: Int ..] specs]
        let (offsets, total) = tgLayout arrays
            placed = sortOn fst [(at, a) | a <- arrays, Just at <- [lookup a.tgName offsets]]
            size a = a.tgCount * fromIntegral (scalarSize a.tgType)
        length placed === length arrays
        total === sum (map tgBytes arrays)
        for_ placed $ \(at, a) -> do
          assert (at `mod` tgAlign a == 0)
          assert (size a <= tgBytes a && tgBytes a < size a + tgAlign a)
        for_ (zip placed (drop 1 placed)) $ \((at, a), (next, _)) -> assert (at + tgBytes a <= next)
        for_ (take 1 (reverse placed)) $ \(at, a) -> at + tgBytes a === total
    , testCase "the interface: launch sizes over dimensions only" $
        map
          (.findingMessage)
          (checkIR (targetCaps @T32) (base []) {kernelGroups = EBuiltin GroupInGrid})
          @?= ["The grid size uses more than dimensions, u32 literals and + - * / %."]
    , testCase "findings are placed: in their kernel, and in the ifs and loops around what they are about" $ do
        let lane0 = EBin Eq (EBuiltin SimdLane) (lu 0)
            placesOf body = [(f.findingRule, f.findingLocation) | f <- C.checkIR (targetCaps @T32) (base body)]
        placesOf [SIf RUniform (ELit (LBool True)) [SFor RUniform "i" (lu 0) (lu 4) (lu 1) [SLet "x" TF32 (ELit (LBool True))]] []]
          @?= [(Typing, Just (Location "k" ["if true", "for i"]))]
        placesOf [SIf RDivergent lane0 [] [SBarrier]]
          @?= [(Regions, Just (Location "k" ["else of if " <> renderExpr lane0]))]
        placesOf [SLet "float" TF32 (f32 0)] @?= [(Names, Just (Location "k" []))]
    , testCase "a loop that steps by 0" $
        findings [SFor RUniform "i" (lu 0) (lu 4) (lu 0) []] @?= [Finding Typing "The loop i steps by 0, so it never ends."]
    , testCase "a target whose SIMD groups have no lanes" $
        checkIR ((targetCaps @T32) {capSimdWidth = 0}) (base []) {kernelSimdWidth = 0}
          @?= [Finding Target "T32's SIMD groups have no lanes."]
    , testCase "names: Rust's prelude's variants" $
        map (.findingMessage) (findings [SLet "Some" TF32 (f32 0)]) @?= ["The name Some is reserved in a target language."]
    , testCase "names: fk_ in any case, and the targets' library functions" $
        map (.findingMessage) (findings [SLet "FK_x" TF32 (f32 0), SLet "max" TF32 (f32 0), SLet "fma" TF32 (f32 0)])
          @?= ["The name FK_x starts with fk_ in some case, which generated code reserves.", "The name max is reserved in a target language.", "The name fma is reserved in a target language."]
    , testCase "names: the words Metal refuses as identifiers" $
        map (.findingMessage) (findings [SLet "as_type" TF32 (f32 0), SLet "ray_data" TF32 (f32 0), SLet "object_data" TF32 (f32 0)])
          @?= ["The name as_type is reserved in a target language.", "The name ray_data is reserved in a target language.", "The name object_data is reserved in a target language."]
    , testCase "typing: an fma of three operands of one float type" $
        map
          (.findingMessage)
          ( findings
              [ SLet "a" TF32 (EFma (f32 1) (f32 2) (ELit (LF16 0x3C00)))
              , SLet "b" TI32 (EFma (ELit (LI32 1)) (ELit (LI32 2)) (ELit (LI32 3)))
              , SLet "c" TF16 (EFma (ELit (LF16 0x3C00)) (ELit (LF16 0x4000)) (ELit (LF16 0x3C00)))
              , SLet "d" TF32 (EFma (f32 1) (f32 2) (f32 3))
              ]
          )
          @?= ["Operands of an fma differ: f32, f32, f16.", "An fma of i32: only f32 and f16 have one."]
    , testCase "uniformity: an fma as uniform as its least uniform operand" $
        findings [SIf RUniform (EFma (f32 1) (f32 2) (EConvert TF32 (u32 SimdLane)) .<. f32 0) [] []]
          @?= [ Finding
                  Uniformity
                  "The condition (fma(f32:0x3f800000, f32:0x40000000, f32(@lane)) < f32:0x0) is declared uniform across the threadgroup but is divergent across lanes."
              ]
    , testCase "launch sizes: ranges without legal values" $ do
        let ranged r = messages (base []) {kernelParams = [Param "n" TU32 (Dim r)]}
        ranged (DimRange 5 7 8) @?= ["The range of n, 5..7 multiple of 8, holds no value."]
        ranged (DimRange 1 16 0) @?= ["The range of n, 1..16 multiple of 0, has no divisor."]
        ranged (DimRange 9 4 1) @?= ["The range of n, 9..4 multiple of 1, holds no value."]
        ranged (DimRange 4294967296 4294967297 1) @?= ["The range of n, 4294967296..4294967297 multiple of 1, holds no value."]
        ranged (DimRange 1 1099511627776 1) @?= ["The range of n, 1..1099511627776 multiple of 1, reaches past what a u32 holds, 4294967295."]
        ranged (DimRange 0 16 1) @?= []
        ranged (DimRange 1 4096 64) @?= []
    , testCase "launch sizes: refused at the least values that break them" $ do
        grid (DimRange 1 16 1) (lu 8 .-. dimN) @?= ["The grid size 8 - n leaves u32 at n = 9: 8 - n is -1."]
        grid (DimRange 1 4096 1) (lu 64 ./. (dimN .-. lu 1)) @?= ["The grid size 64 / (n - 1) divides by 0 at n = 1: n - 1 is 0."]
        grid (DimRange 0 16 1) (lu 64 .%. dimN) @?= ["The grid size 64 % n divides by 0 at n = 0: n is 0."]
        grid (DimRange 1 16 1) (lu 64 ./. lu 0) @?= ["The grid size 64 / 0 divides by 0."]
        grid (DimRange 1 65536 1) (dimN .*. dimN ./. dimN)
          @?= ["The grid size n * n / n leaves u32 at n = 65536: n * n is 4294967296."]
        grid (DimRange 1 4096 1) (dimN ./. lu 64 .-. lu 1) @?= ["The grid size n / 64 - 1 leaves u32 at n = 1: n / 64 - 1 is -1."]
        messages
          (base [])
            { kernelParams = [Param "n" TU32 (Dim (DimRange 1 2048 1))]
            , kernelBuffers = [b {bufExtent = dimN .*. dimN .*. dimN} | b <- (base []).kernelBuffers]
            }
          @?= ["The extent of out, n * n * n, leaves u32 at n = 1626: n * n * n is 4298942376."]
        messages
          (base [])
            { kernelParams = [Param "m" TU32 (Dim (DimRange 1 16 1)), Param "n" TU32 (Dim (DimRange 1 16 1))]
            , kernelTables = [Table "t" TF32 16 (dimM .-. dimN) (lu 4)]
            }
          @?= ["The number of entries of table t, m - n, leaves u32 at m = 1, n = 2: m - n is -1."]
    , testCase "launch sizes: from the least legal value up, exact where a dimension occurs twice" $ do
        grid (DimRange 1 4096 64) (dimN ./. lu 64 .-. lu 1) @?= []
        grid (DimRange 1 65536 1) (dimN .-. dimN ./. lu 2) @?= []
        grid (DimRange 1 65535 1) (dimN .*. dimN .-. dimN .*. dimN) @?= []
        grid (DimRange 0 4096 1) (dimN .+. lu 0) @?= []
    , testCase "launch sizes: what the search leaves open is refused" $
        grid (DimRange 1 4294967232 1) ((dimN .+. lu 63) ./. lu 64 .*. lu 64 .-. dimN)
          @?= ["The grid size (n + 63) / 64 * 64 - n could not be shown to stay in u32, dividing by no 0, for every legal value of n: 4096 boxes of them leave it open."]
    ]

-- | A small, sound kernel around a body.
base :: [Stmt] -> Kernel
base body =
  Kernel
    { kernelName = "k"
    , kernelThreads = 64
    , kernelSimdWidth = 32
    , kernelGroups = ELit (LU32 1)
    , kernelParams = []
    , kernelBuffers = [Buffer "out" ReadWrite TF32 16 (ELit (LU32 64)) False]
    , kernelTables = []
    , kernelTgArrays = []
    , kernelBody = body
    }

-- | The gate's findings with their places left out, which "findings are placed" tests.
checkIR :: TargetCaps -> Kernel -> [Finding]
checkIR caps = map unplaced . C.checkIR caps

check :: forall t. (Target t) => B.Kernel t -> Either [Finding] (Checked t)
check = first (map unplaced) . C.check

unplaced :: Finding -> Finding
unplaced f = f {findingLocation = Nothing}

findings :: [Stmt] -> [Finding]
findings = checkIR (targetCaps @T32) . base

messages :: Kernel -> [Text]
messages = map (.findingMessage) . checkIR (targetCaps @T32)

-- | The gate's messages for a body over a threadgroup array t of 64 f32.
epochOf :: [Stmt] -> [Text]
epochOf body = messages (base body) {kernelTgArrays = [TgArray "t" TF32 64]}

-- | The gate's findings for a grid over one dimension n.
grid :: DimRange -> Expr -> [Text]
grid r g = messages (base []) {kernelParams = [Param "n" TU32 (Dim r)], kernelGroups = g}

dimM, dimN :: Expr
dimM = EParam "m"
dimN = EParam "n"

(.+.), (.-.), (.*.), (./.), (.%.) :: Expr -> Expr -> Expr
a .+. b = EBin Add a b
a .-. b = EBin Sub a b
a .*. b = EBin Mul a b
a ./. b = EBin Div a b
a .%. b = EBin Rem a b

infixl 6 .+., .-.
infixl 7 .*., ./., .%.

u32 :: Builtin -> Expr
u32 = EBuiltin

f32 :: Float -> Expr
f32 = ELit . LF32

(.<.), (.==.) :: Expr -> Expr -> Expr
a .<. b = EBin Lt a b
a .==. b = EBin Eq a b

lu :: Integer -> Expr
lu = ELit . LU32 . fromInteger

-- | Passes the types, fails the gate: whenU on a per-thread condition.
lying :: B.Kernel T32
lying = B.kernel "lying" $ do
  B.threads @64
  B.groups 1
  pure . pure $ B.whenU (B.threadIndex B..< 5) (pure ())

showT :: Int -> Text
showT = T.pack . show
