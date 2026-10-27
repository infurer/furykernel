-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The program layer: the gate, the f64 evaluator and the planner, on a
decoder layer shaped like a mixture-of-experts transformer's, at a small
size.
-}
module Test.FuryKernel.ProgramTest (
  tests,
  layer,
) where

import Control.Exception qualified as E
import Control.Monad (foldM, void, when)
import Data.Char (isDigit)
import Data.Either (fromLeft, isLeft, isRight)
import Data.List (unzip4, zip4)
import Data.Map.Strict qualified as Map
import Data.Maybe (catMaybes)
import Data.Text (Text)
import Data.Text qualified as T
import FuryKernel.Finding (Finding (..), Location (..), Rule (..))
import FuryKernel.Numerics (narrowBF16, widenBF16)
import FuryKernel.Program
import FuryKernel.Program.Eval
import FuryKernel.Type (ScalarType (..))
import FuryKernel.Variant (splitJoin)
import GHC.Float (double2Float, float2Double)
import Hedgehog (Gen, Property, annotateShow, assert, evalIO, failure, forAll, forAllWith, property, success, (===))
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Numeric.Natural (Natural)
import Test.FuryKernel.PlanCheck (planIssues)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (assertBool, assertFailure, testCase, (@?=))
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
  testGroup
    "program"
    [ testCase "a layer passes the gate" $ either (assertFailure . show) (const (pure ())) (checkProgram (layer False))
    , testCase "the planner cuts a layer where threadgroups must wait for each other" $
        fmap (T.lines . renderPlan . plan defaultPlanOptions) (checkProgram (layer False))
          @?= Right layerPlan
    , testCase "a layer with keys as values plans the same, without the value projection" $
        fmap (map (.stepName) . (.planSteps) . plan defaultPlanOptions) (checkProgram (layer True))
          @?= Right ["s0_gemv", "s1_store", "s2_attend", "s3_combine", "s4_gemv", "s5_gemv", "s6_gemv", "s7_gemv", "s8_x2"]
    , testCase "asked to, the planner stores a layer's GELUs before the anchors that read them" $ do
        let names o = fmap (map (.stepName) . (.planSteps) . plan o) (checkProgram (layer False))
        names defaultPlanOptions {optStoreGelus = True}
          @?= Right ["s0_gemv", "s1_store", "s2_attend", "s3_combine", "s4_gemv", "s5_gemv", "s6_mlp_act", "s7_gemv", "s8_experts_act", "s9_gemv", "s10_x2"]
        -- Gated, the dense MLP's beside the route, in the light step that stores it.
        names defaultPlanOptions {optGated = True, optStoreGelus = True}
          @?= Right ["s0_gemv", "s1_store", "s2_attend", "s3_combine", "s4_gemv", "s5_gemv", "s6_route_ids", "s7_gemv", "s8_gemv", "s9_experts_act", "s10_gemv", "s11_x2"]
    , testCase "gated, the experts wait for a stored route" $
        fmap (map (.stepName) . (.planSteps) . plan defaultPlanOptions {optGated = True}) (checkProgram (layer False))
          @?= Right ["s0_gemv", "s1_store", "s2_attend", "s3_combine", "s4_gemv", "s5_gemv", "s6_route_ids", "s7_gemv", "s8_gemv", "s9_gemv", "s10_x2"]
    , testCase "gated, a route the program outputs is stored beside the experts' route" $
        fmap (T.lines . renderPlan . plan defaultPlanOptions {optGated = True}) (checkProgram predicting)
          @?= Right
            [ "plan predicting"
            , "  step s0_gemv"
            , "    anchor router_logits"
            , "    anchor ahead_logits"
            , "    reads x f32"
            , "    stores router_logits f32"
            , "    stores ahead_logits f32"
            , "  step s1_route_ids"
            , "    recompute route_ids"
            , "    recompute ahead_ids"
            , "    reads router_logits f32"
            , "    reads ahead_logits f32"
            , "    stores route_ids u32"
            , "    stores route_weights f32"
            , "    stores ahead_ids u32"
            , "  step s2_gemv"
            , "    anchor experts_out"
            , "    reads x f32"
            , "    reads route_ids u32"
            , "    stores experts_out f32"
            , "  step s3_experts_sum"
            , "    recompute experts_sum"
            , "    reads route_weights f32"
            , "    reads experts_out f32"
            , "    stores experts_sum f32"
            ]
    , testCase "the head: softcap on the tiles, argmax a light step after them" $
        fmap (T.lines . renderPlan . plan defaultPlanOptions) (checkProgram headProgram)
          @?= Right
            [ "plan head"
            , "  step s0_gemv"
            , "    recompute final_normed"
            , "    anchor logits"
            , "    epilogue capped"
            , "    reads x f32"
            , "    stores capped f32"
            , "  step s1_token"
            , "    recompute token"
            , "    reads capped f32"
            , "    stores token u32"
            ]
    , testCase "a step's kernel splits back at the first _x_ whose right side is a step's name" $ do
        fmap (map (.stepName) . (.planSteps) . plan defaultPlanOptions) (checkProgram embedProgram) @?= Right ["s0_x"]
        let bucket r = maybe False (\ds -> not (T.null ds) && T.all isDigit ds) (T.stripPrefix "m" r)
        (splitJoin bucket "embed_x_s0_x_x_m1" >>= splitJoin isStepName . fst) @?= Just ("embed", "s0_x")
        splitJoin isStepName "embed_x_s0_x" @?= Just ("embed", "s0_x")
        splitJoin isStepName "embed_x_x_s1_store" @?= Just ("embed_x", "s1_store")
        map isStepName ["s0_x", "s12_attend", "s_x", "s0", "s0_", "x0_s"] @?= [True, True, False, False, False, False]
    , testCase "a value its chain reaches by many paths is planned over each node once: 40 doublings" $ do
        let doubled = program "doubled" $ do
              x <- input "x" (wholeRow 4) TF32
              w <- weight "w" [4, 4]
              y <- foldM (\v k -> add ("d" <> T.pack (show k)) v v) x [1 .. 40 :: Int]
              linear "y" w 0 y >>= output "y"
        fmap (map (\s -> (s.stepName, length s.stepWork)) . (.planSteps) . plan defaultPlanOptions) (checkProgram doubled) @?= Right [("s0_gemv", 41)]
    , testCase "a program renders as its sources, nodes and outputs" $
        T.lines (renderProgram small)
          @?= [ "program small"
              , "  source x : input [4/4] f32"
              , "  source w : weight [4, 4] bf16 out_in"
              , "  x : [4/4] f32 = x input(x)"
              , "  y : [4/4] f32 = y linear(w, x) segment 4"
              , "  n : [4/4] f32 = n rmsnorm(y, eps f64:0x3eb0c6f7a0b5ed8d, factor f64:0x3ff0000000000000)"
              , "  output n = n"
              ]
    , testCase "the embedding: one light step" $
        fmap (T.lines . renderPlan . plan defaultPlanOptions) (checkProgram embedProgram)
          @?= Right ["plan embed", "  step s0_x", "    recompute x", "    stores x f32"]
    , testCase "elementwise work on two anchors' tiles waits for them to be stored" $
        fmap (T.lines . renderPlan . plan defaultPlanOptions) (checkProgram twoTiles)
          @?= Right
            [ "plan two_tiles"
            , "  step s0_gemv"
            , "    anchor a"
            , "    anchor b"
            , "    reads x f32"
            , "    stores a f32"
            , "    stores b f32"
            , "  step s1_c"
            , "    recompute c"
            , "    reads a f32"
            , "    reads b f32"
            , "    stores c f32"
            ]
    , testCase "attention as the program's output is combined, in a light step" $
        fmap (T.lines . renderPlan . plan defaultPlanOptions) (checkProgram (attending "attn_out" (output "o")))
          @?= Right
            [ "plan attn_out"
            , "  step s0_store"
            , "    effect store_k"
            , "    effect store_v"
            , "    reads kv f32"
            , "    reads vv f32"
            , "  step s1_attend"
            , "    anchor o"
            , "    reads q f32"
            , "    stores o partials"
            , "  step s2_combine"
            , "    combine o"
            , "    reads o partials"
            , "    stores o f32"
            ]
    , testCase "elementwise work on attention's output waits for its combine" $
        fmap (T.lines . renderPlan . plan defaultPlanOptions) (checkProgram attnCapped)
          @?= Right
            [ "plan attn_capped"
            , "  step s0_store"
            , "    effect store_k"
            , "    effect store_v"
            , "    reads kv f32"
            , "    reads vv f32"
            , "  step s1_attend"
            , "    anchor o"
            , "    reads q f32"
            , "    stores o partials"
            , "  step s2_combine"
            , "    combine o"
            , "    recompute capped"
            , "    reads o partials"
            , "    stores capped f32"
            ]
    , testCase "a store of attention's output into a cache waits for its combine" $
        fmap (T.lines . renderPlan . plan defaultPlanOptions) (checkProgram attnStored)
          @?= Right
            [ "plan attn_stored"
            , "  step s0_store"
            , "    effect store_k"
            , "    effect store_v"
            , "    reads kv f32"
            , "    reads vv f32"
            , "  step s1_attend"
            , "    anchor o"
            , "    reads q f32"
            , "    stores o partials"
            , "  step s2_combine"
            , "    combine o"
            , "    effect store_o"
            , "    reads o partials"
            , "    stores o f32"
            ]
    , testCase "a norm of attention's output before a linear layer: the combine stored once, by a light step" $
        fmap (T.lines . renderPlan . plan defaultPlanOptions) (checkProgram attnNormed)
          @?= Right
            [ "plan attn_normed"
            , "  step s0_store"
            , "    effect store_k"
            , "    effect store_v"
            , "    reads kv f32"
            , "    reads vv f32"
            , "  step s1_attend"
            , "    anchor o"
            , "    reads q f32"
            , "    stores o partials"
            , "  step s2_combine"
            , "    combine o"
            , "    recompute o_normed"
            , "    reads o partials"
            , "    stores o_normed f32"
            , "  step s3_gemv"
            , "    anchor out"
            , "    reads o_normed f32"
            , "    stores out f32"
            ]
    , testCase "attention's output read by a scale the engine reads and by a norm before a linear layer: combined once, the scale made where it is combined" $
        fmap (T.lines . renderPlan . plan defaultPlanOptions) (checkProgram attnMixed)
          @?= Right
            [ "plan attn_mixed"
            , "  step s0_store"
            , "    effect store_k"
            , "    effect store_v"
            , "    reads kv f32"
            , "    reads vv f32"
            , "  step s1_attend"
            , "    anchor o"
            , "    reads q f32"
            , "    stores o partials"
            , "  step s2_combine"
            , "    combine o"
            , "    recompute scaled"
            , "    recompute sub_norm"
            , "    reads o partials"
            , "    stores scaled f32"
            , "    stores sub_norm f32"
            , "  step s3_gemv"
            , "    anchor out"
            , "    reads sub_norm f32"
            , "    stores out f32"
            ]
    , testCase "attention's output stored into a cache after a linear layer reads its norm: combined once, stored by the step that combines it" $
        fmap (T.lines . renderPlan . plan defaultPlanOptions) (checkProgram attnKept)
          @?= Right
            [ "plan attn_kept"
            , "  step s0_store"
            , "    effect store_k"
            , "    effect store_v"
            , "    reads kv f32"
            , "    reads vv f32"
            , "  step s1_attend"
            , "    anchor o"
            , "    reads q f32"
            , "    stores o partials"
            , "  step s2_combine"
            , "    combine o"
            , "    recompute sub_norm"
            , "    reads o partials"
            , "    stores o f32"
            , "    stores sub_norm f32"
            , "  step s3_gemv"
            , "    anchor out"
            , "    reads sub_norm f32"
            , "    stores out f32"
            , "  step s4_store"
            , "    effect store_o"
            , "    reads o f32"
            ]
    , testCase "attention's output read by a scale and a norm: the evaluator over what its kernels stored agrees with its own" evalMixed
    , testCase "every test program's plans read only what earlier steps stored, under every set of options" $
        [ (prog.programName, opts, found)
        | prog <- planned
        , opts <- optionSets
        , let found = either (map (T.pack . show)) (planIssues opts . plan opts) (checkProgram prog)
        , not (null found)
        ]
          @?= []
    , testCase "a ring with room for a step's rows of one sequence past its window: its stores land before the attention" $ do
        let opts = defaultPlanOptions {optSequenceRows = 3}
            pl = plan opts (checked ringProgram)
        [w | s <- pl.planSteps, w <- s.stepWork, isStaging w] @?= []
        planIssues opts pl @?= []
    , testCase "a ring without that room: its stores staged, the attention reading them there, landing after it with its combine" $ do
        let opts = defaultPlanOptions {optSequenceRows = 4}
            pl = plan opts (checked ringProgram)
        renderPlan pl
          @?= T.unlines
            [ "plan ring"
            , "  step s0_store"
            , "    stage store_k"
            , "    stage store_v"
            , "    reads kv f32"
            , "    reads vv f32"
            , "    stores kv staged for k"
            , "    stores vv staged for v"
            , "  step s1_attend"
            , "    anchor o"
            , "    reads q f32"
            , "    reads kv staged for k"
            , "    reads vv staged for v"
            , "    stores o partials"
            , "  step s2_combine"
            , "    land store_k"
            , "    land store_v"
            , "    combine o"
            , "    reads kv staged for k"
            , "    reads vv staged for v"
            , "    reads o partials"
            , "    stores o f32"
            ]
        planIssues opts pl @?= []
    , testCase "a combine step names the caches it lands in, not attention's: it reads only the partials" $ do
        let cachesOf rows = [s.stepCaches | s <- (plan defaultPlanOptions {optSequenceRows = rows} (checked ringProgram)).planSteps, s.stepName == "s2_combine"]
        cachesOf 3 @?= [[]]
        fmap length (cachesOf 4) @?= [2]
    , testCase "what recomputing an operation costs: a gather that sums rows, more than its kind" $
        [nodeCost pooled n.nodeOp | n <- pooled.programNodes, Gather {} <- [n.nodeOp]] @?= [Expensive]
    , testCase "PlanCheck refuses a staged store landing with its attention, and stores first or staged against the ring's room" $ do
        let opts = defaultPlanOptions {optSequenceRows = 4}
            pl = plan opts (checked ringProgram)
            first = plan defaultPlanOptions (checked ringProgram)
            lands = [w | s <- pl.planSteps, w@(Land _) <- s.stepWork]
            -- The lands moved into the attention's step.
            early = pl {planSteps = [s {stepWork = filter (not . isLand) s.stepWork <> [w | s.stepName == "s1_attend", w <- lands]} | s <- pl.planSteps]}
        assertBool "a land in the attention's step passed" (any ("lands in a step no later than o's" `T.isInfixOf`) (planIssues opts early))
        planIssues opts first
          @?= [ "store_k: stored first, though its ring hasn't room for a step's rows of one sequence"
              , "store_v: stored first, though its ring hasn't room for a step's rows of one sequence"
              ]
        planIssues defaultPlanOptions pl
          @?= [ "store_k: staged, though its ring has room for a step's rows of one sequence"
              , "store_v: staged, though its ring has room for a step's rows of one sequence"
              ]
    , testCase "the gate names what is wrong" $
        either (map (.findingRule)) (const []) (checkProgram badProgram) @?= [Shapes, Shapes]
    , testCase "the gate bounds a routed layer's ids by its table's entries" $ do
        findings (routedOver 8 8) @?= []
        findings (routedOver 16 8)
          @?= [Finding Bounds "experts_out: the bound of its ids, 16, is more than the 8 entries of experts."]
        findings (routedOver 16 0)
          @?= [Finding Bounds "experts_out: the bound of its ids, 16, is more than the 0 entries of experts."]
        findings argmaxRouted
          @?= [Finding Bounds "experts_out: the bound of its ids, 64, is more than the 8 entries of experts."]
        findings inputRouted
          @?= [Finding Bounds "experts_out: nothing bounds its ids, which are neither a route's nor an argmax's."]
    , testCase "the gate bounds row integers above 0, and tokens by the rows they gather" $ do
        findings (embedBelow 64) @?= []
        findings (embedBelow 65)
          @?= [Finding Bounds "embedded: the bound of tokens, 65, is more than the 64 rows of embed."]
        findings (embedBelow 0) @?= [Finding Bounds "The bound of tokens, 0, admits no value."]
    , testCase "the gate bounds a store's slots by its ring's, its blocks by the pool's and its positions by its block table" $ do
        findings (storedInto (Ring 2 1 4 2) 2 64) @?= []
        findings (storedInto (Ring 2 1 4 2) 3 64)
          @?= [Finding Bounds "store_k: the bound of slot, 3, is more than the 2 slots of k."]
        findings (storedInto (Paged 4 2 1 2) 4 4) @?= []
        findings (storedInto (Paged 4 2 1 2) 5 4)
          @?= [Finding Bounds "store_k: the bound of blocks, 5, is more than the 4 blocks of k."]
        findings (storedInto (Paged 4 2 1 2) 4 5)
          @?= [Finding Bounds "store_k: the bound of position, 5, is more than the 4 positions of its block table."]
    , testCase "the gate bounds attention's blocks by the pool's, and its positions by its block table" $ do
        findings (pagedAttention 4 4) @?= []
        findings (pagedAttention 5 4)
          @?= [Finding Bounds "o: the bound of blocks, 5, is more than the 4 blocks of k."]
        findings (pagedAttention 4 5)
          @?= [Finding Bounds "o: the bound of position, 5, is more than the 4 positions of its block table."]
    , testCase "the gate holds attention over a ring to the positions it holds: its window, or every position below the bound" $ do
        findings (overRing 2 (Just 2) 64) @?= []
        findings (overRing 2 (Just 3) 64)
          @?= [Finding Bounds "o: its window of 3 positions is longer than the 2 that k's ring holds."]
        findings (overRing 4 Nothing 4) @?= []
        findings (overRing 4 Nothing 5)
          @?= [Finding Bounds "o: with no window it reads every position below the bound of position, 5, more than the 4 that k's ring holds."]
    , testCase "sources keep their names, and the gate refuses one given twice" $ do
        [s.sourceName | s <- weightTwice.programSources] @?= ["x", "embed", "embed"]
        findings weightTwice @?= [Finding Names "The name embed is used 2 times."]
        findings weightAfterValue @?= [Finding Names "The name q_norm is used 2 times."]
    , testCase "the gate refuses ids the planner can't key on: out of program order, or a value made twice or by no node it lists" $ do
        let l = layer False
            n = length l.programNodes
        findings l {programNodes = [nd {nodeId = NodeId (n - 1 - i)} | (i, nd) <- zip [0 ..] l.programNodes]}
          @?= [Finding Structure ("The node ids don't increase in program order: " <> T.pack (show (n - 1)) <> ", then " <> T.pack (show (n - 2)) <> ".")]
        findings small {programNodes = [nd {nodeId = NodeId 0} | nd <- small.programNodes]}
          @?= [Finding Structure "The node ids don't increase in program order: 0, then 0."]
        findings small {programSources = reverse small.programSources}
          @?= [Finding Structure "The source ids don't increase in program order: 1, then 0."]
        findings normOverwrites
          @?= [ Finding Structure "The value ids don't increase in program order: 1, then 1."
              , Finding Structure "The value y is made by 2 nodes."
              , Finding Structure "The value n is made by 2 nodes."
              ]
        findings small {programValues = filter ((/= "y") . (.valueName)) small.programValues}
          @?= [Finding Structure "y makes a value the program doesn't list.", Finding Shapes "n reads a value no earlier node makes."]
        findings unlisted @?= [Finding Structure "ghost makes a value the program doesn't list."]
    , testCase "the gate refuses a second store into a cache" $
        findings storedTwice
          @?= [Finding Structure "store_again stores into k, which store_k already does: a program stores into a cache once."]
    , testCase "the gate refuses caches, heads and rows of no elements, and shapes whose segments don't divide their rows" $ do
        findings (storedInto (Ring 1 1 0 2) 1 1) @?= [Finding Shapes "The cache k has a length of 0."]
        findings (storedInto (Ring 0 1 4 2) 1 4)
          @?= [Finding Shapes "The cache k has 0 slots.", Finding Bounds "store_k: the bound of slot, 1, is more than the 0 slots of k."]
        findings (storedInto (Paged 4 0 1 2) 4 4)
          @?= [ Finding Shapes "The cache k has a block size of 0."
              , Finding Bounds "store_k: the bound of position, 4, is more than the 0 positions of its block table."
              ]
        findings (storedInto (Ring 1 1 4 0) 1 4)
          @?= [Finding Shapes "The cache k has a head dimension of 0.", Finding Shapes "store_k: its value isn't the 1 heads of 0 of k."]
        findings noQueryHeads
          @?= [Finding Shapes "o: it has no query heads.", Finding Shapes "o: its queries aren't its heads."]
        findings unevenInput @?= [Finding Shapes "x: x is [10/3], its segments not dividing its row."]
        findings emptyArgmax @?= [Finding Shapes "x: x is [0/0], a row of no elements."]
        findings emptyRouted @?= [Finding Shapes "x: x is [0/0], a row of no elements."]
    , testCase "the gate places its findings: in the program, and in the node they are about" $
        [f.findingLocation | f <- fromLeft [] (checkProgram (withOp (\case RmsNorm _ g c v -> RmsNorm (1 / 0) g c v; op -> op) small {programName = "small_x_one"}))]
          @?= [Just (Location "small_x_one" []), Just (Location "small_x_one" ["n"])]
    , testCase "the gate refuses caches but of bf16, weights but of a float type or without a layout, parameters that aren't finite, and names generated code can't carry" $ do
        findings (withSource (\case SWeight sh _ l -> SWeight sh TU8 l; k -> k) small) @?= [Finding Shapes "The weight w holds u8, not bf16, f16 or f32."]
        findings (withSource (\case SWeight sh t _ -> SWeight sh t ""; k -> k) small) @?= [Finding Shapes "The weight w names no layout."]
        findings (withSource (\case SCache l _ -> SCache l TU8; k -> k) (storedInto (Ring 1 1 4 2) 1 4)) @?= [Finding Shapes "The cache k holds u8, not bf16."]
        findings (withOp (\case RmsNorm _ g c v -> RmsNorm (1 / 0) g c v; op -> op) small) @?= [Finding Shapes "n: its epsilon isn't positive and finite."]
        findings (withOp (\case RmsNorm e g _ v -> RmsNorm e g (0 / 0) v; op -> op) small) @?= [Finding Shapes "n: its factor isn't finite."]
        findings (withOp (\case Softcap _ v -> Softcap (1 / 0) v; op -> op) headProgram) @?= [Finding Shapes "capped: its cap isn't positive and finite."]
        findings (withOp (\case Attend a q to -> Attend a {attnScale = 0} q to; op -> op) (attending "attn_out" (output "o")))
          @?= [Finding Shapes "o: its scale isn't positive and finite."]
        findings (withOp (\case Gather t tok _ -> Gather t tok (0 / 0); op -> op) embedProgram) @?= [Finding Shapes "x: its factor isn't finite."]
        findings small {programName = "small_x_one"} @?= [Finding Names "The program's name small_x_one holds _x_, which joins it to its steps' names."]
        findings small {programOutputs = [o {outputName = "float"} | o <- small.programOutputs]} @?= [Finding Names "The name float is reserved in a target language."]
    , testProperty "the evaluator returns, never throws, whatever a checked program is bound to" evalTotal
    , testCase "the evaluator refuses row integers outside their bounds" $
        fmap (.evalOutputs) (evaluate (checked (embedBelow 4)) (bindings 1) {bindRowInts = Map.fromList [("tokens", [[9]])], bindWeights = Map.fromList [("embed", replicate 512 1)]})
          @?= Left "tokens: row 0 holds 9, which isn't below its bound of 4."
    , testCase "the evaluator: a linear layer, a norm and a softcap" evalSmall
    , testCase "the evaluator: attention over one stored position returns its value" evalAttention
    , testCase "the evaluator, given what kernels stored: a rounding the bound allows kept, a further one refused" evalStored
    , testCase "the evaluator's bounds refuse a store of a well-conditioned value a step or two off, or truncated" boundsRefuse
    , testCase "the evaluator's bounds take binary32's stores through cancellation, at subnormal ties and the overflow tie, and NaN" boundsAccept
    , testCase "the evaluator's bounds take a rotary embedding's binary32 stores far into the context, its angles products or from tables" boundsLongRope
    , testProperty "the evaluator's bounds take the stores of long binary32 dot products, summed in order or pairwise" boundsLongDots
    , testProperty "the evaluator's bounds refuse the bf16 steps beside a well-conditioned dot product's rounding" boundsTight
    , testProperty "the evaluator: a step's rows attend as if they ran one at a time, through a ring shorter than the step" evalCausal
    , testCase "the evaluator, given what kernels staged: each row's staged values read, the cache holding the last" evalStaged
    , testCase "the evaluator: routes take the top k, ties to the lower index" evalRoute
    , testCase "the evaluator binds caches sparsely: two blocks of a paged pool of 2^33 elements, one slot of a ring" evalSparse
    , testCase "the evaluator: a concatenation's row, the first's then the second's" evalConcat
    , testCase "the gate refuses to concatenate segments of other widths" $
        assertBool "a concatenation of other segments passed" (isLeft (checkProgram badConcat))
    , testCase "the evaluator: a rotary embedding of two axes turns each half of a head by its own position" evalAxialRope
    , testCase "the evaluator: a gather of several rows sums them, times its factor" evalPooledGather
    , testCase "the evaluator: standardizing scales, subtracts the bias and scales again" evalStandardize
    , testCase "the gate refuses a rotary embedding of two axes over one position, and weights not a row wide" $ do
        assertBool "two axes over one position passed" (isLeft (checkProgram (axialRope 1)))
        assertBool "a bias of another width passed" (isLeft (checkProgram (standardized 3)))
        findings (withOp (\case Standardize b s _ v -> Standardize b s (1 / 0) v; op -> op) (standardized 2)) @?= [Finding Shapes "y: its factor isn't finite."]
    , testCase "the planner stores a gather of several rows before the linear layer that reads it" $
        fmap (map (.stepName) . (.planSteps) . plan defaultPlanOptions) (checkProgram pooled)
          @?= Right ["s0_pooled", "s1_gemv"]
    , testProperty "bf16 rounding of a double agrees with narrowing a float" roundingAgrees
    ]

------------------------------------------------------------------------------
-- A small layer

-- | Hidden 32, four query heads of 8 over two KV heads, 8 experts of 16 of which a row takes 2.
layer :: Bool -> Program
layer kEqV = program (if kEqV then "layer_kv" else "layer") $ do
  x <- input "x" (wholeRow hidden) TF32
  pos <- rowInts "position" 1 64
  slot <- rowInts "slot" 1 2
  preAttn <- weight "pre_attn_norm" [hidden]
  wq <- weight "q_proj" [4 * 8, hidden]
  wk <- weight "k_proj" [2 * 8, hidden]
  wv <- weight "v_proj" [2 * 8, hidden]
  qNorm <- weight "q_norm" [8]
  kNorm <- weight "k_norm" [8]
  wo <- weight "o_proj" [hidden, 4 * 8]
  postAttn <- weight "post_attn_norm" [hidden]
  preMlp <- weight "pre_mlp_norm" [hidden]
  gateW <- weight "gate_proj" [48, hidden]
  upW <- weight "up_proj" [48, hidden]
  down <- weight "down_proj" [hidden, 48]
  postMlp <- weight "post_mlp_norm" [hidden]
  routerScale <- weight "router_scale" [hidden]
  router <- weight "router_proj" [8, hidden]
  expertScale <- weight "expert_scale" [8]
  preMoe <- weight "pre_moe_norm" [hidden]
  expertsGateUp <- table "experts_gate_up" 8 [2 * 16, hidden]
  expertsDown <- table "experts_down" 8 [hidden, 16]
  postMoe <- weight "post_moe_norm" [hidden]
  postFfn <- weight "post_ffn_norm" [hidden]
  scalar <- weight "layer_scalar" [1]
  kCache <- cache "k_cache" (Ring 2 2 12 8)
  vCache <- cache "v_cache" (Ring 2 2 12 8)
  -- Attention
  a <- rmsnorm "attn_in" eps (Just preAttn) 1 x
  q <- linear "q" wq 8 a
  k <- linear "k" wk 8 a
  v <- if kEqV then pure k else linear "v" wv 8 a
  qn <- rmsnorm "q_normed" eps (Just qNorm) 1 q
  kn <- rmsnorm "k_normed" eps (Just kNorm) 1 k
  vn <- rmsnorm "v_normed" eps Nothing 1 v
  qr <- rope "q_rotated" (RopeSpec 10000 4 1 RopeHalves) pos qn
  kr <- rope "k_rotated" (RopeSpec 10000 4 1 RopeHalves) pos kn
  kvStore "store_k" StoreAt {storeCache = kCache, storePositions = pos, storeSequences = slot} kr
  kvStore "store_v" StoreAt {storeCache = vCache, storePositions = pos, storeSequences = slot} vn
  o <- attend "attended" (AttnSpec 4 2 8 (Just 6) 1) qr AttendTo {attendKeys = kCache, attendValues = vCache, attendPositions = pos, attendSequences = slot}
  out <- linear "attn_out" wo 0 o
  outNormed <- rmsnorm "attn_out_normed" eps (Just postAttn) 1 out
  x1 <- add "x1" x outNormed
  -- The dense MLP and the experts
  b <- rmsnorm "mlp_in" eps (Just preMlp) 1 x1
  gate <- linear "gate" gateW 0 b
  up <- linear "up" upW 0 b
  r <- rmsnorm "router_in" eps (Just routerScale) (1 / sqrt 32) x1
  logits <- linear "router_logits" router 0 r
  g <- geluMul "mlp_act" gate up
  (ids, ws) <- route "route" 2 (Just expertScale) logits
  c <- rmsnorm "moe_in" eps (Just preMoe) 1 x1
  d <- linear "mlp_out" down 0 g
  gue <- linearRouted "experts_gate_up_out" expertsGateUp ids c
  ge <- gatedGelu "experts_act" gue
  de <- linearRouted "experts_out" expertsDown ids ge
  h1 <- rmsnorm "mlp_out_normed" eps (Just postMlp) 1 d
  s <- combineRanks "experts_sum" ws de
  h2 <- rmsnorm "moe_out_normed" eps (Just postMoe) 1 s
  hs <- add "ffn_sum" h1 h2
  f <- rmsnorm "ffn_out" eps (Just postFfn) 1 hs
  x2 <- add "x2_unscaled" x1 f
  y <- scaleBy "x2" scalar x2
  output "x" y
 where
  hidden = 32 :: Natural
  eps = 1e-6

-- | The plan of 'layer': nine steps.
layerPlan :: [Text]
layerPlan =
  [ "plan layer"
  , "  step s0_gemv"
  , "    recompute attn_in"
  , "    anchor q"
  , "    anchor k"
  , "    anchor v"
  , "    reads x f32"
  , "    stores q f32"
  , "    stores k f32"
  , "    stores v f32"
  , "  step s1_store"
  , "    recompute k_normed"
  , "    recompute v_normed"
  , "    recompute k_rotated"
  , "    effect store_k"
  , "    effect store_v"
  , "    reads k f32"
  , "    reads v f32"
  , "  step s2_attend"
  , "    recompute q_normed"
  , "    recompute q_rotated"
  , "    anchor attended"
  , "    reads q f32"
  , "    stores attended partials"
  , "  step s3_combine"
  , "    combine attended"
  , "    reads attended partials"
  , "    stores attended f32"
  , "  step s4_gemv"
  , "    anchor attn_out"
  , "    reads attended f32"
  , "    stores attn_out f32"
  , "  step s5_gemv"
  , "    recompute attn_out_normed"
  , "    recompute x1"
  , "    recompute mlp_in"
  , "    anchor gate"
  , "    anchor up"
  , "    recompute router_in"
  , "    anchor router_logits"
  , "    reads x f32"
  , "    reads attn_out f32"
  , "    stores x1 f32"
  , "    stores gate f32"
  , "    stores up f32"
  , "    stores router_logits f32"
  , "  step s6_gemv"
  , "    recompute mlp_act"
  , "    recompute route_ids"
  , "    recompute moe_in"
  , "    anchor mlp_out"
  , "    anchor experts_gate_up_out"
  , "    reads x1 f32"
  , "    reads gate f32"
  , "    reads up f32"
  , "    reads router_logits f32"
  , "    stores route_ids u32"
  , "    stores route_weights f32"
  , "    stores mlp_out f32"
  , "    stores experts_gate_up_out f32"
  , "  step s7_gemv"
  , "    recompute experts_act"
  , "    anchor experts_out"
  , "    reads route_ids u32"
  , "    reads experts_gate_up_out f32"
  , "    stores experts_out f32"
  , "  step s8_x2"
  , "    recompute mlp_out_normed"
  , "    recompute experts_sum"
  , "    recompute moe_out_normed"
  , "    recompute ffn_sum"
  , "    recompute ffn_out"
  , "    recompute x2_unscaled"
  , "    recompute x2"
  , "    reads x1 f32"
  , "    reads route_weights f32"
  , "    reads mlp_out f32"
  , "    reads experts_out f32"
  , "    stores x2 f32"
  ]

-- | The final norm, the tied unembedding, the softcap and the greedy token.
headProgram :: Program
headProgram = program "head" $ do
  x <- input "x" (wholeRow 32) TF32
  norm <- weight "final_norm" [32]
  embed <- weight "embed" [64, 32]
  n <- rmsnorm "final_normed" 1e-6 (Just norm) 1 x
  logits <- linear "logits" embed 0 n
  capped <- softcap "capped" 30 logits
  token <- argmax "token" capped
  output "token" token

-- | Two linear layers of one input, and the sum of their outputs.
twoTiles :: Program
twoTiles = program "two_tiles" $ do
  x <- input "x" (wholeRow 32) TF32
  wa <- weight "wa" [16, 32]
  wb <- weight "wb" [16, 32]
  a <- linear "a" wa 0 x
  b <- linear "b" wb 0 x
  c <- add "c" a b
  output "c" c

-- | The programs the planner's tests plan.
planned :: [Program]
planned =
  [ layer False
  , layer True
  , headProgram
  , twoTiles
  , embedProgram
  , attending "attn_out" (output "o")
  , attnCapped
  , attnStored
  , attnNormed
  , attnMixed
  , attnKept
  , predicting
  , ringProgram
  , pooled
  ]

isStaging, isLand :: Work -> Bool
isStaging = \case
  Stage _ -> True
  Land _ -> True
  _ -> False
isLand = \case
  Land _ -> True
  _ -> False

-- | Every set of the planner's options.
optionSets :: [PlanOptions]
optionSets =
  [ defaultPlanOptions {optHorizontal = h, optGated = g, optStoreGelus = s, optSequenceRows = r}
  | h <- [True, False]
  , g <- [False, True]
  , s <- [False, True]
  , r <- [1, 64]
  ]

{- | Attention over two stored positions of a ring, whose output @rest@
takes: its tiles are partials, which only its combine reads.
-}
attending :: Text -> (Val -> P ()) -> Program
attending name rest = program name $ do
  q <- input "q" (Shape 4 2) TF32
  pos <- rowInts "position" 1 4
  slot <- rowInts "slot" 1 2
  k <- cache "k" (Ring 2 1 4 2)
  v <- cache "v" (Ring 2 1 4 2)
  kv <- input "kv" (Shape 2 2) TF32
  vv <- input "vv" (Shape 2 2) TF32
  kvStore "store_k" StoreAt {storeCache = k, storePositions = pos, storeSequences = slot} kv
  kvStore "store_v" StoreAt {storeCache = v, storePositions = pos, storeSequences = slot} vv
  o <- attend "o" (AttnSpec 2 1 2 Nothing 1) q AttendTo {attendKeys = k, attendValues = v, attendPositions = pos, attendSequences = slot}
  rest o

-- | A softcap of attention's output, which the program outputs.
attnCapped :: Program
attnCapped = attending "attn_capped" $ \o -> do
  y <- softcap "capped" 50 o
  output "y" y

-- | Attention's output stored into a cache of its own.
attnStored :: Program
attnStored = attending "attn_stored" $ \o -> do
  c <- cache "o_cache" (Ring 2 2 4 2)
  pos <- rowInts "o_position" 1 64
  slot <- rowInts "o_slot" 1 2
  kvStore "store_o" StoreAt {storeCache = c, storePositions = pos, storeSequences = slot} o
  output "o" o

-- | A norm of attention's output before the linear layer that reads it.
attnNormed :: Program
attnNormed = attending "attn_normed" $ \o -> do
  w <- weight "o_proj" [4, 4]
  n <- rmsnorm "o_normed" 1e-6 Nothing 1 o
  out <- linear "out" w 0 n
  output "out" out

-- | Attention's output scaled for the engine, and normed before a linear layer.
attnMixed :: Program
attnMixed = attending "attn_mixed" $ \o -> do
  s <- weight "attn_scale" [1]
  w <- weight "o_proj" [4, 4]
  y <- scaleBy "scaled" s o
  n <- rmsnorm "sub_norm" 1e-6 Nothing 1 o
  out <- linear "out" w 0 n
  output "y" y
  output "out" out

-- | Attention's output normed before a linear layer, and stored into a cache after it.
attnKept :: Program
attnKept = attending "attn_kept" $ \o -> do
  w <- weight "o_proj" [4, 4]
  n <- rmsnorm "sub_norm" 1e-6 Nothing 1 o
  out <- linear "out" w 0 n
  c <- cache "o_cache" (Ring 2 2 4 2)
  pos <- rowInts "o_position" 1 64
  slot <- rowInts "o_slot" 1 2
  kvStore "store_o" StoreAt {storeCache = c, storePositions = pos, storeSequences = slot} o
  output "out" out

{- | Experts of a route, and a second route the engine reads: a prediction,
say, of a later layer's experts.
-}
predicting :: Program
predicting = program "predicting" $ do
  x <- input "x" (wholeRow 32) TF32
  router <- weight "router_proj" [8, 32]
  ahead <- weight "ahead_proj" [8, 32]
  experts <- table "experts" 8 [16, 32]
  logits <- linear "router_logits" router 0 x
  aheadLogits <- linear "ahead_logits" ahead 0 x
  (ids, ws) <- route "route" 2 Nothing logits
  (aheadIds, _) <- route "ahead" 2 Nothing aheadLogits
  y <- linearRouted "experts_out" experts ids x
  s <- combineRanks "experts_sum" ws y
  output "y" s
  output "ahead" aheadIds

-- | Tokens to the residual stream, scaled.
embedProgram :: Program
embedProgram = program "embed" $ do
  tokens <- rowInts "tokens" 1 64
  embed <- weight "embed" [64, 32]
  x <- gather "x" embed tokens 5.65625
  output "x" x

-- | A linear layer whose weight has the wrong width, and a norm whose gamma does.
badProgram :: Program
badProgram = program "bad" $ do
  x <- input "x" (wholeRow 16) TF32
  w <- weight "w" [8, 12]
  g <- weight "g" [4]
  y <- linear "y" w 0 x
  z <- rmsnorm "z" 1e-6 (Just g) 1 y
  output "z" z

{- | What the gate finds wrong with a program: nothing, if it passes.
| The gate's findings, their places left out, which "the gate places its findings" tests.
-}
findings :: Program -> [Finding]
findings = map (\f -> f {findingLocation = Nothing}) . fromLeft [] . checkProgram

-- | The program with every node's operation mapped.
withOp :: (Op -> Op) -> Program -> Program
withOp f p = p {programNodes = [n {nodeOp = f n.nodeOp} | n <- p.programNodes]}

-- | The program with every source's kind mapped.
withSource :: (SourceKind -> SourceKind) -> Program -> Program
withSource f p = p {programSources = [s {sourceKind = f s.sourceKind} | s <- p.programSources]}

-- | A route over @logits@ logits into a table of @entries@ experts.
routedOver :: Natural -> Natural -> Program
routedOver logits entries = program "routed" $ do
  x <- input "x" (wholeRow 4) TF32
  router <- weight "router" [logits, 4]
  experts <- table "experts" entries [2, 4]
  (ids, weights) <- linear "router_logits" router 0 x >>= route "route" 2 Nothing
  y <- linearRouted "experts_out" experts ids x
  combineRanks "experts_sum" weights y >>= output "y"

-- | An argmax over 64 logits as the ids into a table of 8 experts.
argmaxRouted :: Program
argmaxRouted = program "argmax_routed" $ do
  x <- input "x" (wholeRow 4) TF32
  w <- weight "w" [64, 4]
  experts <- table "experts" 8 [2, 4]
  ids <- linear "logits" w 0 x >>= argmax "top"
  linearRouted "experts_out" experts ids x >>= output "y"

-- | The engine's u32 as the ids into a table of 8 experts.
inputRouted :: Program
inputRouted = program "input_routed" $ do
  x <- input "x" (wholeRow 4) TF32
  ids <- input "ids" (wholeRow 2) TU32
  experts <- table "experts" 8 [2, 4]
  linearRouted "experts_out" experts ids x >>= output "y"

-- | An embedding of tokens below @bound@ from 64 rows.
embedBelow :: Natural -> Program
embedBelow bound = program "embed_below" $ do
  tokens <- rowInts "tokens" 1 bound
  embed <- weight "embed" [64, 8]
  gather "embedded" embed tokens 1 >>= output "x"

-- | Stores of one value into one cache, twice.
storedTwice :: Program
storedTwice = program "stored_twice" $ do
  x <- input "x" (Shape 2 2) TF32
  pos <- rowInts "position" 1 4
  slot <- rowInts "slot" 1 1
  k <- cache "k" (Ring 1 1 4 2)
  kvStore "store_k" StoreAt {storeCache = k, storePositions = pos, storeSequences = slot} x
  kvStore "store_again" StoreAt {storeCache = k, storePositions = pos, storeSequences = slot} x

{- | A store into a cache of the layout, through a slot or a block table of
two below @sequences@, at positions below @positions@.
-}
storedInto :: CacheLayout -> Natural -> Natural -> Program
storedInto layout sequences positions = program "stored_into" $ do
  x <- input "x" (Shape 2 2) TF32
  pos <- rowInts "position" 1 positions
  sq <- case layout of
    Ring {} -> rowInts "slot" 1 sequences
    Paged {} -> rowInts "blocks" 2 sequences
  k <- cache "k" layout
  kvStore "store_k" StoreAt {storeCache = k, storePositions = pos, storeSequences = sq} x

{- | Attention over a pool of four blocks of two positions, through a block
table of two below @blocks@, from positions below @positions@.
-}
pagedAttention :: Natural -> Natural -> Program
pagedAttention blocks positions = program "paged" $ do
  q <- input "q" (Shape 4 2) TF32
  pos <- rowInts "position" 1 positions
  blockTable <- rowInts "blocks" 2 blocks
  k <- cache "k" (Paged 4 2 1 2)
  v <- cache "v" (Paged 4 2 1 2)
  attend "o" (AttnSpec 2 1 2 Nothing 1) q AttendTo {attendKeys = k, attendValues = v, attendPositions = pos, attendSequences = blockTable} >>= output "o"

{- | Attention over a ring of @len@ positions, from positions below
@bound@, with the window given.
-}
overRing :: Natural -> Maybe Natural -> Natural -> Program
overRing len window bound = program "over_ring" $ do
  q <- input "q" (Shape 2 2) TF32
  pos <- rowInts "position" 1 bound
  slot <- rowInts "slot" 1 1
  k <- cache "k" (Ring 1 1 len 2)
  v <- cache "v" (Ring 1 1 len 2)
  attend "o" (AttnSpec 1 1 2 window 1) q AttendTo {attendKeys = k, attendValues = v, attendPositions = pos, attendSequences = slot} >>= output "o"

-- | A weight named twice.
weightTwice :: Program
weightTwice = program "twice" $ do
  x <- input "x" (wholeRow 4) TF32
  w1 <- weight "embed" [4, 4]
  y <- linear "y" w1 0 x
  w2 <- weight "embed" [4, 4]
  linear "z" w2 0 y >>= output "z"

-- | A weight named as an earlier value.
weightAfterValue :: Program
weightAfterValue = program "shadow" $ do
  x <- input "x" (wholeRow 4) TF32
  w <- weight "w" [4, 4]
  y <- linear "q_norm" w 0 x
  g <- weight "q_norm" [4]
  rmsnorm "z" 1e-6 (Just g) 1 y >>= output "z"

-- | A linear layer and a norm of it.
small :: Program
small = program "small" $ do
  x <- input "x" (wholeRow 4) TF32
  w <- weight "w" [4, 4]
  y <- linear "y" w 0 x
  rmsnorm "n" 1e-6 Nothing 1 y >>= output "n"

-- | 'small', its norm writing the linear layer's value id.
normOverwrites :: Program
normOverwrites =
  small
    { programNodes = [if nd.nodeName == "n" then nd {nodeOutputs = [yId], nodeOp = normOf nd.nodeOp} else nd | nd <- small.programNodes]
    , programValues = [if v.valueId == nId then v {valueId = yId} else v | v <- small.programValues]
    , programOutputs = [o {outputValue = yId} | o <- small.programOutputs]
    }
 where
  idOf name = case [v.valueId | v <- small.programValues, v.valueName == name] of
    v : _ -> v
    [] -> ValueId (-1)
  yId = idOf "y"
  nId = idOf "n"
  normOf = \case
    RmsNorm e g f _ -> RmsNorm e g f yId
    o -> o

-- | 'small' and a linear layer of a weight 7 wide over x of 4, its output unlisted.
unlisted :: Program
unlisted =
  small
    { programSources = small.programSources <> [Source (SourceId 9) "bad_w" (SWeight [4, 7] TBF16 "out_in")]
    , programNodes = small.programNodes <> [Node (NodeId 9) "ghost" (Linear (SourceId 9) 3 (ValueId 0)) [ValueId 99]]
    }

-- | Attention of no query heads over two key and value heads.
noQueryHeads :: Program
noQueryHeads = program "no_heads" $ do
  q <- input "q" (Shape 4 2) TF32
  pos <- rowInts "position" 1 4
  slot <- rowInts "slot" 1 1
  k <- cache "k" (Ring 1 2 4 2)
  v <- cache "v" (Ring 1 2 4 2)
  attend "o" (AttnSpec 0 2 2 Nothing 1) q AttendTo {attendKeys = k, attendValues = v, attendPositions = pos, attendSequences = slot} >>= output "o"

-- | An input of 10 elements in segments of 3.
unevenInput :: Program
unevenInput = program "uneven" $ do
  x <- input "x" (Shape 10 3) TF32
  w <- weight "w" [4, 10]
  linear "y" w 0 x >>= output "y"

-- | The argmax of a row of no elements.
emptyArgmax :: Program
emptyArgmax = program "empty_argmax" $ do
  x <- input "x" (wholeRow 0) TF32
  argmax "t" x >>= output "t"

-- | A routed layer whose ids are the argmax of a row of no elements, into a table of none.
emptyRouted :: Program
emptyRouted = program "empty_routed" $ do
  x <- input "x" (wholeRow 0) TF32
  experts <- table "experts" 0 [2, 4]
  ids <- argmax "top" x
  w <- input "w" (wholeRow 4) TF32
  linearRouted "experts_out" experts ids w >>= output "y"

------------------------------------------------------------------------------
-- The evaluator

{- | The evaluator over random bindings of a test program's sources: each
bound or not, of the size it declares or one off, row integers below their
bound or at it, the step's rows as given or one more.
-}
evalTotal :: Property
evalTotal = property $ do
  prog <- forAllWith (T.unpack . (.programName)) (Gen.element (planned <> [routedOver 8 8, axialRope 2, standardized 2, overRing 4 Nothing 4]))
  m <- forAll (Gen.int (Range.linear 0 3))
  b <- forAllWith describe (bindingsOf m prog)
  r <- evalIO (E.try (E.evaluate (forced (evaluate (checked prog) b))))
  case r of
    Left (e :: E.SomeException) -> annotateShow e >> failure
    Right _ -> success
 where
  forced = \case
    Left e -> T.length e
    Right ev -> length (show (Map.toList ev.evalValues, Map.toList ev.evalBounds, Map.toList ev.evalOutputs, Map.toList ev.evalCaches))
  describe b' =
    show
      ( b'.bindRows
      , Map.map (map length) b'.bindInputs
      , Map.map (map length) b'.bindRowInts
      , Map.map length b'.bindWeights
      , Map.map (map length) b'.bindTables
      , Map.map (Map.map length) b'.bindCaches
      )

bindingsOf :: Int -> Program -> Gen Bindings
bindingsOf m prog = do
  rows <- Gen.frequency [(9, pure m), (1, pure (m + 1))]
  let sources = prog.programSources
      named :: Source -> Gen a -> Gen (Maybe (Text, a))
      named src g = Gen.frequency [(9, Just . (src.sourceName,) <$> g), (1, pure Nothing)]
      size :: Int -> Gen Int
      size n = Gen.frequency [(8, pure n), (1, pure (n + 1)), (1, pure (max 0 (n - 1)))]
      list n = size n >>= \k -> Gen.list (Range.singleton k) (Gen.double (Range.linearFrac (-2) 2))
      perRow :: Gen a -> Gen [a]
      perRow = Gen.list (Range.singleton rows)
  ins <- sequence [named src (perRow (list (fromIntegral sh.shapeWidth))) | src <- sources, SInput sh _ <- [src.sourceKind]]
  ints <-
    sequence
      [ named src (perRow (size (fromIntegral k) >>= \c -> Gen.list (Range.singleton c) (Gen.integral (Range.linear 0 (fromIntegral bound)))))
      | src <- sources
      , SRowInts k bound <- [src.sourceKind]
      ]
  ws <- sequence [named src (list (fromIntegral (product dims))) | src <- sources, SWeight dims _ _ <- [src.sourceKind]]
  ts <-
    sequence
      [ named src (size (fromIntegral n) >>= \c -> Gen.list (Range.singleton c) (list (fromIntegral (product dims))))
      | src <- sources
      , STable n dims _ _ <- [src.sourceKind]
      ]
  -- Some of a cache's slots or blocks, one past them at times, each of
  -- its size or one off.
  cs <-
    sequence
      [ named src (Map.fromList <$> traverse (\u -> (u,) <$> list (unitElements layout)) chosen)
      | src <- sources
      , SCache layout _ <- [src.sourceKind]
      , let chosen = [0 .. min 2 (fromIntegral (unitsIn layout))]
      ]
  pure (Bindings m (Map.fromList (catMaybes ins)) (Map.fromList (catMaybes ints)) (Map.fromList (catMaybes ws)) (Map.fromList (catMaybes ts)) (Map.fromList (catMaybes cs)))
 where
  unitElements = \case
    Ring _ heads len d -> fromIntegral (heads * len * d)
    Paged _ blockSize heads d -> fromIntegral (blockSize * heads * d)
  unitsIn = \case
    Ring n _ _ _ -> n
    Paged n _ _ _ -> n

checked :: Program -> CheckedProgram
checked p = either (error . show) id (checkProgram p)

-- | A cache's elements, as its slots or blocks of @n@ each, from the first.
slots :: Int -> [Double] -> Map.Map Int [Double]
slots n xs = Map.fromList (zip [0 ..] (takeWhile (not . null) (map (take n) (iterate (drop n) xs))))

bindings :: Int -> Bindings
bindings m = Bindings m Map.empty Map.empty Map.empty Map.empty Map.empty

evalSmall :: IO ()
evalSmall = do
  let p = program "small" $ do
        x <- input "x" (wholeRow 2) TF32
        w <- weight "w" [2, 2]
        y <- linear "y" w 0 x
        n <- rmsnorm "n" 1e-6 Nothing 1 y
        c <- softcap "c" 30 n
        output "c" c
      b =
        (bindings 1)
          { bindInputs = Map.fromList [("x", [[1, 2]])]
          , bindWeights = Map.fromList [("w", [1, 0, 1, 1])]
          }
  -- y = (1, 3), mean square 5, normed (1, 3) / sqrt (5 + eps).
  let s = (5 + 1e-6) ** (-0.5)
  fmap (Map.lookup "c" . (.evalOutputs)) (evaluate (checked p) b)
    @?= Right (Just [[30 * tanh (1 * s / 30), 30 * tanh (3 * s / 30)]])

evalAttention :: IO ()
evalAttention = do
  let p = program "attention" $ do
        q <- input "q" (Shape 4 2) TF32
        pos <- rowInts "position" 1 4
        slot <- rowInts "slot" 1 1
        k <- cache "k" (Ring 1 1 4 2)
        v <- cache "v" (Ring 1 1 4 2)
        kv <- input "kv" (Shape 2 2) TF32
        kvStore "store_k" StoreAt {storeCache = k, storePositions = pos, storeSequences = slot} kv
        kvStore "store_v" StoreAt {storeCache = v, storePositions = pos, storeSequences = slot} kv
        o <- attend "o" (AttnSpec 2 1 2 Nothing 1) q AttendTo {attendKeys = k, attendValues = v, attendPositions = pos, attendSequences = slot}
        output "o" o
      b =
        (bindings 1)
          { bindInputs = Map.fromList [("q", [[1, 2, 3, 4]]), ("kv", [[0.1, 0.2]])]
          , bindRowInts = Map.fromList [("position", [[0]]), ("slot", [[0]])]
          , bindCaches = Map.fromList [("k", slots 8 (replicate 8 0)), ("v", slots 8 (replicate 8 0))]
          }
      stored = map (float2Double . widenBF16 . narrowBF16) [0.1, 0.2]
  fmap (Map.lookup "o" . (.evalOutputs)) (evaluate (checked p) b)
    @?= Right (Just [stored <> stored])

evalStored :: IO ()
evalStored = do
  let p = program "attention" $ do
        q <- input "q" (Shape 4 2) TF32
        pos <- rowInts "position" 1 4
        slot <- rowInts "slot" 1 1
        k <- cache "k" (Ring 1 1 4 2)
        v <- cache "v" (Ring 1 1 4 2)
        one <- weight "one" [1]
        kv <- input "kv" (Shape 2 2) TF32 >>= scaleBy "scaled" one
        kvStore "store_k" StoreAt {storeCache = k, storePositions = pos, storeSequences = slot} kv
        kvStore "store_v" StoreAt {storeCache = v, storePositions = pos, storeSequences = slot} kv
        o <- attend "o" (AttnSpec 2 1 2 Nothing 1) q AttendTo {attendKeys = k, attendValues = v, attendPositions = pos, attendSequences = slot}
        output "o" o
      b =
        (bindings 1)
          { bindInputs = Map.fromList [("q", [[1, 2, 3, 4]]), ("kv", [[midpoint, 0.2]])]
          , bindWeights = Map.fromList [("one", [1])]
          , bindRowInts = Map.fromList [("position", [[0]]), ("slot", [[0]])]
          , bindCaches = Map.fromList [("k", slots 8 (replicate 8 0)), ("v", slots 8 (replicate 8 0))]
          }
      -- Halfway between bf16's 1 and the next, 1 + 2^-7, where binary32's
      -- rounding takes it either way; the evaluator rounds to the even 1.
      midpoint = 1 + 2 ^^ (-8 :: Int)
      ulp = 2 ^^ (-7 :: Int)
      second = roundBF16 0.2
      kept x = Kept (Map.fromList [("k", slots 8 ([x, second] <> replicate 6 0)), ("v", slots 8 ([x, second] <> replicate 6 0))]) Map.empty
  fmap (Map.lookup "o" . (.evalOutputs)) (evaluateStored (kept (1 + ulp)) (checked p) b)
    @?= Right (Just [[1 + ulp, second, 1 + ulp, second]])
  case evaluateStored (kept (1 + 2 * ulp)) (checked p) b of
    Left _ -> pure ()
    Right _ -> assertFailure "a store two steps from the midpoint's rounding was kept"
  case evaluateStored (kept (1 - ulp / 2)) (checked p) b of
    Left _ -> pure ()
    Right _ -> assertFailure "a store a step below the midpoint's rounding was kept"

-- | attn_mixed over the caches as its own evaluation leaves them: the same outputs.
evalMixed :: IO ()
evalMixed = do
  let b =
        (bindings 1)
          { bindInputs = Map.fromList [("q", [[1, 2, 3, 4]]), ("kv", [[0.5, -0.25]]), ("vv", [[1, 2]])]
          , bindWeights = Map.fromList [("attn_scale", [2]), ("o_proj", [if i `mod` 5 == 0 then 1 else 0.5 | i <- [0 .. 15 :: Int]])]
          , bindRowInts = Map.fromList [("position", [[0]]), ("slot", [[1]])]
          , bindCaches = Map.fromList [("k", slots 8 (replicate 16 0)), ("v", slots 8 (replicate 16 0))]
          }
  case evaluate (checked attnMixed) b of
    Left e -> assertFailure (T.unpack e)
    Right e ->
      fmap (.evalOutputs) (evaluateStored (Kept e.evalCaches Map.empty) (checked attnMixed) b) @?= Right e.evalOutputs

------------------------------------------------------------------------------
-- The evaluator's bounds

-- | A linear layer of one output over @k@ inputs, stored into a cache of one element.
storedDot :: Natural -> Program
storedDot k = program "stored_dot" $ do
  x <- input "x" (wholeRow k) TF32
  w <- weight "w" [1, k]
  pos <- rowInts "position" 1 1
  slot <- rowInts "slot" 1 1
  c <- cache "c" (Ring 1 1 1 1)
  y <- linear "y" w 0 x
  kvStore "store" StoreAt {storeCache = c, storePositions = pos, storeSequences = slot} y
  output "y" y

dotBindings :: [Double] -> [Double] -> Bindings
dotBindings w x =
  (bindings 1)
    { bindInputs = Map.fromList [("x", [x])]
    , bindWeights = Map.fromList [("w", w)]
    , bindRowInts = Map.fromList [("position", [[0]]), ("slot", [[0]])]
    , bindCaches = Map.fromList [("c", slots 1 [0])]
    }

-- | Whether the evaluator keeps a kernel's store of @k@ for the dot product of @w@ and @x@.
keeps :: [Double] -> [Double] -> Double -> Bool
keeps w x k = isRight (evaluateStored (Kept (Map.fromList [("c", slots 1 [k])]) Map.empty) (checked (storedDot (fromIntegral (length x)))) (dotBindings w x))

boundsRefuse :: IO ()
boundsRefuse = do
  -- 1, exactly: bf16's steps are 2^-8 below it, 2^-7 above.
  let one = keeps [1] [1]
  assertBool "1 was refused for 1" (one 1)
  assertBool "two steps below 1 were kept for 1" (not (one (1 - 2 ^^ (-7 :: Int))))
  assertBool "a step below 1 was kept for 1" (not (one (1 - 2 ^^ (-8 :: Int))))
  assertBool "a step above 1 was kept for 1" (not (one (1 + 2 ^^ (-7 :: Int))))
  -- 1 + 3 · 2^-9 rounds up to 1 + 2^-7; truncated, it is 1.
  let up = keeps [1] [1 + 3 * 2 ^^ (-9 :: Int)]
      down = keeps [1] [-(1 + 3 * 2 ^^ (-9 :: Int))]
  assertBool "the rounding was refused" (up (1 + 2 ^^ (-7 :: Int)))
  assertBool "the truncation was kept" (not (up 1))
  assertBool "the negative rounding was refused" (down (-(1 + 2 ^^ (-7 :: Int))))
  assertBool "the negative truncation was kept" (not (down (-1)))

boundsAccept :: IO ()
boundsAccept = do
  -- binary32 adds 1 + 2^-30 to 1, then -1 to 0; the exact sum is 2^-30.
  let cancelled = keeps [1, 1, 1] [1, 2 ^^ (-30 :: Int), -1]
      f32 = [1, 2 ^^ (-30 :: Int), -1] :: [Float]
  sum f32 @?= 0
  assertBool "binary32's 0 was refused" (cancelled 0)
  assertBool "the exact 2^-30 was refused" (cancelled (2 ^^ (-30 :: Int)))
  -- 2^-134 is halfway between 0 and bf16's least subnormal, 2^-133.
  let tie = keeps [2 ^^ (-67 :: Int)] [2 ^^ (-67 :: Int)]
  assertBool "0 was refused at the tie" (tie 0)
  assertBool "2^-133 was refused at the tie" (tie (2 ^^ (-133 :: Int)))
  -- 3 · 2^-134 is halfway between the subnormals 2^-133 and 2^-132.
  let subnormals = keeps [3 * 2 ^^ (-67 :: Int)] [2 ^^ (-67 :: Int)]
  assertBool "2^-133 was refused" (subnormals (2 ^^ (-133 :: Int)))
  assertBool "2^-132 was refused" (subnormals (2 ^^ (-132 :: Int)))
  -- (2 - 2^-8) · 2^127 is halfway between bf16's largest finite value
  -- and 2^128, which overflows to infinity.
  let overflowing = keeps [2 ^^ (63 :: Int)] [(2 - 2 ^^ (-8 :: Int)) * 2 ^^ (64 :: Int)]
  assertBool "the largest finite value was refused" (overflowing ((2 - 2 ^^ (-7 :: Int)) * 2 ^^ (127 :: Int)))
  assertBool "infinity was refused" (overflowing (1 / 0))
  assertBool "a step below the largest finite value was kept" (not (overflowing ((2 - 2 ^^ (-6 :: Int)) * 2 ^^ (127 :: Int))))
  -- NaN stores as NaN.
  let nan = keeps [1] [0 / 0]
  assertBool "NaN was refused for NaN" (nan (0 / 0))
  assertBool "0 was kept for NaN" (not (nan 0))

-- | A head of 16 turned at positions below 2^18 by @θ@ = 10000, and stored.
storedRope :: Program
storedRope = program "stored_rope" $ do
  x <- input "x" (Shape 16 16) TF32
  pos <- rowInts "position" 1 262144
  slot <- rowInts "slot" 1 1
  c <- cache "c" (Ring 1 1 1 16)
  turned <- rope "turned" (RopeSpec 10000 8 1 RopeHalves) pos x
  kvStore "store" StoreAt {storeCache = c, storePositions = pos, storeSequences = slot} turned
  output "turned" turned

boundsLongRope :: IO ()
boundsLongRope = do
  let p = 262143 :: Int
      xs = [0.3, -0.7, 1.1, 0.25, -1.3, 0.9, 0.6, -0.2, 0.45, -0.35, 0.8, -1.05, 0.15, 0.55, -0.65, 1.2] :: [Float]
      freq i = 10000 ** negate (fromIntegral (2 * i) / 16) :: Double
      -- The pairs (i, i + 8) turned by the cosines and sines given, in binary32.
      turnedBy :: (Int -> (Float, Float)) -> [Double]
      turnedBy cs =
        [ roundBF16 (float2Double v)
        | j <- [0 .. 15]
        , let i = j `mod` 8
              (c, s) = cs i
              v = if j < 8 then xs !! j * c - xs !! (j + 8) * s else xs !! j * c + xs !! i * s
        ]
      -- The angle a binary32 product of the position and the frequency,
      -- rounded once.
      byProduct i = let angle = fromIntegral p * double2Float (freq i) in (cos angle, sin angle)
      -- Tables of 256 q · θ_i and r · θ_i for p = 256 q + r, rounded once,
      -- by the angle-addition formulas.
      byTables i =
        let (q, r) = p `divMod` 256
            at f n = double2Float (f (fromIntegral n * freq i))
            (ch, sh, cl, sl) = (at cos (256 * q), at sin (256 * q), at cos r, at sin r)
         in (ch * cl - sh * sl, sh * cl + ch * sl)
      b =
        (bindings 1)
          { bindInputs = Map.fromList [("x", [map float2Double xs])]
          , bindRowInts = Map.fromList [("position", [[fromIntegral p]]), ("slot", [[0]])]
          , bindCaches = Map.fromList [("c", slots 16 (replicate 16 0))]
          }
      keptBy stored = evaluateStored (Kept (Map.fromList [("c", slots 16 stored)]) Map.empty) (checked storedRope) b
  -- binary32's angles turn some elements to other bf16 than the exact ones do.
  case fmap (Map.lookup "c" . (.evalCaches)) (evaluate (checked storedRope) b) of
    Right (Just own) -> assertBool "binary32's angles rounded as the exact ones" (own /= slots 16 (turnedBy byProduct))
    other -> assertFailure (show other)
  mapM_
    (\(how, stored) -> either (\err -> assertFailure (how <> ": " <> T.unpack err)) (const (pure ())) (keptBy stored))
    [("products", turnedBy byProduct), ("tables", turnedBy byTables)]

boundsLongDots :: Property
boundsLongDots = property $ do
  k <- forAll (Gen.element [2048, 4096])
  w <- forAll (Gen.list (Range.singleton k) (Gen.float (Range.linearFrac (-1) 1)))
  x <- forAll (Gen.list (Range.singleton k) (Gen.float (Range.linearFrac (-1) 1)))
  let products = zipWith (*) w x
      pairwise = \case
        [] -> 0
        [a] -> a
        as -> let (l, r) = splitAt (length as `div` 2) as in pairwise l + pairwise r
      -- Left to right.
      inOrder = sum products
      kept v = keeps (map float2Double w) (map float2Double x) (roundBF16 (float2Double v))
  annotateShow (inOrder, pairwise products)
  assert (kept inOrder)
  assert (kept (pairwise products))

{- | Positive terms, so that the dot product's bound is a few hundredths
of a bf16 step: unless the exact value is nearer a midpoint than that, the
bf16 values beside its rounding are refused.
-}
boundsTight :: Property
boundsTight = property $ do
  w <- forAll (Gen.list (Range.singleton 64) (Gen.float (Range.linearFrac 0.5 1)))
  x <- forAll (Gen.list (Range.singleton 64) (Gen.float (Range.linearFrac 0.5 1)))
  let ws = map float2Double w
      xs = map float2Double x
      exact = sum (zipWith (*) ws xs)
      own = roundBF16 exact
      step = 2 ^^ (exponent own - 8)
      above = own + step
      below = if own == 2 ^^ (exponent own - 1) then own - step / 2 else own - step
      fromMidpoints = min (exact - (below + own) / 2) ((own + above) / 2 - exact) / exact
  annotateShow (exact, fromMidpoints)
  assert (keeps ws xs own)
  when (fromMidpoints > 2 ^^ (-14 :: Int)) $ do
    assert (not (keeps ws xs above))
    assert (not (keeps ws xs below))

{- | Attention over a ring of 4 positions with a window of 2: a step of a
sequence's next @n@ positions evaluates as its rows one at a time do,
outputs and the cache it leaves, though a step of more than 3 stores over
positions its first rows read.
-}
ringProgram :: Program
ringProgram = program "ring" $ do
  q <- input "q" (Shape 4 2) TF32
  pos <- rowInts "position" 1 64
  slot <- rowInts "slot" 1 2
  k <- cache "k" (Ring 2 1 4 2)
  v <- cache "v" (Ring 2 1 4 2)
  kv <- input "kv" (Shape 2 2) TF32
  vv <- input "vv" (Shape 2 2) TF32
  kvStore "store_k" StoreAt {storeCache = k, storePositions = pos, storeSequences = slot} kv
  kvStore "store_v" StoreAt {storeCache = v, storePositions = pos, storeSequences = slot} vv
  o <- attend "o" (AttnSpec 2 1 2 (Just 2) 1) q AttendTo {attendKeys = k, attendValues = v, attendPositions = pos, attendSequences = slot}
  output "o" o

-- | 'ringProgram', its keys and values scaled by a weight before they are stored.
ringScaled :: Program
ringScaled = program "ring_scaled" $ do
  q <- input "q" (Shape 4 2) TF32
  pos <- rowInts "position" 1 64
  slot <- rowInts "slot" 1 2
  k <- cache "k" (Ring 2 1 4 2)
  v <- cache "v" (Ring 2 1 4 2)
  one <- weight "one" [1]
  kv <- input "kv" (Shape 2 2) TF32 >>= scaleBy "kv_scaled" one
  vv <- input "vv" (Shape 2 2) TF32 >>= scaleBy "vv_scaled" one
  kvStore "store_k" StoreAt {storeCache = k, storePositions = pos, storeSequences = slot} kv
  kvStore "store_v" StoreAt {storeCache = v, storePositions = pos, storeSequences = slot} vv
  o <- attend "o" (AttnSpec 2 1 2 (Just 2) 1) q AttendTo {attendKeys = k, attendValues = v, attendPositions = pos, attendSequences = slot}
  output "o" o

evalCausal :: Property
evalCausal = property $ do
  start <- forAll (Gen.int (Range.linear 0 9))
  n <- forAll (Gen.int (Range.linear 1 9))
  let element = Gen.double (Range.linearFrac (-2) 2)
  qs <- forAll (Gen.list (Range.singleton n) (Gen.list (Range.singleton 4) element))
  ks <- forAll (Gen.list (Range.singleton n) (Gen.list (Range.singleton 2) element))
  vs <- forAll (Gen.list (Range.singleton n) (Gen.list (Range.singleton 2) element))
  cache0 <- forAll (Gen.list (Range.singleton 16) element)
  let caches0 = Map.fromList [("k", slots 8 cache0), ("v", slots 8 (reverse cache0))]
      step these caches = do
        let (ps, qr, kr, vr) = unzip4 these
        evaluate
          (checked ringProgram)
          (bindings (length these))
            { bindInputs = Map.fromList [("q", qr), ("kv", kr), ("vv", vr)]
            , bindRowInts = Map.fromList [("position", [[fromIntegral p'] | p' <- ps]), ("slot", map (const [1]) ps)]
            , bindCaches = caches
            }
      rows' = zip4 [start ..] qs ks vs
      whole = step rows' caches0
      oneByOne =
        foldl
          ( \acc r -> do
              (outs, caches) <- acc
              e <- step [r] caches
              pure (outs <> Map.findWithDefault [] "o" e.evalOutputs, e.evalCaches)
          )
          (Right ([], caches0))
          rows'
  fmap (\e -> (Map.findWithDefault [] "o" e.evalOutputs, e.evalCaches)) whole === oneByOne

evalStaged :: IO ()
evalStaged = do
  let
    -- Positions 0 to 4 of slot 0: position 4 overwrites position 0. Row
    -- 0's first element is halfway between bf16's 1 and 1 + 2^-7.
    kvs = [[if i == 1 then 1 + 2 ^^ (-8 :: Int) else 0.1 * fromIntegral i, 0.2] | i <- [1 .. 5 :: Int]]
    b =
      (bindings 5)
        { bindInputs = Map.fromList [("q", replicate 5 [1, 2, 3, 4]), ("kv", kvs), ("vv", kvs)]
        , bindWeights = Map.fromList [("one", [1])]
        , bindRowInts = Map.fromList [("position", [[i] | i <- [0 .. 4]]), ("slot", replicate 5 [0])]
        , bindCaches = Map.fromList [("k", slots 8 (replicate 16 0)), ("v", slots 8 (replicate 16 0))]
        }
    -- The kernels staged each row's rounding, row 0's first element the
    -- midpoint's other one.
    ulp = 2 ^^ (-7 :: Int)
    staged = [[roundBF16 x + (if r == 0 && j == 0 then ulp else 0) | (j, x) <- zip [0 :: Int ..] kv] | (r, kv) <- zip [0 :: Int ..] kvs]
    left = concat (staged !! 4 : take 3 (drop 1 staged)) <> replicate 8 0
    kept cache' = Kept (Map.fromList [("k", slots 8 cache'), ("v", slots 8 cache')]) (Map.fromList [("k", staged), ("v", staged)])
    own = evaluate (checked ringScaled) b
  case (evaluateStored (kept left) (checked ringScaled) b, own) of
    (Right e, Right o) -> do
      -- Row 1 reads row 0's staged key, which the cache no longer holds.
      let rowOne = (!! 1) . Map.findWithDefault [] "o"
      assertBool "row 1 reads row 0's staged values" (rowOne e.evalOutputs /= rowOne o.evalOutputs)
      Map.lookup "k" e.evalCaches @?= Just (slots 8 left)
    (Left err, _) -> assertFailure (T.unpack err)
    (_, Left err) -> assertFailure (T.unpack err)
  case evaluateStored (kept (concat (take 4 staged) <> replicate 8 0)) (checked ringScaled) b of
    Left _ -> pure ()
    Right _ -> assertFailure "a cache that kept a store a later row's overwrote was taken"
  let twoOff = [[x + (if r == 0 && j == 0 then ulp else 0) | (j, x) <- zip [0 :: Int ..] row] | (r, row) <- zip [0 :: Int ..] staged]
  case evaluateStored (Kept (Map.fromList [("k", slots 8 left), ("v", slots 8 left)]) (Map.fromList [("k", twoOff), ("v", twoOff)])) (checked ringScaled) b of
    Left _ -> pure ()
    Right _ -> assertFailure "a staged value two steps from the midpoint's rounding was taken"

{- | Attention over a paged pool of 2^30 blocks of 4 positions, 2^33
elements in all, its keys and values stored first.
-}
hugePool :: Program
hugePool = program "huge_pool" $ do
  q <- input "q" (Shape 2 2) TF32
  pos <- rowInts "position" 1 8
  blocks <- rowInts "blocks" 2 (2 ^ (30 :: Int))
  k <- cache "k" (Paged (2 ^ (30 :: Int)) 4 1 2)
  v <- cache "v" (Paged (2 ^ (30 :: Int)) 4 1 2)
  kv <- input "kv" (Shape 2 2) TF32
  kvStore "store_k" StoreAt {storeCache = k, storePositions = pos, storeSequences = blocks} kv
  kvStore "store_v" StoreAt {storeCache = v, storePositions = pos, storeSequences = blocks} kv
  o <- attend "o" (AttnSpec 1 1 2 Nothing 1) q AttendTo {attendKeys = k, attendValues = v, attendPositions = pos, attendSequences = blocks}
  output "o" o

evalSparse :: IO ()
evalSparse = do
  -- Blocks 123456789 and 987654321 of the 2^30, a row at position 5: the
  -- second block's element 2 on. Positions 0 to 3 hold ones, 4 zeros.
  let (b1, b2) = (123456789, 987654321)
      pool = Map.fromList [(b1, replicate 8 1), (b2, replicate 8 0)]
      b caches =
        (bindings 1)
          { bindInputs = Map.fromList [("q", [[0, 0]]), ("kv", [[0.5, 0.25]])]
          , bindRowInts = Map.fromList [("position", [[5]]), ("blocks", [[fromIntegral b1, fromIntegral b2]])]
          , bindCaches = Map.fromList [("k", caches), ("v", caches)]
          }
  case evaluate (checked hugePool) (b pool) of
    Right e -> do
      -- A query of zeros weighs the six positions alike.
      let want = [sum [x / 6 | x <- [1, 1, 1, 1, 0, 0.5]], sum [x / 6 | x <- [1, 1, 1, 1, 0, 0.25]]]
      case Map.lookup "o" e.evalOutputs of
        Just [o] -> assertBool (show o) (maximum (zipWith (\x y -> abs (x - y)) o want) < 1e-12)
        other -> assertFailure (show other)
      Map.lookup "k" e.evalCaches @?= Just (Map.fromList [(b1, replicate 8 1), (b2, [0, 0, 0.5, 0.25, 0, 0, 0, 0])])
    Left err -> assertFailure (T.unpack err)
  void (evaluate (checked hugePool) (b (Map.delete b2 pool)))
    @?= Left "A store to k at block 987654321, element 2, which isn't bound."
  void (evaluate (checked hugePool) (b (Map.delete b1 pool)))
    @?= Left "k: block 123456789, element 0, isn't bound."
  -- A ring bound by the slot its step's sequence is in, of the two.
  let ring = program "ring_slots" $ do
        q <- input "q" (Shape 4 2) TF32
        pos <- rowInts "position" 1 4
        slot <- rowInts "slot" 1 2
        k <- cache "k" (Ring 2 1 4 2)
        v <- cache "v" (Ring 2 1 4 2)
        kv <- input "kv" (Shape 2 2) TF32
        kvStore "store_k" StoreAt {storeCache = k, storePositions = pos, storeSequences = slot} kv
        kvStore "store_v" StoreAt {storeCache = v, storePositions = pos, storeSequences = slot} kv
        o <- attend "o" (AttnSpec 2 1 2 Nothing 1) q AttendTo {attendKeys = k, attendValues = v, attendPositions = pos, attendSequences = slot}
        output "o" o
      ringStep caches =
        (bindings 1)
          { bindInputs = Map.fromList [("q", [[1, 2, 3, 4]]), ("kv", [[0.5, 0.25]])]
          , bindRowInts = Map.fromList [("position", [[2]]), ("slot", [[1]])]
          , bindCaches = Map.fromList [("k", caches), ("v", caches)]
          }
      outputs = fmap (Map.lookup "o" . (.evalOutputs)) . evaluate (checked ring) . ringStep
  outputs (slots 8 [1 .. 16]) @?= outputs (Map.fromList [(1, [9 .. 16])])
  void (outputs (Map.fromList [(0, [1 .. 8])])) @?= Left "A store to k at slot 1, element 4, which isn't bound."

evalRoute :: IO ()
evalRoute = do
  let p = program "routing" $ do
        x <- input "x" (wholeRow 4) TF32
        (ids, ws) <- route "r" 2 Nothing x
        output "ids" ids
        output "ws" ws
      b = (bindings 1) {bindInputs = Map.fromList [("x", [[1, 3, 3, 2]])]}
  fmap (\e -> (Map.lookup "ids" e.evalOutputs, Map.lookup "ws" e.evalOutputs)) (evaluate (checked p) b)
    @?= Right (Just [[1, 2]], Just [[0.5, 0.5]])

evalConcat :: IO ()
evalConcat = do
  let p = program "joined" $ do
        x <- input "x" (wholeRow 2) TF32
        y <- input "y" (wholeRow 2) TF32
        w <- weight "w" [1, 4]
        joined <- concatenate "joined" x y
        z <- linear "z" w 0 joined
        output "joined" joined
        output "z" z
      b =
        (bindings 2)
          { bindInputs = Map.fromList [("x", [[1, 2], [3, 4]]), ("y", [[5, 6], [7, 8]])]
          , bindWeights = Map.fromList [("w", [1, 10, 100, 1000])]
          }
  fmap (\e -> (Map.lookup "joined" e.evalOutputs, Map.lookup "z" e.evalOutputs)) (evaluate (checked p) b)
    @?= Right (Just [[1, 2, 5, 6], [3, 4, 7, 8]], Just [[6521], [8743]])

-- | Rows of one segment of 2 and of two segments of 1, concatenated.
badConcat :: Program
badConcat = program "mismatched" $ do
  x <- input "x" (wholeRow 2) TF32
  y <- input "y" (Shape 2 1) TF32
  joined <- concatenate "joined" x y
  output "joined" joined

-- | A head of 8 turned in two parts of 4 by a position each, its source as wide as given.
axialRope :: Natural -> Program
axialRope width = program "axial" $ do
  x <- input "x" (Shape 8 8) TF32
  pos <- rowInts "position" width 64
  turned <- rope "turned" (RopeSpec 10000 2 2 RopeHalves) pos x
  output "turned" turned

evalAxialRope :: IO ()
evalAxialRope = do
  let b =
        (bindings 2)
          { bindInputs = Map.fromList [("x", replicate 2 [1, 2, 3, 4, 5, 6, 7, 8])]
          , bindRowInts = Map.fromList [("position", [[3, 0], [0, 5]])]
          }
      -- A part of 4: pairs (0, 2) and (1, 3), turned by p and p / 100.
      turn :: Double -> [Double] -> [Double]
      turn at [a, c, e, g] =
        let (s0, c0) = (sin at, cos at)
            (s1, c1) = (sin (at / 100), cos (at / 100))
         in [a * c0 - e * s0, c * c1 - g * s1, e * c0 + a * s0, g * c1 + c * s1]
      turn _ xs = xs
      expected = [turn 3 [1, 2, 3, 4] <> [5, 6, 7, 8], [1, 2, 3, 4] <> turn 5 [5, 6, 7, 8]]
  case fmap (Map.lookup "turned" . (.evalOutputs)) (evaluate (checked (axialRope 2)) b) of
    Right (Just rows) -> assertBool (show rows) (and (zipWith (\r e -> maximum (zipWith (\u v -> abs (u - v)) r e) < 1e-12) rows expected))
    other -> assertFailure (show other)

-- | Three rows of a table of 6, summed and divided by 3, then a linear layer.
pooled :: Program
pooled = program "pooled" $ do
  t <- weight "table" [6, 4]
  tok <- rowInts "rows" 3 6
  w <- weight "w" [2, 4]
  rows <- gather "pooled" t tok (1 / 3)
  out <- linear "out" w 0 rows
  output "out" out

evalPooledGather :: IO ()
evalPooledGather = do
  let p = program "summed" $ do
        t <- weight "table" [4, 2]
        tok <- rowInts "rows" 2 4
        rows <- gather "summed" t tok 0.5
        output "summed" rows
      b =
        (bindings 2)
          { bindWeights = Map.fromList [("table", [1, 2, 3, 4, 5, 6, 7, 8])]
          , bindRowInts = Map.fromList [("rows", [[0, 1], [2, 3]])]
          }
  fmap (Map.lookup "summed" . (.evalOutputs)) (evaluate (checked p) b) @?= Right (Just [[2, 3], [6, 7]])

-- | Rows of 2 standardized by a bias as wide as given.
standardized :: Natural -> Program
standardized width = program "standardized" $ do
  x <- input "x" (wholeRow 2) TF32
  bias <- weight "bias" [width]
  scale <- weight "scale" [2]
  y <- standardize "y" bias scale 2 x
  output "y" y

evalStandardize :: IO ()
evalStandardize = do
  let b =
        (bindings 2)
          { bindInputs = Map.fromList [("x", [[1, 2], [3, 4]])]
          , bindWeights = Map.fromList [("bias", [1, 1]), ("scale", [2, 3])]
          }
  fmap (Map.lookup "y" . (.evalOutputs)) (evaluate (checked (standardized 2)) b) @?= Right (Just [[2, 9], [10, 21]])

roundingAgrees :: Property
roundingAgrees = property $ do
  x <- forAll (Gen.float (Range.exponentialFloatFrom 0 (-3.0e38) 3.0e38))
  roundBF16 (float2Double x) === float2Double (widenBF16 (narrowBF16 x))
