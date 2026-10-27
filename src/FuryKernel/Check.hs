-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE RoleAnnotations #-}

{- | The gate.

'check' is total: it never throws, and it returns either a 'Checked'
kernel, whose constructor is not exported, or the findings that stop it.
Every backend takes a 'Checked' kernel, so a kernel that breaks a rule
cannot reach one: there is no value of the right type to hand it.

The checks work on the IR alone, so they hold for any IR, built by the
typed surface or not, and re-check what the types promise:

* names valid in every target language and unique;
* typing of every expression and statement;
* launch sizes: every dimension with legal values, and the grid and the
  buffers' and tables' sizes in u32 at every step, dividing by no zero,
  for every legal value of their dimensions ("FuryKernel.Check.Launch");
* uniformity: a divergence analysis computes how uniform every value is
  (across the threadgroup, across each SIMD group, or per lane) and
  verifies the conditions and loop bounds the builder declared uniform. A
  value loaded from a buffer or a table the kernel never writes is as
  uniform as its index, and so is one loaded from a threadgroup array
  that no store can change in the load's barrier epoch: from the barrier
  before it to the one after, every side of every if and every pass of
  every loop counted, whatever their conditions; any other threadgroup
  load is divergent;
* regions: barriers only where every thread of the threadgroup arrives,
  SIMD-group operations only where every lane of the group does;
* resources: the thread count, each threadgroup array and their total in
  the kernel's layout ('FuryKernel.IR.tgLayout'), the buffer bindings and
  the alignments, by the rules of "FuryKernel.Constraint", and the SIMD
  width the kernel was built for;
* packs: each pack's width against the target's lane loads and its
  buffer's alignment, by the same module's rule.

Races, bounds and the order of memory operations come with the hazard and
bounds checks of later stages.
-}
module FuryKernel.Check (
  Checked,
  checkedIR,
  check,
  checkIR,
  nameProblem,
  reservedNames,
) where

import Control.Applicative ((<|>))
import Control.Monad (forM_, unless, when)
import Control.Monad.Trans.State.Strict (State, execState, gets, modify')
import Data.Char (isAsciiLower, isAsciiUpper, isDigit)
import Data.Foldable (traverse_)
import Data.List (nub)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set (Set)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as T
import FuryKernel.Build (Kernel, kernelCaps, kernelIR, kernelProblems)
import FuryKernel.Check.Launch (checkLaunchSizes)
import FuryKernel.Constraint (checkAlign, checkBuffers, checkPack, checkTgArrayOf, checkTgTotal, checkThreads)
import FuryKernel.Finding (Finding (..), Location (..), Rule (..), locatedAt)
import FuryKernel.IR hiding (Kernel (..))
import FuryKernel.IR qualified as IR
import FuryKernel.Target (Target, TargetCaps (..), targetCaps)
import FuryKernel.Type (ScalarType (..), isIntegral, isStorageOnly, scalarTypeName)

-- | A kernel that passed the gate for target @t@.
type role Checked nominal

newtype Checked t = Checked IR.Kernel

-- | The IR of a checked kernel, for backends.
checkedIR :: Checked t -> IR.Kernel
checkedIR (Checked k) = k

{- | A kernel through the gate of its own target: one built for another
target's caps, its IR baking in that target's SIMD width and limits, is
refused.
-}
check :: forall t. (Target t) => Kernel t -> Either [Finding] (Checked t)
check k
  | kernelCaps k /= caps = Left [Finding Target ("The kernel was built for " <> (kernelCaps k).capName <> ", whose SIMD width and limits its IR holds; it is checked only for that target, not for " <> caps.capName <> ".")]
  | otherwise = case map (placedIn (kernelIR k)) (kernelProblems k) <> checkIR caps (kernelIR k) of
      [] -> Right (Checked (kernelIR k))
      findings -> Left findings
 where
  caps = targetCaps @t

-- | Every finding for a kernel's IR on a target with these limits.
checkIR :: TargetCaps -> IR.Kernel -> [Finding]
checkIR caps k = map (placedIn k) $ checkNames k <> checkTyping k <> checkLaunchSizes k <> checkUniformity k <> checkResources caps k <> checkPacks caps k <> checkFragments caps k

{- | A finding placed in its kernel, if nothing placed it more closely: a
finding's place is its kernel and the ifs and loops around what it is about.
-}
placedIn :: IR.Kernel -> Finding -> Finding
placedIn k f = case f.findingLocation of
  Nothing -> locatedAt (Location k.kernelName []) f
  Just _ -> f

-- | Where the statements of an if, its else, and a loop are.
ifPlace, elsePlace :: Expr -> Text
ifPlace c = "if " <> renderExpr c
elsePlace c = "else of if " <> renderExpr c

forPlace :: Name -> Text
forPlace v = "for " <> v

------------------------------------------------------------------------------
-- Names

{- | Words no name may be: C++ keywords, which every shading language
inherits, the qualifiers, types, builtins and library functions of the
Metal, CUDA and HIP languages, the words Metal refuses as identifiers, and
the Rust keywords that cannot be raw identifiers (generated Rust escapes
the others) and its prelude's variants. Generated code reserves names
starting with @fk_@, in any case; no name may start with an underscore,
which C++ reserves in places.
-}
reservedNames :: Set Text
reservedNames =
  Set.fromList . T.words $
    T.unwords
      [ "alignas alignof and and_eq asm auto bitand bitor bool break case catch char"
      , "char8_t char16_t char32_t class compl concept const consteval constexpr"
      , "constinit const_cast continue co_await co_return co_yield decltype default"
      , "delete do double dynamic_cast else enum explicit export extern false float"
      , "for friend goto if inline int long mutable namespace new noexcept not not_eq"
      , "nullptr operator or or_eq private protected public register reinterpret_cast"
      , "requires return short signed sizeof static static_assert static_cast struct"
      , "switch template this thread_local throw true try typedef typeid typename union"
      , "unsigned using virtual void volatile wchar_t while xor xor_eq"
      , "kernel vertex fragment device constant threadgroup threadgroup_imageblock"
      , "thread visible stage_in half bfloat uint ushort uchar ulong size_t ptrdiff_t"
      , "metal main threadIdx blockIdx blockDim gridDim warpSize"
      , "max min select fma simd_sum simd_max simd_min local global"
      , "as_type ray_data object_data"
      , "self Self super crate None Some Ok Err"
      ]

checkNames :: IR.Kernel -> [Finding]
checkNames k =
  [Finding Names m | Just m <- map nameProblem allNames]
    <> [ Finding Names ("The name " <> n <> " is declared " <> T.pack (show c) <> " times.")
       | (n, c) <- Map.toList (Map.fromListWith (+) [(n, 1 :: Int) | n <- allNames])
       , c > 1
       ]
 where
  allNames =
    [k.kernelName]
      <> map (.paramName) k.kernelParams
      <> map (.bufName) k.kernelBuffers
      <> map (.tableName) k.kernelTables
      <> map (.tgName) k.kernelTgArrays
      <> concatMap bound k.kernelBody
  bound = \case
    SLet n _ _ -> [n]
    SVar n _ _ -> [n]
    SSimd n _ _ _ -> [n]
    SFragVar n _ -> [n]
    SLoadPack ns _ _ _ -> ns
    SLoadPackVia ns _ _ _ _ -> ns
    SIf _ _ th el -> concatMap bound th <> concatMap bound el
    SFor _ v _ _ _ body -> v : concatMap bound body
    _ -> []

{- | What is wrong with a name in generated code, if anything: it must be an
identifier of every target language, not reserved by one, not start with
an underscore or with @fk_@, and not hold two underscores in a row.
-}
nameProblem :: Name -> Maybe Text
nameProblem n
  | T.null n = Just "An empty name."
  | not (identStart (T.head n) && T.all identChar n) =
      Just ("The name " <> n <> " is not an identifier: use letters, digits and underscores.")
  | "_" `T.isPrefixOf` n = Just ("The name " <> n <> " starts with an underscore, which C++ reserves.")
  | "__" `T.isInfixOf` n = Just ("The name " <> n <> " holds two underscores in a row, which C++ reserves.")
  | "fk_" `T.isPrefixOf` T.toLower n = Just ("The name " <> n <> " starts with fk_ in some case, which generated code reserves.")
  | n `Set.member` reservedNames || vectorType n =
      Just ("The name " <> n <> " is reserved in a target language.")
  | otherwise = Nothing
 where
  identStart c = isAsciiLower c || isAsciiUpper c || c == '_'
  identChar c = identStart c || isDigit c
  -- float4, uint2, half3x3 and the like.
  vectorType name =
    any
      (\base -> maybe False shape (T.stripPrefix base name))
      ["float", "half", "bfloat", "double", "int", "uint", "short", "ushort", "char", "uchar", "long", "ulong", "bool"]
  shape s = case T.unpack s of
    [a] -> a `elem` ("234" :: String)
    [a, 'x', b] -> a `elem` ("234" :: String) && b `elem` ("234" :: String)
    _ -> False

------------------------------------------------------------------------------
-- Typing

data Local = Local {localType :: ScalarType, localMutable :: Bool, localFragment :: Bool}

data TypingState = TypingState
  { scopes :: [Map Name Local]
  , typingFindings :: [Finding]
  , typingPlace :: [Text]
  -- ^ The constructs around the statement typed, the innermost first.
  }

checkTyping :: IR.Kernel -> [Finding]
checkTyping k =
  reverse (execState run (TypingState [Map.empty] [] [])).typingFindings
    <> interfaceTyping k
 where
  params = Map.fromList [(p.paramName, p.paramType) | p <- k.kernelParams]
  buffers = Map.fromList [(b.bufName, b) | b <- k.kernelBuffers]
  tables = Map.fromList [(t.tableName, t) | t <- k.kernelTables]
  arrays = Map.fromList [(a.tgName, a) | a <- k.kernelTgArrays]

  run = traverse_ stmt k.kernelBody

  bad :: Text -> State TypingState ()
  bad m = modify' (\s -> s {typingFindings = locatedAt (Location k.kernelName (reverse s.typingPlace)) (Finding Typing m) : s.typingFindings})

  -- Inside a construct: what is found there is placed in it.
  within :: Text -> State TypingState () -> State TypingState ()
  within place m = do
    modify' (\s -> s {typingPlace = place : s.typingPlace})
    m
    modify' (\s -> s {typingPlace = drop 1 s.typingPlace})

  declare :: Name -> Local -> State TypingState ()
  declare n l = modify' $ \s -> case s.scopes of
    top : rest -> s {scopes = Map.insert n l top : rest}
    [] -> s {scopes = [Map.singleton n l]}

  scoped :: State TypingState () -> State TypingState ()
  scoped m = do
    modify' (\s -> s {scopes = Map.empty : s.scopes})
    m
    modify' (\s -> s {scopes = drop 1 s.scopes})

  lookupLocal :: Name -> State TypingState (Maybe Local)
  lookupLocal n = gets (\s -> foldr (\m acc -> Map.lookup n m <|> acc) Nothing s.scopes)

  -- The type of an expression, or Nothing after reporting why it has none.
  typeOf :: Expr -> State TypingState (Maybe ScalarType)
  typeOf = \case
    ELit l
      | canonicalNaN l -> pure (Just (literalType l))
      | otherwise -> Nothing <$ bad ("The literal " <> renderExpr (ELit l) <> " is a NaN other than its type's canonical one, the quiet NaN with no payload or sign.")
    EVar n ->
      lookupLocal n >>= \case
        Just l
          | l.localFragment -> Nothing <$ bad ("The fragment " <> n <> " is used as a value.")
          | otherwise -> pure (Just l.localType)
        Nothing -> Nothing <$ bad ("The local " <> n <> " is used outside its scope or before it is bound.")
    EParam n -> case Map.lookup n params of
      Just t -> pure (Just t)
      Nothing -> Nothing <$ bad ("No parameter " <> n <> ".")
    EBuiltin _ -> pure (Just TU32)
    EUn op e -> typeOf e >>= maybe (pure Nothing) (unary op)
    EBin op a b -> do
      ta <- typeOf a
      tb <- typeOf b
      case (ta, tb) of
        (Just x, Just y)
          | x /= y -> Nothing <$ bad ("Operands of " <> T.pack (show op) <> " differ: " <> scalarTypeName x <> " and " <> scalarTypeName y <> ".")
          | otherwise -> binary op x
        _ -> pure Nothing
    ESelect c a b -> do
      tc <- typeOf c
      ta <- typeOf a
      tb <- typeOf b
      case tc of
        Just t | t /= TBool -> bad "The condition of a select is not a bool."
        _ -> pure ()
      case (ta, tb) of
        (Just x, Just y) | x /= y -> Nothing <$ bad ("The sides of a select differ: " <> scalarTypeName x <> " and " <> scalarTypeName y <> ".")
        _ -> pure ta
    EFma a b c -> do
      ts <- mapM typeOf [a, b, c]
      case sequence ts of
        Just [x, y, z]
          | any (/= x) [y, z] -> Nothing <$ bad ("Operands of an fma differ: " <> T.intercalate ", " (map scalarTypeName [x, y, z]) <> ".")
          | x `elem` [TF32, TF16] -> pure (Just x)
          | otherwise -> Nothing <$ bad ("An fma of " <> scalarTypeName x <> ": only f32 and f16 have one.")
        _ -> pure Nothing
    EConvert to e ->
      typeOf e >>= \case
        Just from
          | convertible from to -> pure (Just to)
          | otherwise -> Nothing <$ bad ("No conversion from " <> scalarTypeName from <> " to " <> scalarTypeName to <> ".")
        Nothing -> pure Nothing
    ENarrowInt _ to e ->
      typeOf e >>= \case
        Just from
          | (from, to) `elem` [(TI32, TI8), (TU32, TU8)] -> pure (Just to)
          | otherwise -> Nothing <$ bad ("No integer narrowing from " <> scalarTypeName from <> " to " <> scalarTypeName to <> ".")
        Nothing -> pure Nothing
    ELoadVia n e i -> do
      index ("the entries of table " <> n) e
      index ("table " <> n) i
      case Map.lookup n tables of
        Just t -> pure (Just t.tableType)
        Nothing -> Nothing <$ bad ("No table " <> n <> ".")
    ELoad n i -> do
      index ("buffer " <> n) i
      case Map.lookup n buffers of
        Just b -> do
          when (b.bufAccess == WriteOnly) $ bad ("A load from the write-only buffer " <> n <> ".")
          pure (Just b.bufType)
        Nothing -> Nothing <$ bad ("No buffer " <> n <> ".")
    ETgLoad n i -> do
      index ("threadgroup array " <> n) i
      case Map.lookup n arrays of
        Just a -> pure (Just a.tgType)
        Nothing -> Nothing <$ bad ("No threadgroup array " <> n <> ".")

  unary op t = case op of
    Not
      | t == TBool -> pure (Just t)
      | otherwise -> Nothing <$ bad "Not of a value that is not a bool."
    _
      | op `elem` [Neg, Abs], numeric t -> pure (Just t)
      | op `elem` [Sqrt, Rsqrt, Exp, Log, Tanh], t `elem` [TF32, TF16] -> pure (Just t)
      | otherwise -> Nothing <$ bad (T.pack (show op) <> " of " <> scalarTypeName t <> ".")

  binary op t
    | op `elem` [Add, Sub, Mul, Div, Min, Max], numeric t = pure (Just t)
    | op == Rem, t `elem` [TI32, TU32] = pure (Just t)
    | op `elem` [Lt, Le, Gt, Ge], numeric t = pure (Just TBool)
    | op `elem` [Eq, Ne], numeric t || t == TBool = pure (Just TBool)
    | op `elem` [And, Or], t == TBool = pure (Just TBool)
    | otherwise = Nothing <$ bad (T.pack (show op) <> " of " <> scalarTypeName t <> ".")

  numeric t = t /= TBool && not (isStorageOnly t)

  convertible from to =
    from == to
      || (from, to) `elem` [(TBF16, TF32), (TF16, TF32), (TI8, TI32), (TU8, TU32), (TF32, TBF16), (TF32, TF16)]
      || (isIntegral from && not (isStorageOnly from) && to == TF32)

  index what i =
    typeOf i >>= \case
      Just TU32 -> pure ()
      Just t -> bad ("The index into " <> what <> " is " <> scalarTypeName t <> ", not u32.")
      Nothing -> pure ()

  expect what want e =
    typeOf e >>= \case
      Just t | t /= want -> bad (what <> " is " <> scalarTypeName t <> ", not " <> scalarTypeName want <> ".")
      _ -> pure ()

  stmt :: Stmt -> State TypingState ()
  stmt = \case
    SLet n t e -> expect ("The value of " <> n) t e >> declare n (Local t False False)
    SVar n t e -> expect ("The value of " <> n) t e >> declare n (Local t True False)
    SAssign n e ->
      lookupLocal n >>= \case
        Just l
          | l.localFragment -> bad ("An assignment to the fragment " <> n <> ".")
          | l.localMutable -> expect ("The value assigned to " <> n) l.localType e
          | otherwise -> bad ("An assignment to " <> n <> ", which is not a var.")
        Nothing -> bad ("An assignment to " <> n <> ", which is not in scope.")
    SStore n i e -> do
      index ("buffer " <> n) i
      case Map.lookup n buffers of
        Just b -> do
          when (b.bufAccess == ReadOnly) $ bad ("A store to the read-only buffer " <> n <> ".")
          expect ("The value stored to " <> n) b.bufType e
        Nothing -> bad ("No buffer " <> n <> ".")
    STgStore n i e -> do
      index ("threadgroup array " <> n) i
      case Map.lookup n arrays of
        Just a -> expect ("The value stored to " <> n) a.tgType e
        Nothing -> bad ("No threadgroup array " <> n <> ".")
    SLoadPackVia ns t n e i -> do
      index ("the entries of table " <> n) e
      index ("table " <> n) i
      case Map.lookup n tables of
        Just tb ->
          when (tb.tableType /= t) $
            bad ("A pack of " <> scalarTypeName t <> " loaded through " <> n <> ", a table of " <> scalarTypeName tb.tableType <> ".")
        Nothing -> bad ("No table " <> n <> ".")
      traverse_ (\l -> declare l (Local t False False)) ns
    SLoadPack ns t n i -> do
      index ("buffer " <> n) i
      case Map.lookup n buffers of
        Just b -> do
          when (b.bufAccess == WriteOnly) $ bad ("A load from the write-only buffer " <> n <> ".")
          when (b.bufType /= t) $
            bad ("A pack of " <> scalarTypeName t <> " loaded from " <> n <> ", a buffer of " <> scalarTypeName b.bufType <> ".")
        Nothing -> bad ("No buffer " <> n <> ".")
      traverse_ (\l -> declare l (Local t False False)) ns
    SStorePack n i es -> do
      index ("buffer " <> n) i
      case Map.lookup n buffers of
        Just b -> do
          when (b.bufAccess == ReadOnly) $ bad ("A store to the read-only buffer " <> n <> ".")
          traverse_ (expect ("A value stored to " <> n) b.bufType) es
        Nothing -> bad ("No buffer " <> n <> ".")
    STgStorePack n i es -> do
      index ("threadgroup array " <> n) i
      case Map.lookup n arrays of
        Just a -> traverse_ (expect ("A value stored to " <> n) a.tgType) es
        Nothing -> bad ("No threadgroup array " <> n <> ".")
    SIf _ c th el -> do
      expect "The condition of an if" TBool c
      within (ifPlace c) (scoped (traverse_ stmt th))
      within (elsePlace c) (scoped (traverse_ stmt el))
    SFor _ v start end step body -> do
      expect ("The start of loop " <> v) TU32 start
      expect ("The end of loop " <> v) TU32 end
      expect ("The step of loop " <> v) TU32 step
      when (step == ELit (LU32 0)) $ bad ("The loop " <> v <> " steps by 0, so it never ends.")
      within (forPlace v) (scoped (declare v (Local TU32 False False) >> traverse_ stmt body))
    SBarrier -> pure ()
    SSimdBarrier -> pure ()
    SSimd n _ t e -> do
      unless (t `elem` [TF32, TF16, TI32, TU32]) $ bad ("No SIMD-group reduction of " <> scalarTypeName t <> ".")
      expect ("The operand of " <> n) t e
      declare n (Local t False False)
    SFragVar n t -> do
      unless (t `elem` [TBF16, TF16, TF32]) $ bad ("No fragment of " <> scalarTypeName t <> ".")
      declare n (Local t True True)
    SFragLoad n m i st _ -> do
      index ("the fragment load into " <> n) i
      index ("the stride of the fragment load into " <> n) st
      t <- fragmentType n
      element ("The fragment " <> n <> " loaded from") m t True
    SFragStore m i st n -> do
      index ("the fragment store of " <> n) i
      index ("the stride of the fragment store of " <> n) st
      t <- fragmentType n
      element ("The fragment " <> n <> " stored to") m t False
    SFragMma d a b c -> do
      td <- fragmentType d
      ta <- fragmentType a
      tb <- fragmentType b
      tc <- fragmentType c
      when (ta /= tb) $ bad ("The fragments " <> a <> " and " <> b <> " multiplied differ in type.")
      when (d `elem` [a, b]) $ bad ("The product into " <> d <> " multiplies " <> d <> " too: only its accumulator may be its result.")
      unless (td == Just TF32 && tc == Just TF32) $ bad ("The product into " <> d <> " accumulates into fragments other than f32.")

  -- A fragment's element type, or Nothing after reporting it isn't one.
  fragmentType n =
    lookupLocal n >>= \case
      Just l | l.localFragment -> pure (Just l.localType)
      Just _ -> Nothing <$ bad ("The local " <> n <> " is not a fragment.")
      Nothing -> Nothing <$ bad ("The fragment " <> n <> " is used outside its scope or before it is declared.")

  -- A fragment's memory: there, of its element type, readable or writable.
  element what m t reading = case m of
    FDevice n -> case Map.lookup n buffers of
      Just b -> do
        when (reading && b.bufAccess == WriteOnly) $ bad (what <> " the write-only buffer " <> n <> ".")
        when (not reading && b.bufAccess == ReadOnly) $ bad (what <> " the read-only buffer " <> n <> ".")
        when (any (/= b.bufType) t) $ bad (what <> " " <> n <> ", a buffer of " <> scalarTypeName b.bufType <> ".")
      Nothing -> bad ("No buffer " <> n <> ".")
    FThreadgroup n -> case Map.lookup n arrays of
      Just a -> when (any (/= a.tgType) t) $ bad (what <> " " <> n <> ", an array of " <> scalarTypeName a.tgType <> ".")
      Nothing -> bad ("No threadgroup array " <> n <> ".")

-- | The grid and the extents: u32 expressions over parameters and literals.
interfaceTyping :: IR.Kernel -> [Finding]
interfaceTyping k =
  [Finding Typing ("The grid size " <> m) | Just m <- [launchOnly k.kernelGroups]]
    <> [Finding Typing ("The extent of " <> b.bufName <> " " <> m) | b <- k.kernelBuffers, Just m <- [launchOnly b.bufExtent]]
    <> [ Finding Typing ("The size of table " <> t.tableName <> " " <> m)
       | t <- k.kernelTables
       , Just m <- [launchOnly t.tableEntries, launchOnly t.tableExtent]
       ]
    <> [ Finding Typing ("The dimension " <> p.paramName <> " is " <> scalarTypeName p.paramType <> ", not u32.")
       | p <- k.kernelParams
       , isDim p.paramRole
       , p.paramType /= TU32
       ]
 where
  isDim = \case
    Dim _ -> True
    Scalar -> False
  dims = Set.fromList [p.paramName | p <- k.kernelParams, isDim p.paramRole]
  -- Launch sizes are computed on the host before the launch, from the
  -- dimensions alone, in integer arithmetic.
  launchOnly e
    | all (`Set.member` dims) (paramsOf e) && launchExpr e = Nothing
    | otherwise = Just "uses more than dimensions, u32 literals and + - * / %."
  launchExpr = \case
    ELit (LU32 _) -> True
    EParam _ -> True
    EBin op a b -> op `elem` [Add, Sub, Mul, Div, Rem] && launchExpr a && launchExpr b
    _ -> False
  paramsOf = \case
    EParam n -> [n]
    EBin _ a b -> paramsOf a <> paramsOf b
    _ -> []

------------------------------------------------------------------------------
-- Uniformity and regions

{- | How uniform a value is: the same in every thread of the threadgroup,
in every lane of each SIMD group, or per lane. Ordered by divergence.
-}
data Uni = U | S | D
  deriving stock (Eq, Ord, Show)

regionUni :: Region -> Uni
regionUni = \case
  RUniform -> U
  RSimdUniform -> S
  RDivergent -> D

checkUniformity :: IR.Kernel -> [Finding]
checkUniformity k = execState (walk [] U (epochs k.kernelBody) k.kernelBody) []
 where
  -- A local's uniformity: the join over everything bound or assigned to
  -- it, and the regions it happens in. Iterated to a fixpoint, since a
  -- loop can carry divergence back to the top.
  locals = fixpoint Map.empty
  top = epochs k.kernelBody
  fixpoint m = let m' = pass U top m k.kernelBody in if m' == m then m else fixpoint m'
  pass r es m ss = foldl' (\m' (e, s) -> step r e m' s) m (zip es ss)
  step r (Epoch stored inside) m = \case
    SLet n _ e -> bind n (uni m e) m
    SVar n _ e -> bind n (max r (uni m e)) m
    SAssign n e -> bind n (max r (uni m e)) m
    SSimd n _ _ _ -> bind n (max r S) m
    SLoadPack ns _ b i -> foldl' (\m' n -> bind n (uni m (ELoad b i)) m') m ns
    SLoadPackVia ns _ b e i -> foldl' (\m' n -> bind n (uni m (ELoadVia b e i)) m') m ns
    SIf d c th el ->
      let r' = inner r d (uni m c)
          (eth, eel) = blocks2 inside
       in pass r' eel (pass r' eth m th) el
    SFor d v start end st body ->
      let control = maximum [uni m start, uni m end, uni m st]
          r' = inner r d control
       in pass r' (blocks1 inside) (bind v (max r' control) m) body
    _ -> m
   where
    uni = uniIn stored
  bind = Map.insertWith max
  -- The region inside a construct: the declared one when it is honest,
  -- the actual one when it is not, and never more uniform than outside.
  inner r d control = maximum [r, regionUni d, control]

  uniIn :: Set Name -> Map Name Uni -> Expr -> Uni
  uniIn stored m = \case
    ELit _ -> U
    EParam _ -> U
    EBuiltin b -> case b of
      GroupInGrid -> U
      SimdGroup -> S
      SimdLane -> D
      ThreadInGroup -> D
    EVar n -> Map.findWithDefault D n m
    EUn _ e -> uniIn stored m e
    EBin _ a b -> max (uniIn stored m a) (uniIn stored m b)
    ESelect c a b -> maximum [uniIn stored m c, uniIn stored m a, uniIn stored m b]
    EFma a b c -> maximum [uniIn stored m a, uniIn stored m b, uniIn stored m c]
    EConvert _ e -> uniIn stored m e
    ENarrowInt _ _ e -> uniIn stored m e
    -- Only a buffer the kernel never writes reads the same in every thread.
    -- Tables are only read: the same entry and index read the same element.
    ELoadVia _ e i -> max (uniIn stored m e) (uniIn stored m i)
    ELoad n i
      | n `elem` readOnly -> uniIn stored m i
      | otherwise -> D
    -- No thread stores to the array in the load's barrier epoch, so every
    -- thread that loads an element reads one value.
    ETgLoad n i
      | not (Set.member n stored) -> uniIn stored m i
      | otherwise -> D
  readOnly = [b.bufName | b <- k.kernelBuffers, b.bufAccess == ReadOnly]

  report :: [Text] -> Rule -> Text -> State [Finding] ()
  report place rule m = modify' (<> [locatedAt (Location k.kernelName (reverse place)) (Finding rule m)])

  walk :: [Text] -> Uni -> [Epoch] -> [Stmt] -> State [Finding] ()
  walk place r es ss = forM_ (zip es ss) (uncurry (stmt place r))

  stmt :: [Text] -> Uni -> Epoch -> Stmt -> State [Finding] ()
  stmt place r (Epoch stored inside) = \case
    SIf d c th el -> do
      let u = uni locals c
          (eth, eel) = blocks2 inside
      declared place d u ("The condition " <> renderExpr c)
      walk (ifPlace c : place) (inner r d u) eth th
      walk (elsePlace c : place) (inner r d u) eel el
    SFor d v start end st body -> do
      let control = maximum [uni locals start, uni locals end, uni locals st]
      declared place d control ("The bounds of loop " <> v)
      walk (forPlace v : place) (inner r d control) (blocks1 inside) body
    SBarrier ->
      unless (r == U) $
        report place Regions "A barrier where not every thread of the threadgroup arrives."
    SSimd n _ _ _ ->
      when (r == D) $
        report place Regions ("The SIMD-group operation " <> n <> " where not every lane of the group is active.")
    SSimdBarrier ->
      when (r == D) $
        report place Regions "A SIMD-group barrier where not every lane of the group arrives."
    SFragVar n _ -> fragment n []
    SFragLoad n _ i st _ -> fragment n [i, st]
    SFragStore _ i st n -> fragment n [i, st]
    SFragMma d _ _ _ -> fragment d []
    _ -> pure ()
   where
    uni = uniIn stored
    -- A SIMD group's fragment operation: every lane active, and its
    -- memory's index and stride the same in every lane.
    fragment n addresses = do
      when (r == D) $
        report place Regions ("The fragment operation on " <> n <> " where not every lane of the group is active.")
      when (any ((> S) . uni locals) addresses) $
        report place Uniformity ("The fragment " <> n <> "'s index or stride differs across the lanes of a SIMD group.")

  declared place d u what =
    when (u > regionUni d) $
      report place Uniformity $
        what
          <> " is declared "
          <> describe (regionUni d)
          <> " but is "
          <> describe u
          <> "."
  describe = \case
    U -> "uniform across the threadgroup"
    S -> "uniform across each SIMD group"
    D -> "divergent across lanes"

{- | A statement's barrier epoch, for the divergence analysis: the
threadgroup arrays that some thread may store to between the threadgroup
barriers around it, and the same for the statements of its blocks (an
if's two, a loop's one). Computed from the statements alone, every side of
every if and every number of a loop's passes counted whatever their
conditions, so it rests on neither the divergence analysis nor the hazard
check; a barrier ends an epoch only where every construct around it is
declared uniform, as the gate holds barriers to be.
-}
data Epoch = Epoch (Set Name) [[Epoch]]

-- | The epochs of a block's statements: what may be stored since the last barrier, and until the next.
epochs :: [Stmt] -> [Epoch]
epochs body = zipWith joined (fst (forward True Set.empty body)) (fst (backward True body Set.empty))
 where
  joined (Stored f fs) (Stored b bs) = Epoch (Set.union f b) (zipWith (zipWith joined) fs bs)

-- | One pass's sets, a statement's and its blocks' statements'.
data Stored = Stored (Set Name) [[Stored]]

{- | From the start: each statement's arrays stored on some path since the
last barrier, and the block's after its end.
-}
forward :: Bool -> Set Name -> [Stmt] -> ([Stored], Set Name)
forward ok before = \case
  [] -> ([], before)
  s : rest ->
    let (here, after) = one s
        (there, out) = forward ok after rest
     in (here : there, out)
 where
  one = \case
    SBarrier | ok -> (Stored before [], Set.empty)
    SIf d _ th el ->
      let (ths, tout) = forward (ok && d == RUniform) before th
          (els, eout) = forward (ok && d == RUniform) before el
       in (Stored before [ths, els], Set.union tout eout)
    SFor d _ _ _ _ body ->
      let header = settle (\h -> Set.union before (snd (forward (ok && d == RUniform) h body))) before
       in (Stored header [fst (forward (ok && d == RUniform) header body)], header)
    other -> (Stored before [], Set.union before (storedBy other))

{- | From the end: each statement's arrays stored on some path from it to
the next barrier, its own stores included, and the block's from its start.
-}
backward :: Bool -> [Stmt] -> Set Name -> ([Stored], Set Name)
backward ok ss after = case ss of
  [] -> ([], after)
  s : rest ->
    let (there, next) = backward ok rest after
        (here, start) = one s next
     in (here : there, start)
 where
  one stmt next = case stmt of
    SBarrier | ok -> (Stored Set.empty [], Set.empty)
    SIf d _ th el ->
      let (ths, tin) = backward (ok && d == RUniform) th next
          (els, ein) = backward (ok && d == RUniform) el next
          start = Set.union tin ein
       in (Stored start [ths, els], start)
    SFor d _ _ _ _ body ->
      let header = settle (Set.union next . snd . backward (ok && d == RUniform) body) next
       in (Stored header [fst (backward (ok && d == RUniform) body header)], header)
    _ -> let start = Set.union next (storedBy stmt) in (Stored start [], start)

-- | The threadgroup array a statement stores to, if any.
storedBy :: Stmt -> Set Name
storedBy = \case
  STgStore n _ _ -> Set.singleton n
  STgStorePack n _ _ -> Set.singleton n
  SFragStore (FThreadgroup n) _ _ _ -> Set.singleton n
  _ -> Set.empty

settle :: (Eq a) => (a -> a) -> a -> a
settle f x = let x' = f x in if x' == x then x else settle f x'

-- | A construct's blocks' epochs: a loop's body, an if's two sides.
blocks1 :: [[Epoch]] -> [Epoch]
blocks1 = \case
  [b] -> b
  _ -> []

blocks2 :: [[Epoch]] -> ([Epoch], [Epoch])
blocks2 = \case
  [a, b] -> (a, b)
  _ -> ([], [])

------------------------------------------------------------------------------
-- Fragments

{- | Fragments against the target: none on a target without them, and
every product's inputs of an element type it takes (its accumulators are
f32, which every target with fragments has).
-}
checkFragments :: TargetCaps -> IR.Kernel -> [Finding]
checkFragments caps k
  | null caps.capFragments =
      [ Finding Fragments (caps.capName <> " has no fragments, and the kernel declares " <> T.intercalate ", " declared <> ".")
      | not (null declared)
      ]
  | otherwise = nub [locatedAt (Location k.kernelName (reverse place)) f | (place, s) <- placed, f <- product' s]
 where
  placed = concatMap (flatten []) k.kernelBody
  statements = map snd placed
  declared = [n | SFragVar n _ <- statements]
  types = Map.fromList [(n, t) | SFragVar n t <- statements]
  flatten place = \case
    SIf _ c th el -> concatMap (flatten (ifPlace c : place)) th <> concatMap (flatten (elsePlace c : place)) el
    SFor _ v _ _ _ body -> concatMap (flatten (forPlace v : place)) body
    st -> [(place, st)]
  product' = \case
    SFragMma _ a b _ ->
      [ Finding Fragments (caps.capName <> " takes no fragments of " <> scalarTypeName t <> " into a product.")
      | t <- nub [t' | n <- [a, b], Just t' <- [Map.lookup n types]]
      , t `notElem` caps.capFragments
      ]
    _ -> []

------------------------------------------------------------------------------
-- Resources

-- | Every pack against the target's lane loads and its buffer's alignment.
checkPacks :: TargetCaps -> IR.Kernel -> [Finding]
checkPacks caps k = nub [locatedAt (Location k.kernelName (reverse place)) f | (place, s) <- concatMap (flatten []) k.kernelBody, f <- pack s]
 where
  buffers = Map.fromList [(b.bufName, b) | b <- k.kernelBuffers]
  flatten place = \case
    SIf _ c th el -> concatMap (flatten (ifPlace c : place)) th <> concatMap (flatten (elsePlace c : place)) el
    SFor _ v _ _ _ body -> concatMap (flatten (forPlace v : place)) body
    s -> [(place, s)]
  tables = Map.fromList [(t.tableName, t) | t <- k.kernelTables]
  pack = \case
    SLoadPack ns _ b _ -> width b (length ns)
    SStorePack b _ es -> width b (length es)
    SLoadPackVia ns _ t _ _ -> case Map.lookup t tables of
      Just tb -> checkPack caps tb.tableType (fromIntegral (length ns)) tb.tableAlign
      Nothing -> []
    STgStorePack a _ es -> case Map.lookup a arrays of
      Just arr -> checkPack caps arr.tgType (fromIntegral (length es)) (tgAlign arr)
      Nothing -> []
    _ -> []
  arrays = Map.fromList [(a.tgName, a) | a <- k.kernelTgArrays]
  -- The typing check reports unknown buffers.
  width b n = case Map.lookup b buffers of
    Just buf -> checkPack caps buf.bufType (fromIntegral n) buf.bufAlign
    Nothing -> []

checkResources :: TargetCaps -> IR.Kernel -> [Finding]
checkResources caps k =
  [ Finding Target ("The kernel was built for SIMD groups of " <> T.pack (show k.kernelSimdWidth) <> " lanes, which its body takes as given; " <> caps.capName <> "'s have " <> T.pack (show caps.capSimdWidth) <> ".")
  | k.kernelSimdWidth /= caps.capSimdWidth
  ]
    <> checkThreads caps k.kernelThreads
    <> concatMap (\a -> checkTgArrayOf caps a.tgType a.tgCount) k.kernelTgArrays
    <> checkTgTotal caps [(a.tgName, tgBytes a) | a <- k.kernelTgArrays]
    <> checkBuffers caps (fromIntegral (length k.kernelBuffers + length k.kernelTables) + (if null k.kernelParams then 0 else 1))
    <> concat [checkAlign ("Buffer " <> b.bufName) b.bufType b.bufAlign | b <- k.kernelBuffers]
    <> concat [checkAlign ("Table " <> t.tableName) t.tableType t.tableAlign | t <- k.kernelTables]
