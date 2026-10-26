-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The binary interface of a kernel: what generated code on both sides of
a launch agrees on.

A kernel's ABI is its buffers, in declaration order, its tables of
addresses after them, and its one parameter struct, laid out as C lays out
a struct of its fields. Its hash covers
names, order, access, types, alignments and offsets: a tuned override of a
kernel must have the same ABI, and the Rust that runtimes' generators emit
asserts at compile time that both sides come from the same one. Launch
geometry (threads, grid) is the backend's choice and outside the hash.

The launch contract (the dimensions' ranges, the extents and the tables'
sizes) is outside the hash too: 'checkOverrides' decides whether overrides
may replace a kernel, their contracts together at least as permissive as
its.

The hashes are FNV-1a over a canonical text, which is simple to recompute
anywhere and is no security measure.
-}
module FuryKernel.Abi (
  Binding (..),
  TableBinding (..),
  Field (..),
  Abi (..),
  abiOf,
  abiDescription,
  checkOverrides,
  checkOverride,
  irHash,
  fnv1a64,
  hex64,
) where

import Data.Bits (xor)
import Data.ByteString qualified as BS
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import Data.Text qualified as T
import Data.Text.Encoding (encodeUtf8)
import Data.Word (Word64)
import FuryKernel.Check (Checked, checkedIR)
import FuryKernel.Check.Launch (Failure (..), Verdict (..), breachText, covers, effort, intersect, kernelBox, launchSize, noLarger, pointText, sizeText)
import FuryKernel.Finding (Finding (..), Rule (..))
import FuryKernel.IR
import FuryKernel.Type (ScalarType, scalarSize, scalarTypeName)
import Numeric (showHex)
import Numeric.Natural (Natural)

data Binding = Binding
  { bindingName :: Name
  , bindingAccess :: Access
  , bindingType :: ScalarType
  , bindingAlign :: Natural
  , bindingExtent :: Expr
  {- ^ Elements the kernel may touch, over the dimensions: a contract the
  launch checks, outside the hash.
  -}
  }
  deriving stock (Eq, Show)

-- | A table of addresses: what its entries point at, and how many there are.
data TableBinding = TableBinding
  { tableBindingName :: Name
  , tableBindingType :: ScalarType
  , tableBindingAlign :: Natural
  , tableBindingEntries :: Expr
  -- ^ A contract the launch checks, outside the hash.
  , tableBindingExtent :: Expr
  -- ^ Elements of each entry's array the kernel may touch, outside the hash.
  }
  deriving stock (Eq, Show)

data Field = Field
  { fieldName :: Name
  , fieldType :: ScalarType
  , fieldOffset :: Natural
  , fieldRole :: ParamRole
  }
  deriving stock (Eq, Show)

data Abi = Abi
  { abiBuffers :: [Binding]
  , abiTables :: [TableBinding]
  -- ^ Bound after the buffers.
  , abiParams :: [Field]
  , abiParamsSize :: Natural
  {- ^ The parameter struct's size, a multiple of its alignment; 0 when
  the kernel has no parameters.
  -}
  , abiHash :: Word64
  }
  deriving stock (Eq, Show)

abiOf :: Kernel -> Abi
abiOf k = abi {abiHash = fnv1a64 (abiDescription abi)}
 where
  abi =
    Abi
      { abiBuffers = [Binding b.bufName b.bufAccess b.bufType b.bufAlign b.bufExtent | b <- k.kernelBuffers]
      , abiTables = [TableBinding t.tableName t.tableType t.tableAlign t.tableEntries t.tableExtent | t <- k.kernelTables]
      , abiParams = fields
      , abiParamsSize = roundUp alignment end
      , abiHash = 0
      }
  (fields, end) = layout 0 k.kernelParams
  layout off = \case
    [] -> ([], off)
    p : ps ->
      let size = fromIntegral (scalarSize p.paramType)
          at = roundUp size off
          (rest, end') = layout (at + size) ps
       in (Field p.paramName p.paramType at p.paramRole : rest, end')
  alignment = maximum (1 : [fromIntegral (scalarSize p.paramType) | p <- k.kernelParams])
  roundUp a n = ((n + a - 1) `div` a) * a

-- | The canonical text the ABI hash is computed over.
abiDescription :: Abi -> Text
abiDescription a =
  T.unlines $
    ["abi 1"]
      <> [ "buffer " <> b.bindingName <> " " <> access b.bindingAccess <> " " <> scalarTypeName b.bindingType <> " align " <> showT b.bindingAlign
         | b <- a.abiBuffers
         ]
      <> ["table " <> t.tableBindingName <> " " <> scalarTypeName t.tableBindingType <> " align " <> showT t.tableBindingAlign | t <- a.abiTables]
      <> ["param " <> f.fieldName <> " " <> scalarTypeName f.fieldType <> " at " <> showT f.fieldOffset | f <- a.abiParams]
      <> ["params size " <> showT a.abiParamsSize]
 where
  access = \case
    ReadOnly -> "read"
    WriteOnly -> "write"
    ReadWrite -> "read_write"

------------------------------------------------------------------------------
-- Overrides

{- | Whether overrides may replace a kernel together, as findings; none
when they may. Each must have the kernel's ABI and take its dimensions as
dimensions; together they must take every legal value of them; and where
one takes the kernel's values, its buffers' extents and its tables' sizes
must be no larger than the kernel's, since its launches keep only the
kernel's contract.
-}
checkOverrides :: Checked t -> [Checked t] -> [Finding]
checkOverrides kernel overrides
  | null os = [Finding Overrides ("No override replaces " <> k.kernelName <> ".")]
  | not (null interface) = interface
  | otherwise = coverage <> concatMap contract os
 where
  k = checkedIR kernel
  os = map checkedIR overrides
  description = T.lines . abiDescription . abiOf
  interface = concatMap interfaceOf os
  interfaceOf o = case [(a, b) | (a, b) <- zipLonger (description o) (description k), a /= b] of
    (a, b) : _ ->
      [ Finding Overrides $
          "The override " <> o.kernelName <> " has another ABI than " <> k.kernelName <> ": " <> case (a, b) of
            (Just x, Just y) -> x <> " where " <> k.kernelName <> " has " <> y <> "."
            (Just x, Nothing) -> x <> ", which " <> k.kernelName <> " lacks."
            (_, y) -> "it lacks " <> fromMaybe "" y <> "."
      ]
    [] ->
      [ Finding Overrides ("The override " <> o.kernelName <> " takes " <> q.paramName <> " as " <> role q.paramRole <> ", and " <> k.kernelName <> " as " <> role p.paramRole <> ".")
      | (p, q) <- zip k.kernelParams o.kernelParams
      , isDim p.paramRole /= isDim q.paramRole
      ]
  zipLonger (x : xs) (y : ys) = (Just x, Just y) : zipLonger xs ys
  zipLonger xs ys = [(Just x, Nothing) | x <- xs] <> [(Nothing, Just y) | y <- ys]
  isDim = \case
    Dim _ -> True
    Scalar -> False
  role r = if isDim r then "a dimension" else "a scalar"
  coverage = case (kernelBox k, traverse kernelBox os) of
    (Just box, Just boxes) -> case covers boxes box of
      Holds -> []
      Breaks pt _ -> [Finding Overrides ("No override of " <> k.kernelName <> " takes " <> pointText pt <> ", which it takes.")]
      Open ->
        [Finding Overrides ("The overrides of " <> k.kernelName <> " could not be shown to take every legal value of its dimensions: " <> showT effort <> " boxes of them leave it open.")]
    -- Kernels through the gate have legal values.
    _ -> []
  contract o = case kernelBox k >>= \kb -> kernelBox o >>= intersect kb of
    -- An override that takes none of the kernel's values is never launched
    -- for it.
    Nothing -> []
    Just box ->
      concat
        [ larger o box ("extent of " <> c.bufName) c.bufExtent b.bufExtent
        | (b, c) <- zip k.kernelBuffers o.kernelBuffers
        ]
        <> concat
          [ larger o box ("number of entries of table " <> c.tableName) c.tableEntries t.tableEntries
              <> larger o box ("extent of each entry of table " <> c.tableName) c.tableExtent t.tableExtent
          | (t, c) <- zip k.kernelTables o.kernelTables
          ]
  larger o box what oe ke = case (launchSize (Map.keysSet box) oe, launchSize (Map.keysSet box) ke) of
    (Just a, Just b) ->
      let theirs = "The override " <> o.kernelName <> "'s " <> what <> ", " <> sizeText a <> ","
          ours = k.kernelName <> "'s, " <> sizeText b
          at pt = if Map.null pt then "" else " at " <> pointText pt
       in case noLarger box a b of
            Holds -> []
            Breaks pt (Larger x y) -> [Finding Overrides (theirs <> " is larger than " <> ours <> "," <> at pt <> ": " <> showT x <> " against " <> showT y <> ".")]
            Breaks pt f -> [Finding Overrides (theirs <> " against " <> ours <> "," <> breachText pt f)]
            Open ->
              [Finding Overrides (theirs <> " could not be shown to be no larger than " <> ours <> ", where both take their dimensions: " <> showT effort <> " boxes of them leave it open.")]
    -- Kernels through the gate have launch sizes.
    _ -> []

-- | 'checkOverrides' for one override, which takes every legal value itself.
checkOverride :: Checked t -> Checked t -> [Finding]
checkOverride kernel override = checkOverrides kernel [override]

{- | The hash of a checked kernel's whole IR, body included: FNV-1a over its
text, which on checked IR fixes the tree.
-}
irHash :: Checked t -> Word64
irHash = fnv1a64 . renderKernel . checkedIR

-- | 64-bit FNV-1a over the UTF-8 bytes of a text.
fnv1a64 :: Text -> Word64
fnv1a64 = BS.foldl' step 0xcbf29ce484222325 . encodeUtf8
 where
  step h b = (h `xor` fromIntegral b) * 0x100000001b3

-- | @0x@ and 16 lowercase hex digits.
hex64 :: Word64 -> Text
hex64 w = "0x" <> T.justifyRight 16 '0' (T.pack (showHex w ""))

showT :: (Show a) => a -> Text
showT = T.pack . show
