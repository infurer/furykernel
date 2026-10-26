-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | Manifests: what a generated kernel is, for tools and people. The
generated code is the source of truth; a manifest describes it.

A manifest is a JSON object, its first fields the format,
@"format": "furykernel-manifest"@, its version, 1, and the target the
kernel was checked for.
-}
module FuryKernel.Manifest (
  manifestFormat,
  kernelManifest,
) where

import Data.Text (Text)
import FuryKernel.Abi
import FuryKernel.Check (Checked, checkedIR)
import FuryKernel.IR
import FuryKernel.Json (Json (..))
import FuryKernel.Target (Target, TargetCaps (..), targetCaps)
import FuryKernel.Type (scalarTypeName)

-- | The value of the @format@ field.
manifestFormat :: Text
manifestFormat = "furykernel-manifest"

kernelManifest :: forall t. (Target t) => Checked t -> Json
kernelManifest c =
  JObject $
    [ ("format", JString manifestFormat)
    , ("version", JInt 1)
    , ("target", JString (targetCaps @t).capName)
    , ("name", JString k.kernelName)
    , ("abi_hash", JString (hex64 abi.abiHash))
    , ("ir_hash", JString (hex64 (irHash c)))
    , ("threads", JInt (toInteger k.kernelThreads))
    , ("groups", JString (renderExpr k.kernelGroups))
    , ("buffers", JArray (map buffer abi.abiBuffers))
    ]
      -- Only kernels with tables say so, so that the others read as before.
      <> [("tables", JArray (map table abi.abiTables)) | not (null abi.abiTables)]
      <> [ ("params", JArray (map field abi.abiParams))
         , ("params_size", JInt (toInteger abi.abiParamsSize))
         , ("threadgroup_memory", JInt (toInteger (snd (tgLayout k.kernelTgArrays))))
         ]
 where
  k = checkedIR c
  abi = abiOf k
  table t =
    JObject
      [ ("name", JString t.tableBindingName)
      , ("type", JString (scalarTypeName t.tableBindingType))
      , ("align", JInt (toInteger t.tableBindingAlign))
      , ("entries", JString (renderExpr t.tableBindingEntries))
      , ("extent", JString (renderExpr t.tableBindingExtent))
      ]
  buffer b =
    JObject
      [ ("name", JString b.bindingName)
      , ("access", JString (access b.bindingAccess))
      , ("type", JString (scalarTypeName b.bindingType))
      , ("align", JInt (toInteger b.bindingAlign))
      , ("extent", JString (renderExpr b.bindingExtent))
      ]
  field f =
    JObject $
      [ ("name", JString f.fieldName)
      , ("type", JString (scalarTypeName f.fieldType))
      , ("offset", JInt (toInteger f.fieldOffset))
      ]
        <> case f.fieldRole of
          Scalar -> []
          Dim r ->
            [
              ( "dim"
              , JObject
                  [ ("min", JInt (toInteger r.dimMin))
                  , ("max", JInt (toInteger r.dimMax))
                  , ("multiple_of", JInt (toInteger r.dimMultipleOf))
                  ]
              )
            ]
  access = \case
    ReadOnly -> "read"
    WriteOnly -> "write"
    ReadWrite -> "read_write"
