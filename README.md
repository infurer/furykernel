<!--
SPDX-FileCopyrightText: 2026 Mika Tammi
SPDX-License-Identifier: BSD-3-Clause
-->

# FuryKernel

FuryKernel is a Haskell eDSL for GPU compute kernels. This repository is
its target-agnostic core:

- the typed surface (`FuryKernel`) and the first-order IR it builds, with
  the hardware's limits checked at the type level where a kernel asks for
  it;
- the gate (`FuryKernel.Check`), which every kernel passes before a target
  lowers it: types, uniform control flow around barriers, threadgroup
  memory, launch sizes, and races between threads;
- a reference interpreter (`FuryKernel.Interp`), which runs a kernel as a
  GPU does, lane by lane in SIMD groups;
- the CPU target (`FuryKernel.Cpu`): checked kernels as Rust that computes
  the interpreter's bits;
- the program layer (`FuryKernel.Program`): a model's step as dataflow,
  with its own gate, a planner that turns it into kernel launches, and an
  f64 evaluator.

## Status

Version 0.2, in development: the language, its checks and the interface
targets implement can still change.

## Targets

Each target is a library of its own, which lowers checked kernels to its
hardware's language: furykernel-metal for Apple GPUs and furykernel-ane for
the Apple Neural Engine; furykernel-cuda, furykernel-rocm and furykernel-l0
come later. This repository holds what they share and knows no models.

## Building and testing

```sh
nix build         # the library, its test suite run in the check phase
nix develop       # GHC, cabal and the tools
cabal build       # inside the dev shell
nix flake check   # the build, its tests and every other check
```

## License

BSD 3-Clause, in `LICENSES/BSD-3-Clause.txt`.
