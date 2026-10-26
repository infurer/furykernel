<!--
SPDX-FileCopyrightText: 2026 Mika Tammi
SPDX-License-Identifier: BSD-3-Clause
-->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Project overview

FuryKernel v0.2: a Haskell eDSL for writing GPU compute kernels, with the
hardware's limits checked by the type system where a kernel asks for it.
This repository is the target-agnostic core: the language, its checks, a
reference interpreter and the interface that targets implement. Each target
is its own library and repository (furykernel-metal, furykernel-ane, later
furykernel-cuda, furykernel-rocm, furykernel-l0).

FuryKernel knows no models. Kernels for a particular model live with the
runtime that runs it; this repository only holds the language and
model-agnostic building blocks.

v0.1 lives on the branches `initial_version` and `megakernel_lowering`. It
was a schedule IR, not a kernel language, and is research material: read it
for ideas, don't copy its conventions blindly, and never merge or tag it.

## Commands

```sh
nix develop                # GHC, cabal, ghcid, hlint, cabal2nix, reuse
cabal build                # inside the dev shell
nix fmt
nix flake check --all-systems --no-update-lock-file
```

After changing `furykernel.cabal`, regenerate `furykernel.nix`:

```sh
cabal2nix . | sed 's|^  src = .*;$|  src = ./.;|' > furykernel.nix
```

`nix flake check` evaluates every system but builds only the current
system's checks. The per-commit gate builds them for all systems; on a Mac
the Linux ones go to its remote builder. Flakes only see files git tracks,
so `git add` new files first:

```sh
nix fmt && git diff --exit-code &&
  nix flake check --all-systems --no-update-lock-file &&
  (
    for s in x86_64-linux aarch64-linux aarch64-darwin; do
      nix build --no-link --impure --expr \
        "builtins.attrValues (builtins.getFlake \"git+file://$PWD\").checks.$s" || exit 1
    done
  )
```

## Architecture

- `flake.nix`: flake-parts on nixos-26.05; systems x86_64-linux,
  aarch64-linux and aarch64-darwin.
- `furykernel.cabal`, `src/`: the library, one package. `furykernel.nix` is
  cabal2nix's output, committed so that evaluation never builds anything
  first.
  - `FuryKernel.Type`: the scalar types. Compute types have arithmetic;
    storage types (bf16, i8, u8) are widened and narrowed explicitly, floats
    rounding to nearest, ties to even, integers clamped or wrapped.
  - `FuryKernel.Numerics`: those conversions as kernels perform them,
    exact widening and round-to-nearest-even narrowing, and binary64 to
    binary16 in one rounding, not through binary32 (`narrowF16Double`).
  - `FuryKernel.Target`: the `Target` class, whose associated types state a
    GPU family's limits, reflected into `TargetCaps`. Target libraries
    define the instances; the core defines none.
  - `FuryKernel.Constraint`: each resource rule twice, as a type family for
    static mode and as a term decider for the gate and dynamic mode, which
    reflects runtime numbers into the static constraints: threads,
    threadgroup arrays, packs. Each condition is a named family that both
    the rule's `Bool` form and its enforced constraint are built from. Their
    messages carry the arithmetic. A threadgroup array takes its size rounded
    up to its alignment; an alignment in device memory is a power of two,
    at least 16 B and the element size.
    `FuryKernel.Finding`: what the rules report, a rule and a message
    (`Finding`), and where one has a place, the kernel or program and the
    place in it (`locatedAt`); fields that come later leave both as they
    are.
  - `FuryKernel.IR`: the untyped, first-order, structured IR that every
    check and pass works on, with a deterministic text form for golden
    tests and the IR hash, versioned by its first line (`ir 1`). A kernel
    records the SIMD width its body was built for, which the gate holds to
    the target's, and a NaN literal is its type's canonical NaN. Of
    floats, sqrt is correctly rounded and `EFma`, IEEE 754's
    fusedMultiplyAdd, rounds once; rsqrt, exp, log and tanh are within a
    bound each target states.
    binary32 subnormals are a target's choice (`BinOp`'s doc): it may take
    them as zeros of their signs in arithmetic, comparisons and
    conversions that compute, and give zeros for subnormal results; an
    operation that returns an operand may return it unflushed. binary16
    keeps its subnormals, and so do the interpreter and the CPU target.
    Statements record the region the builder declared (uniform,
    SIMD-uniform, divergent); the gate verifies the declarations.
    Threadgroup arrays lie in the kernel's layout (`tgLayout`): each aligned
    to 16 B, or its element size if larger, and taking its size rounded up
    to that; targets allocate the total.
  - `FuryKernel.Build`: the typed surface. `Sig` declares the interface
    (dimensions, parameters, buffers with access and alignment, threads,
    grid), `Scope` the threadgroup memory, `K r` the body in region `r`.
    Storage types have no `Num` and integers no `FloatScalar`; `fmaE`
    multiplies and adds with one rounding, the only fusion targets do;
    barriers need every thread
    (`AllThreads`), SIMD-group operations every lane (`Converged`), loads
    and stores the right access; each refusal has its own message.
    Locals are named by the author (`let_`, `var`) and stay named.
    Buffers carry their target, for the pack rule: `loadPack` and
    `storePack` move a `Pack` of consecutive elements at once, indexed in
    packs, so a pack is aligned whenever its buffer is.
  - `FuryKernel.Check`: the gate. `check` returns a `Checked` kernel, whose
    constructor is private, or the findings; backends take only `Checked`
    kernels. A kernel keeps the caps of the target it was built for, and
    the gate refuses it for any other; `Checked`, `Kernel` and the builder's
    indexed types are nominal in their parameters, and a kernel's fields
    are read by functions, so neither `coerce` nor a record update changes
    a target, region, access or type. It re-checks the IR whatever built
    it: names, typing, a
    divergence analysis that verifies every declared uniform condition and
    loop bound (a threadgroup load as uniform as its index where no store
    to its array can run in its barrier epoch, every side of every if and
    pass of every loop counted, as a reduction's result is; any other
    divergent), the regions of barriers and SIMD-group operations, the
    resources and the packs.
    A finding carries its place (`findingLocation`): its kernel, and the
    ifs and loops around what it is about.
  - `FuryKernel.Check.Launch`: launch sizes over the dimensions' legal
    values, the multiples of each one's divisor in its range that a u32
    holds. The gate refuses a dimension without any, and a grid or extent
    that leaves u32 at some step or divides by zero for some of them:
    interval arithmetic bounds each step over a box of legal values, a box
    it leaves open is split in two down to single points, where it is exact,
    and a refusal names the values that break it (or the boxes, 4096 of
    them, that could not settle it). The same search gives a grid's fewest
    threadgroups (`lowest`) and shows one size no larger than another
    (`noLarger`, after polynomial normal forms).
  - `FuryKernel.Interp`: the SIMT interpreter, the reference the kernels'
    specifications are checked with. Lanes run in lockstep with active
    masks; arithmetic is the IR's (binary16 rounded per operation, RNE
    narrowing of floats, fma exact and rounded once, rsqrt, exp, log and
    tanh in binary64 and rounded once, binary16 not through binary32,
    integers narrowed clamped or wrapped, NaNs widened
    quiet, minimum and maximum as IEEE 754-2019's minimumNumber and
    maximumNumber with -0 below +0 and two NaNs' quiet NaN, i32 and u32
    addition, subtraction and multiplication wrapping around, SIMD-group
    reductions pairwise, neighbours first, both operands of And and Or
    and both sides of a select evaluated); the stores of a statement land in a
    seeded order, so races show; undefined behaviour is an error: accesses
    past what the kernel declares, however much the launch gives,
    dimensions outside their range, integer division by zero or overflow.
  - `FuryKernel.Reduce`: threadgroup reductions (`tgReduceSum` from -0,
    `tgReduceMax` from -inf) from SIMD-group reductions and a scratch array, and
    split-K's combine (`combineSplits`), which adds each output's partials
    in split order. Their order depends on the thread count, the SIMD
    width and the splits only, so a row's result doesn't depend on the
    batch.
  - `FuryKernel.Abi`: a kernel's binary interface, its buffers and its
    parameter struct in C layout, and the FNV-1a hash of their canonical
    description, which tuned overrides keep and runtimes' generators assert;
    the IR hash, of a checked kernel, covers the body too.
    `FuryKernel.Manifest` describes a
    kernel in JSON (format `furykernel-manifest`, version 1, with the
    target it was checked for; `FuryKernel.Json`: deterministic writing,
    and reading of integer-only JSON with unique keys).
    The launch contract (ranges and extents) is outside the hash:
    `checkOverrides` lets overrides replace a kernel when each has its ABI
    and its dimensions as dimensions, together they take every legal value
    of them, and where one takes the kernel's values its extents are no
    larger.
  - `FuryKernel.Backend`: the interface target libraries implement: checked
    kernels lowered to one translation unit, the options a runtime compiles
    it with (`CompileOptions`: the language version, and `Strict`
    arithmetic, the IR's, with nothing fast-math assumes and no
    multiply-add contracted but an `fmaE`), and its launch tables (entry, binding
    indices, geometry, ABI). `FuryKernel.Emit.Rust`: the
    target-independent pieces of generated Rust (parameter structs with
    their layout asserted, launch sizes in u64), which runtimes' generators
    combine with their own types.
    The gate keeps launch sizes in u32 at every step, so their u64 agrees
    with the interpreter's u32 for dimensions in their ranges.
  - `FuryKernel`: what a kernel author imports.
- `test/`: the test suite, tasty with hedgehog properties, all from one
  seed unless `--hedgehog-replay` gives another. Conversions are
  checked exhaustively where the domain is 16 bits wide, and otherwise
  against exact `Rational` arithmetic. Test targets have SIMD widths from 8
  to 64. Each rule's conditions, as named families and in the decider, are
  checked to agree on sample points.
  The interpreter at its edges: SIMD-group sums in their pairwise order,
  neighbours first (ones at lanes 16 and 17 tell it from lane order and
  from xor 16 first), in f32 and f16; fma exact, at ties, subnormals and
  IEEE 754's special cases; rsqrt, exp, log and tanh rounded once from
  binary64, over every binary16 and where macOS's binary32 exp misrounds;
  integers narrowed clamped and wrapped, and minimum and
  maximum up to what the IR leaves to a target, compared by class; a
  threadgroup sum of -0s is -0.
  The IR's text, golden, for straight lines, packs, an if, a SIMD-group sum
  and threadgroup memory; one kernel's IR and ABI hashes pinned.
  Kernels run on the interpreter against f64 specifications: an RMSNorm
  within one bf16 ulp on every SIMD width, and bitwise batch-invariant; a
  bf16 GEMV family with packs and split-K (`Test.FuryKernel.Gemv`), bit
  for bit where f32 sums are exact and within the bound of f32
  accumulation elsewhere, a row's bits the same for every m, place in the
  batch, rows per SIMD group and thread count.
  `test/no-compile/`: programs static mode must reject, each at one
  place, with the message it must reject them with on its one
  `-- EXPECT:` line.
- `nix/`: `package.nix` (the library, built with GHC 9.12 under
  `-Werror`), `checks.nix` (the build with its tests, the no-compile tests,
  hlint, cabal2nix-in-sync and `reuse lint`), `devshell.nix` (`checks.devshell` builds its shell) and
  `treefmt.nix` (`nix fmt` and `checks.treefmt`).
- `.gitlab-ci.yml`: GitLab CI on x86_64-linux runners. `checks` runs
  `nix flake check --all-systems`, which builds x86_64-linux's checks;
  `publish-github`, on master once those passed, pushes master, and the
  tags it holds, to the public GitHub repository.

## Conventions

- Every file has SPDX copyright and license comments
  (`2026 Mika Tammi`, `BSD-3-Clause`). Files that can't hold comments are
  annotated in `REUSE.toml`.
- Every file type has a formatter in `nix/treefmt.nix`, enabled in the
  commit that adds the first file of that type. Haskell is formatted by
  fourmolu (`fourmolu.yaml`), cabal files by cabal-gild.
- GHC 9.12 (`haskell.packages.ghc912`, set once in `nix/package.nix`),
  GHC2024 and the warnings in `furykernel.cabal`'s `warnings` stanza;
  `-Werror` only in the Nix build, so work in progress still loads in ghci.
  Modules have export lists.
- Nothing host-specific in the repository: no hostnames, absolute paths of
  anyone's machine, or names of private repositories and services. master
  is mirrored to a public GitHub repository in full.

## Git workflow

- One commit per reasonably sized feature. Docs (README, this file) change
  in the same commit as the feature they describe.
- Subjects in nixpkgs style, `<attr or area>: <summary>`. Short bodies: why,
  and facts that aren't obvious from the diff.
- Commit with `git commit -s`, and with an `Assisted-by:` trailer naming
  the AI assistant that helped, such as `Assisted-by: Claude Opus 5.5`,
  directly above the Signed-off-by. No other trailers.
- `flake.lock` changes only together with the input change that needs them.
- Every commit passes the per-commit gate (see Commands).
- Corrections to earlier commits go in as `git commit --fixup=<commit>` (or
  `--squash`) and are autosquashed before the merge request.
- v0.2 work happens on `initialv02` and reaches master through merge
  requests on the GitLab repository, which CI mirrors to GitHub. Never push
  to GitHub by hand. Mika signs the commits with GPG before the merge.

## Common mistakes to avoid

- `path:` flake inputs, or `flake.lock` updates riding along in unrelated
  commits.
- Committing `HANDOFF.md` or other session notes.
- Writing SPDX tag names followed by a colon in prose: `reuse lint` parses
  them anywhere in a file.
