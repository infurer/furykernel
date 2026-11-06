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

- `flake.nix`: flake-parts on nixos-26.05, crane and rust-overlay for the
  CPU target's check; systems x86_64-linux, aarch64-linux and
  aarch64-darwin.
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
    GPU family's limits, reflected into `TargetCaps`, and the input types
    of the fragment products it has (`targetFragments`, none by default).
    Target libraries define the instances; the core defines one,
    `FuryKernel.Cpu`'s `Cpu`.
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
    A conversion of a NaN gives a NaN whose sign, payload and quiet bit
    are the target's (`EConvert`'s doc); the interpreter and the CPU
    target quiet it and keep its sign and payload.
    Statements record the region the builder declared (uniform,
    SIMD-uniform, divergent); the gate verifies the declarations. Tables
    of device addresses, bound after the buffers, are read through, and
    written through where their access allows: each entry the start of an
    array (an expert's weights, a block of a KV cache), wherever the
    runtime keeps it.
    Threadgroup arrays lie in the kernel's layout (`tgLayout`): each aligned
    to 16 B, or its element size if larger, and taking its size rounded up
    to that; targets allocate the total.
  - `FuryKernel.Build`: the typed surface. `Sig` declares the interface
    (dimensions, parameters, buffers with access and alignment, threads,
    grid), `Scope` the threadgroup memory, `K r` the body in region `r`.
    Storage types have no `Num` and integers no `FloatScalar`; `fmaE`
    multiplies and adds with one rounding, the only fusion targets do;
    barriers need every thread
    (`AllThreads`), SIMD-group operations and barriers (`simdBarrier`)
    every lane (`Converged`), loads
    and stores the right access; each refusal has its own message.
    `apart` declares that the threadgroups touch disjoint elements of a
    buffer where one stores to it, as the launch's data keeps (rows at
    distinct positions, routes to distinct experts).
    Locals are named by the author (`let_`, `var`) and stay named.
    Buffers carry their target, for the pack rule: `loadPack` and
    `storePack` move a `Pack` of consecutive elements at once, indexed in
    packs, so a pack is aligned whenever its buffer is; `tgStorePack`
    writes two or more elements, as wide as a lane loads, to a threadgroup
    array, aligned for it as
    every threadgroup array is. `table` declares a read-only table and
    `tableOf` one of any access; `loadVia` and `loadPackVia` read through
    one, `storeVia` stores through one, `fragLoadVia`, `fragLoadViaT` and
    `fragStoreVia` move fragments through one. `distinct` declares that
    no two of a table's entries point into one array, `apartVia` that the
    threadgroups touch disjoint elements of its arrays where one stores
    through it. Fragments are the
    8×8 matrices a SIMD group holds together (`fragment`): loaded from
    device or threadgroup memory, plain or transposed (`fragLoad`,
    `fragLoadT`, `fragLoadTg`, `fragLoadTgT`), widened into f32 ones
    element by element (`fragConvert`), multiplied into f32
    accumulators (`fragMma`), their rows scaled each by its own
    (`fragScaleRows`) and stored; every lane takes part
    (`Converged`).
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
    divergent), the regions of barriers, SIMD-group operations and
    fragments (whose indices and strides must be SIMD-uniform), the
    resources, the packs, and fragments of types the target multiplies;
    then, on a kernel that passes all of those, the hazards
    (`FuryKernel.Check.Hazard`): no two threads touch one element of
    memory unordered while one of them stores to it. A barrier orders
    threadgroup memory, a SIMD-group barrier its group's lanes, and
    nothing orders device memory between threads, but the threadgroups of
    a buffer declared `apart` are the launch's to keep apart. Through a
    table, two accesses meet only at one entry if its entries are declared
    `distinct`, and at any two otherwise; `apartVia` keeps its
    threadgroups apart as `apart` does a buffer's. Each pair of
    accesses that could race is a system of affine constraints over both
    threads, their loops, the dimensions and their branches' conditions,
    solved in the integers (`FuryKernel.Check.Affine`, the Omega test, with
    a row-major index taken apart digit by digit first: once for a pair,
    by bounds propagated once for a system); what the check can't rule
    out is a finding. Where a uniform loop's passes hold
    different numbers of barriers, a pair inside it is checked in one pass
    of it, where both threads see the same barriers, and in two, apart by
    at least the barriers between them. In such a pass, what both threads
    compute from the same values is the same, and so is a local the
    divergence analysis finds uniform; a minimum or a maximum is either
    operand; and a uniform branch whose sides' barriers differ leaves at
    least those after each access on its side. Indices are taken as
    computed without wrapping around, which the bounds check of a later
    stage verifies.
    A finding carries its place (`findingLocation`): its kernel, and the
    ifs and loops around what it is about.
  - `FuryKernel.Check.Launch`: launch sizes over the dimensions' legal
    values, the multiples of each one's divisor in its range that a u32
    holds. The gate refuses a dimension without any, and a grid, extent or
    table size that leaves u32 at some step or divides by zero for some of
    them: interval arithmetic bounds each step over a box of legal values, a
    box it leaves open is split in two down to single points, where it is
    exact, and a refusal names the values that break it (or the boxes, 4096
    of them, that could not settle it). The same search gives a grid's
    fewest threadgroups (`lowest`) and shows one size no larger than another
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
    and both sides of a select evaluated); undefined behaviour is an error:
    accesses past what the kernel declares, however much the launch gives,
    dimensions outside their range, integer division by zero or overflow,
    and races: every element remembers who touched it after which barriers,
    and two threads touching one unordered, by the hazard check's rules,
    one of them storing, stop the run (`Race`). A table's entries are
    bound sparsely (`runWithTables`, which returns what the kernel stored
    through its writable ones): an entry not bound is an error where it is
    reached (`UnboundEntry`). A launch is made by `launch` and set by
    `withSeed`, `withTables` and `withSameAs`, which can point entries into
    one array, which a table declared `distinct` refuses
    (`SharedEntries`) and which otherwise share their elements' races.
    Threadgroups run in an order drawn from the seed. `runIR` runs IR the
    gate refused, such as the hazard check's mutants. A fragment product
    adds each element's eight products to its accumulator in f32, in k
    order; GPUs may add in another, so kernels with fragments are held to
    f32 accumulation's bound against the interpreter, not to its bits.
  - `FuryKernel.Reduce`: threadgroup reductions (`tgReduceSum` from -0,
    `tgReduceMax` from -inf) from SIMD-group reductions and a scratch array, and
    split-K's combine (`combineSplits`), which adds each output's partials
    in split order. Their order depends on the thread count, the SIMD
    width and the splits only, so a row's result doesn't depend on the
    batch.
  - `FuryKernel.Tile`: tiles of matrix products. `gemmTile` computes a
    threadgroup's tile of C = A · Bᵀ from element functions the caller
    gives (f32): each step of k staged in threadgroup memory by every
    thread, a run of 4 consecutive elements of A's or 8 of B's a thread,
    stored as packs (the target reads a run's sources at once), multiplied
    as 8×8 f32 fragments by the SIMD groups, then handed back element by
    element through each SIMD group's share of a small array, a few
    fragments a round, so a fragment's layout in the lanes stays the
    target's. An element's sum goes over k in order, eight at a time, the
    same whatever the tile's shape and step, so a row's result depends on
    neither the batch nor the tile.
    Its k is known to be a multiple of its step (a literal one, or a
    dimension declared a multiple of it); a tile that breaks a
    precondition is a finding, and nothing of it is built.
  - `FuryKernel.Abi`: a kernel's binary interface, its buffers and its
    parameter struct in C layout, and the FNV-1a hash of their canonical
    description, which tuned overrides keep and runtimes' generators assert;
    the IR hash, of a checked kernel, covers the body too.
    `FuryKernel.Manifest` describes a
    kernel in JSON (format `furykernel-manifest`, version 1, with the
    target it was checked for; `FuryKernel.Json`: deterministic writing,
    and reading of integer-only JSON with unique keys).
    The launch contract (ranges, extents, tables' sizes, `apart`) is outside
    the hash: `checkOverrides` lets overrides replace a kernel when each has
    its ABI and its dimensions as dimensions, together they take every legal
    value of them, and where one takes the kernel's values its extents and
    tables' sizes are no larger and its buffers `apart` only where the
    kernel's are.
  - `FuryKernel.Variant`: variant spaces. A family declares its axes (the
    choices that change how its kernels compute, as rows, threads, pack
    width and split-K; marked when they decide a reduction's order), its
    shape classes and which of their dimensions are batch sizes.
    `enumerate` prunes every point of a class by legality (the family's
    builder and the gate's rules but the hazard check), divisibility,
    parallelism (at least two threadgroups per core of a measured
    `Profile`), duplicates (the same IR as a kept point) and last the
    hazard check (`checkStaged`), run once for each kernel left, and names
    what is left `family_class_x_axiskey`.
    A launch's fewest threadgroups are the grid's least value over the
    dimensions' legal values.
  - `FuryKernel.Selection`: the committed choice of one variant per
    shape class (`selection.json`, format `furykernel-selection` version
    1, with the GPU profile tuning measured and each pick's IR hash), read
    (its format and version first) and validated against a space and a
    profile: made for that profile, every class picked once, every pick
    surviving pruning and still the kernel it was picked as, and classes
    that differ only in batch size agreeing on the axes that order
    reductions, so that a row's result doesn't depend on its batch.
  - `FuryKernel.Backend`: the interface target libraries implement: checked
    kernels lowered to one translation unit, the options a runtime compiles
    it with (`CompileOptions`: the language version, and `Strict`
    arithmetic, the IR's, with nothing fast-math assumes and no
    multiply-add contracted but an `fmaE`), and its launch tables (entry, binding
    indices, geometry, ABI). `FuryKernel.Emit.Rust`: the
    target-independent pieces of generated Rust (parameter structs with
    their layout asserted, launch sizes in u64, dimensions asserted in
    their ranges), which runtimes' generators combine with their own
    types.
    The gate keeps launch sizes in u32 at every step, so their u64 agrees
    with the interpreter's u32 for dimensions in their ranges.
  - `FuryKernel.Cpu`: the CPU target, `Cpu` with the portable kernels'
    limits, and its backend: checked kernels as Rust, in the interpreter's
    model (every local a vector with a lane per thread, each statement run
    for the active lanes in order, branches splitting them), so that
    runtimes without a GPU run the kernels their GPUs run; fragments are
    64 f32 elements a SIMD group, multiplied as the interpreter does. A
    read-only table is a `&[&[T]]`, a writable one a `&mut [&mut [T]]`,
    whose entries Rust's borrows keep distinct. Its output needs the
    `half` crate. Rsqrt, exp, log and tanh are binary64's, rounded once,
    as the interpreter's: to binary16 by a rounding of its own, since the
    crate's `from_f64` goes through binary32 on x86 and keeps only the
    upper bits in its portable code. An fma is `mul_add` in f32 and goes
    through binary64 in f16.
    It refuses two kernels of one name in a module, and two whose names
    make one Rust type name.
    A kernel asserts its dimensions' ranges and multiples before it
    computes anything from them, so a launch the interpreter refuses
    panics, naming the kernel, the dimension, its value and its range
    (`FuryKernel.Emit.Rust`'s `rustDimChecks`, for runtimes' launch code
    too).
    `checks.cpu` holds its Rust to the interpreter bit for bit, but for the
    NaN payloads of arithmetic (`test/cpu`), and `checks.cpu-clippy` holds
    it to pedantic clippy.
  - `FuryKernel.Program`: the program layer, what a model computes in one
    step. `.Graph`: straight-line dataflow over row values (each of a
    step's m rows one token, its elements in segments such as heads or an
    expert's ranks), made by a small vocabulary of operations (linear
    layers, plain or routed through a table of weights; RMSNorm; rotary
    embeddings; routing; stores into KV caches and attention over them,
    ring or paged; gathers; elementwise work; concatenations of rows)
    from sources: per-row
    activations and integers (each below a bound), weights, tables,
    caches. A store names its cache and its rows' positions and sequences,
    and an attention its keys, values, positions and sequences, by field
    (`StoreAt`, `AttendTo`); RoPE's pairs, GELU's form and a route's
    weights are parameters, with one value each so far (`RopeHalves`,
    `GeluTanh`, `SoftmaxOfTopK`). `.Build` writes programs, sources named
    exactly as given, as the engine binds them. `.Check` is their gate
    (`CheckedProgram`: ids
    unique and increasing in program order, every value made by one
    node and every cache stored into by one, rows and caches of at least
    one element, each index below what it indexes, attention over a ring
    reading no more positions than it holds). `.Eval` gives them
    their meaning in f64, rounding only where a cache stores, to its
    element type (`CacheElem`: bf16 or f16), a step's rows storing and
    attending in row order, as if one at a time
    (so a ring shorter than a step still gives each row its keys; a row
    reaching past its position reads as the row that stored its reach)
    (`evaluateStored`: over what kernels stored, the caches as they left
    them and the rows they staged, each store checked against its
    value's bound). Every value carries a bound, how far a kernel's
    binary32 value can be from it, derived through the program for
    kernels that keep to a contract: binary32 throughout, with +, −, ×,
    /, sqrt and fma correctly rounded, fused or not, sums and products
    in any order, subnormal operands and results perhaps zeros of their
    signs; exp and log within 4 ulp, tanh within 5, sin and cos within 4,
    rsqrt within 2; RMSNorm, RoPE (its angle the binary32 product of the
    position and the frequency), GELU, softcap, routes and attention in
    the forms `Eval.hs` states. A store is right iff it lies between the
    bf16 roundings of the exact value less and plus its bound; past a
    route or an arg-max the bounds leave undecided, and where a value can
    overflow, values are unbounded. Caches are bound sparsely, by the
    slots of a ring or the blocks of a paged pool a step uses (a pool's
    blocks each an array of its own, reached through a table of their
    addresses); it refuses a binding of another size than its source
    declares, and a read or store of a slot or block not bound: over a
    checked program it returns, never throws. `.Plan`
    turns a checked program into steps, one kernel launch each: anchors
    (linear layers, attention) set a step's grid; other operations are
    recomputed in every threadgroup from stored values, or run on a linear
    layer's tiles (attention's hold partials, which only its combine
    reads); a cut comes where an operation needs what several
    threadgroups of a step made; light steps hold stores into caches,
    attention's combine (once: its step stores what later steps read
    and makes the outputs that need nothing more) and the outputs no
    anchor computes. Asked to
    (`optStoreGelus`), a plan stores a GELU an anchor reads first, by a
    light step, and a gather that sums rows, too dear to recompute for
    every output (`nodeCost`), is stored before an anchor reads it. A
    store into
    a ring without room for a step's rows of one sequence
    (`optSequenceRows`) past the positions a row reads is staged: a light
    step keeps its values, the attention reads the step's own positions
    there, and a light step after the last attention reading the ring
    lands them. Gated
    (`optGated`), the routes routed layers read, and those the program
    outputs, are stored by a light step before the first step that reads
    a table, so that an engine can fetch the routed weights, and read
    the other routes, in between. A placement never changes a value's
    arithmetic, only where it runs.
    A combine step names only the caches it lands in, and `nodeCost` what
    recomputing an operation costs, as the planner counts it.
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
  and threadgroup memory, tables and fragments; one kernel's IR and ABI
  hashes pinned.
  Kernels run on the interpreter against f64 specifications: an RMSNorm
  within one bf16 ulp on every SIMD width, and bitwise batch-invariant; a
  bf16 GEMV family with packs and split-K (`Test.FuryKernel.Gemv`), bit
  for bit where f32 sums are exact and within the bound of f32
  accumulation elsewhere, a row's bits the same for every m, place in the
  batch, rows per SIMD group and thread count; a GEMM of fragments
  (`Test.FuryKernel.FragmentTest`) for bf16, f16 and f32 inputs, within
  f32 accumulation's bound, and refused on a target without them; a
  GEMM of tiles (`Test.FuryKernel.TileTest`), partial tiles included,
  against f64, a row's bits the same whatever the rows with it and
  whatever the tile's shape and step.
  Races (`Test.FuryKernel.HazardTest`): kernels that race through
  threadgroup and device memory, fragments and SIMD groups, refused by the
  hazard check with their messages and stopped by the interpreter; the
  library's reductions and tiles pass both, race without their barriers,
  and lose no barrier whose race only the interpreter finds, and so do
  passes over a staged array of as many blocks as loaded, which race a
  barrier short or with their store in a block; a store and a load kept
  apart by a value per pass, a local a search finds, a branch's barrier
  or a minimum pass both, and race in their near misses; a buffer
  declared `apart` passes stores at loaded places, still checked within a
  threadgroup and at run time, and so does a table written through,
  declared `distinct` and `apartVia`. The interpreter runs a table of
  2^30 entries of which four are bound, refuses an unbound entry and
  entries a `distinct` table shares, and finds the race of two shared
  entries in an undeclared one. The Omega test against brute force
  (`Test.FuryKernel.AffineTest`).
  The CPU target's Rust, golden; `checks.cpu` holds it to the
  interpreter's bits and refusals (a dimension out of its range or off
  its multiple). Programs: the gate, the evaluator on small cases
  (caches bound sparsely, a paged pool of 2^33 elements by two of its
  blocks; a step's rows through a ring shorter than the step, as its rows
  one at a time; its bounds refusing wrong stores and a well-conditioned
  value's neighbours, and taking binary32's right ones through
  cancellation, at subnormal and overflow ties, NaN, RoPE far into the
  context and long dots), and the planner's
  steps for a mixture-of-experts decoder layer, a head and an embedding,
  against golden text, stores staged where a ring hasn't room for a
  step's rows.
  `test/no-compile/`: programs static mode must reject, each at one
  place, with the message it must reject them with on its one
  `-- EXPECT:` line.
  `test/cpu/`: `checks.cpu`'s crate; `Cases.hs` writes its kernels as the
  CPU target lowers them, the interpreter's inputs and outputs, and the
  pieces of `FuryKernel.Emit.Rust` that runtimes generate.
- `nix/`: `package.nix` (the library, built with GHC 9.12 under
  `-Werror`), `checks.nix` (the build with its tests, the no-compile tests,
  the CPU target's Rust against the interpreter and under clippy, hlint,
  cabal2nix-in-sync and `reuse lint`), `rust.nix` (crane and the Rust
  toolchain for that check), `devshell.nix` (`checks.devshell` builds its shell) and `treefmt.nix` (`nix fmt`
  and `checks.treefmt`).
- `.gitlab-ci.yml`: GitLab CI on x86_64-linux runners. `checks` runs
  `nix flake check --all-systems`, which builds x86_64-linux's checks, and
  builds the packages of x86_64-linux and aarch64-linux, whose tests run in
  their checkPhase; `publish-github`, on master once those passed, pushes
  master, and the tags it holds, to the public GitHub repository.

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
