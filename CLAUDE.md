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
    reflects runtime numbers into the static constraints. Each condition is
    a named family that both the rule's `Bool` form and its enforced
    constraint are built from. Their messages carry the arithmetic.
    A threadgroup array takes its size rounded up to its alignment; an
    alignment in device memory is a power of two, at least 16 B and the
    element size.
    `FuryKernel.Finding`: what the rules report, a rule and a message
    (`Finding`), and where one has a place, the kernel or program and the
    place in it (`locatedAt`); fields that come later leave both as they
    are.
- `test/`: the test suite, tasty with hedgehog properties, all from one
  seed unless `--hedgehog-replay` gives another. Conversions are
  checked exhaustively where the domain is 16 bits wide, and otherwise
  against exact `Rational` arithmetic. Test targets have SIMD widths from 8
  to 64. Each rule's conditions, as named families and in the decider, are
  checked to agree on sample points.
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
