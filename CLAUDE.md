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

## Conventions

- Every file has SPDX copyright and license comments
  (`2026 Mika Tammi`, `BSD-3-Clause`). Files that can't hold comments are
  annotated in `REUSE.toml`.
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
- Corrections to earlier commits go in as `git commit --fixup=<commit>` (or
  `--squash`) and are autosquashed before the merge request.
- v0.2 work happens on `initialv02` and reaches master through merge
  requests on the GitLab repository, which CI mirrors to GitHub. Never push
  to GitHub by hand. Mika signs the commits with GPG before the merge.

## Common mistakes to avoid

- Committing `HANDOFF.md` or other session notes.
- Writing SPDX tag names followed by a colon in prose: `reuse lint` parses
  them anywhere in a file.
