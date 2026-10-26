# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# Repository-wide checks beyond treefmt's own.
{self, ...}: {
  perSystem = {
    config,
    haskellPackages,
    pkgs,
    ...
  }: {
    checks = {
      # Static mode rejects what it should: every file in test/no-compile
      # fails to type-check at one place, with the message its one
      # `-- EXPECT:` line names, found outside the source lines GHC echoes.
      no-compile = let
        ghc = haskellPackages.ghcWithPackages (_: [config.packages.furykernel]);
      in
        pkgs.runCommand "no-compile" {
          nativeBuildInputs = [ghc];
        } ''
          status=0
          for f in ${self}/test/no-compile/*.hs; do
            name=''${f#${self}/}
            want=$(sed -n 's/^-- EXPECT: //p' "$f")
            expects=$(grep -c '^-- EXPECT: ' "$f" || true)
            if [ "$expects" -ne 1 ]; then
              echo "$name: $expects -- EXPECT: lines, not one" >&2
              status=1
            elif ghc -fno-code -outputdir "$TMPDIR/out" "$f" > log 2>&1; then
              echo "$name: compiles, but should fail with: $want" >&2
              status=1
            elif places=$(grep -o -E '^[^ ]+: error:' log | sort -u | wc -l); [ "$places" -ne 1 ]; then
              echo "$name: fails at $places places, not one" >&2
              cat log >&2
              status=1
            elif ! grep -v -E '^ *[0-9]* *[|]' log | tr '\n' ' ' | tr -s ' ' | grep -q -F -- "$want"; then
              echo "$name: fails, but not with: $want" >&2
              cat log >&2
              status=1
            fi
          done
          [ "$status" -eq 0 ]
          touch $out
        '';

      # The library and its tests, under -Werror.
      build = pkgs.haskell.lib.compose.doCheck config.packages.furykernel;

      # The hints `nix fmt` cannot apply on its own.
      hlint =
        pkgs.runCommand "hlint" {
          nativeBuildInputs = [pkgs.hlint];
        } ''
          hlint --hint=${self}/.hlint.yaml ${self}/src ${self}/test
          touch $out
        '';

      # furykernel.nix is cabal2nix's output, except for its `src`, which
      # cabal2nix sets to whatever path it was given.
      cabal2nix-in-sync =
        pkgs.runCommand "cabal2nix-in-sync" {
          nativeBuildInputs = [pkgs.cabal2nix pkgs.diffutils];
        } ''
          cabal2nix ${self} | sed 's|^  src = .*;$|  src = ./.;|' > want.nix
          if ! diff -u ${self}/furykernel.nix want.nix; then
            echo "furykernel.nix is out of date with furykernel.cabal. Regenerate it:" >&2
            echo "  cabal2nix . | sed 's|^  src = .*;\$|  src = ./.;|' > furykernel.nix" >&2
            exit 1
          fi
          touch $out
        '';

      # Every file carries SPDX copyright and license information.
      reuse =
        pkgs.runCommand "reuse-lint" {
          nativeBuildInputs = [pkgs.reuse];
        } ''
          cp -r ${self}/. src
          cd src
          reuse lint
          touch $out
        '';

      # `nix develop`'s shell, the gate's tools in it.
      devshell = config.devShells.default;
    };
  };
}
