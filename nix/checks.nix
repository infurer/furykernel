# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# Repository-wide checks beyond treefmt's own.
{self, ...}: {
  perSystem = {
    config,
    pkgs,
    ...
  }: {
    checks = {
      # The library and its tests, under -Werror.
      build = pkgs.haskell.lib.compose.doCheck config.packages.furykernel;

      # The hints `nix fmt` cannot apply on its own.
      hlint =
        pkgs.runCommand "hlint" {
          nativeBuildInputs = [pkgs.hlint];
        } ''
          hlint ${self}/src
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
    };
  };
}
