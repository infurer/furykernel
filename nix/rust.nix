# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# The Rust toolchain from rust-overlay (test/cpu/rust-toolchain.toml) and
# crane, for checks.cpu's crate. One buildDepsOnly produces its
# dependencies' cargoArtifacts, which the check's run and its clippy reuse.
{inputs, ...}: {
  perSystem = {system, ...}: let
    pkgs = import inputs.nixpkgs {
      inherit system;
      overlays = [inputs.rust-overlay.overlays.default];
    };
    rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ../test/cpu/rust-toolchain.toml;
    craneLib = (inputs.crane.mkLib pkgs).overrideToolchain rustToolchain;
    # The crate's sources are generated (checks.nix); its dependencies come
    # from the committed Cargo.lock alone.
    cpuCheckArgs = {
      pname = "furykernel-cpu-check";
      version = "0.0.0";
      strictDeps = true;
      cargoVendorDir = craneLib.vendorCargoDeps {cargoLock = ../test/cpu/Cargo.lock;};
    };
  in {
    _module.args = {
      inherit craneLib rustToolchain cpuCheckArgs;
      cpuCheckArtifacts = craneLib.buildDepsOnly (cpuCheckArgs
        // {
          src = pkgs.lib.fileset.toSource {
            root = ../test/cpu;
            fileset = pkgs.lib.fileset.unions [../test/cpu/Cargo.toml ../test/cpu/Cargo.lock];
          };
        });
    };
  };
}
