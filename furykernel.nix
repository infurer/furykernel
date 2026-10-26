{ mkDerivation, base, lib }:
mkDerivation {
  pname = "furykernel";
  version = "0.2.0.0";
  src = ./.;
  libraryHaskellDepends = [ base ];
  homepage = "https://github.com/infurer/furykernel";
  description = "A typed eDSL for GPU compute kernels";
  license = lib.meta.getLicenseFromSpdxId "BSD-3-Clause";
}
