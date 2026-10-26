{ mkDerivation, array, base, bytestring, containers, hedgehog, lib
, tasty, tasty-hedgehog, tasty-hunit, text, transformers
}:
mkDerivation {
  pname = "furykernel";
  version = "0.2.0.0";
  src = ./.;
  libraryHaskellDepends = [
    array base bytestring containers text transformers
  ];
  testHaskellDepends = [
    base containers hedgehog tasty tasty-hedgehog tasty-hunit text
  ];
  homepage = "https://github.com/infurer/furykernel";
  description = "A typed eDSL for GPU compute kernels";
  license = lib.meta.getLicenseFromSpdxId "BSD-3-Clause";
}
