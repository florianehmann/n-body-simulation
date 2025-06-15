{
  description = "Project dependencies";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      lib = nixpkgs.lib;

      # Runtime shared libraries needed by X11/OpenGL apps like miniquad
      libs = [
        pkgs.xorg.libX11
        pkgs.xorg.libXcursor
        pkgs.xorg.libXrandr
        pkgs.xorg.libXi
        pkgs.xorg.libXinerama
        pkgs.xorg.libXext
        pkgs.xorg.libXxf86vm
        pkgs.libGL
        pkgs.mesa
        pkgs.libxkbcommon
        pkgs.openssl
      ];

      ldLibPath = lib.concatStringsSep ":" (map (pkg: "${pkg}/lib") libs);

    in {
      devShells = {
        x86_64-linux = {
          default = pkgs.mkShell {
            buildInputs = with pkgs; [
              rustc
              cargo
              gcc
              pkg-config
            ] ++ libs;

            shellHook = ''
              export LD_LIBRARY_PATH=${ldLibPath}:$LD_LIBRARY_PATH
            '';
          };
        };
      };
    };
}
