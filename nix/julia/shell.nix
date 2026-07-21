{ inputs, ... }:
{
    perSystem =
        {
            lib,
            pkgs,
            system,
            ...
        }:
        {
            devshells.julia = {
                devshell = {
                    name = "julia";
                    motd = "";
                };

                commands = [
                    {
                        name = "update-registry";
                        category = "[julia]";
                        help = "Update Julia package registry";
                        command = lib.getExe (
                            pkgs.writers.writePython3Bin "update-registry" {
                                libraries = [ inputs.nima.packages.${system}.default ];
                            } ./update-registry.py
                        );
                    }
                ];

                env = [
                    {
                        name = "JULIA_NUM_THREADS";
                        value = "auto";
                    }
                ];

                packages = [ pkgs.nix-prefetch-git ];
            };
        };
}
