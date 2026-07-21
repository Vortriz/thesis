{
    perSystem =
        { pkgs, ... }:
        {
            treefmt = {
                settings.formatter = {
                    jlfmt = {
                        priority = 1;
                        command = pkgs.julia.withPackages.override {
                            augmentedRegistry = pkgs.callPackage ./_registry.nix { };
                        } [ "JuliaFormatter" ];
                        options = [ "${./fmt.jl}" ];
                        includes = [ "*.jl" ];
                    };
                };
            };
        };
}
