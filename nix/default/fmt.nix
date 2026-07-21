{
    perSystem = {
        treefmt = {
            # Used to find the project root
            projectRootFile = "flake.nix";

            programs = {
                deadnix.enable = true;
                statix.enable = true;
                nixfmt = {
                    enable = true;
                    indent = 4;
                };
                prettier = {
                    enable = true;
                    settings.tabWidth = 4;
                };
            };

            settings = {
                formatter = {
                    deadnix.priority = 1;
                    statix.priority = 2;
                    nixfmt.priority = 3;
                    prettier.priority = 4;
                };
            };
        };
    };
}
