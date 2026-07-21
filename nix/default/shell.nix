{
    perSystem =
        { config, ... }:
        {
            devshells.default = {
                devshell = {
                    name = "base";
                    motd = "";
                    startup.default.text = config.pre-commit.shellHook;
                };

                packages = config.pre-commit.settings.enabledPackages;
            };
        };
}
