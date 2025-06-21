pkgs: {
  # All config is built for "x86_64-linux". To use other systems, edit `systems` variable in `flake.nix`.

  # Default (main) environment options
  default = {
    enable = true;

    extraPackages = with pkgs; [
      typst
    ];
  };

  # Python configuration
  python = {
    enable = true;
    package = pkgs.python312;

    # Python tools that you would usually install via `uv tool install` should also be go here
    extraPackages =
      (with pkgs; [
        nodejs # for using copilot in marimo
      ])
      ++ (with pkgs.python312Packages; [
        ruff
        python-lsp-server
      ]);
  };

  # Julia configuration
  julia = {
    enable = true;
    package = pkgs.julia-bin; # It is suggested to use julia-bin
    env = {
      JULIA_NUM_THREADS = "auto";
    };

    # Packages that you want available in the Julia environment (NOT the julia packages themselves)
    extraPackages = with pkgs; [];
  };
}
