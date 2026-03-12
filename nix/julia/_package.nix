pkgs:
pkgs.julia.withPackages.override
    {
        augmentedRegistry = pkgs.callPackage ./_registry.nix { };
    }
    [
        "Pluto"
        "ArgParse"
        "LanguageServer"
        "JuliaFormatter"

        # Core
        "Yao"
        "YaoPlots"
        "CairoMakie"
        "WGLMakie"
        "QuantumToolbox"
        "OptimalTransport"
        "ExactOptimalTransport"
        "Combinatorics"
        "Zygote"
        "Enzyme"
        "OffsetArrays"
        "StatsBase"
        "Optimisers"
        "HDF5"
        "ProgressLogging"

        # Perf
        "JET"
        "BenchmarkTools"
        "ProfileCanvas"
    ]
