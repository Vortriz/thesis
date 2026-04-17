pkgs:
pkgs.julia.withPackages.override
    {
        augmentedRegistry = pkgs.callPackage ./_registry.nix { };
        precompile = false;
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
        "Plots"
        "QuantumToolbox"
        "OptimalTransport"
        "ExactOptimalTransport"
        "Combinatorics"
        "Zygote"
        "Enzyme"
        "OffsetArrays"
        "StatsBase"
        "Optimisers"
        "ProgressLogging"
        "Hyperopt"
        "Bessels"
        "FFTW"
        "JLD2"

        # Perf
        "JET"
        "BenchmarkTools"
        "ProfileCanvas"
    ]
