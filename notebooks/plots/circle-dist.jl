### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ e5838e3c-a6b4-11f1-85d4-956007b80228
begin
    import Pkg

    # activate the shared project environment
    Pkg.activate(joinpath(@__DIR__, ".."))

    # This file needs to be executed from notebooks/ dir
    cd(joinpath(@__DIR__, ".."))
    base_path = ".."
    Pkg.develop(Pkg.PackageSpec(; path=base_path))
    Pkg.precompile()

    using GQML
    using CairoMakie: save
end

# ╔═╡ 96922a17-256b-4781-a358-61a6293496f6
begin
    const n = 10000
    dist1 = CircleDist(; n_samples=n)
    dist2 = HaarDist(; n_qubits=1, n_samples=n)
end;

# ╔═╡ 74533573-287d-481c-8b4b-9adfe581d633
begin
    plot1 = plot_bloch(dist1; square=true)
    save("../assets/images/circle-dist.svg", plot1)
    plot1
end

# ╔═╡ 5d9a7b51-4287-4d30-a724-0ff1e40e328b
begin
    plot2 = plot_bloch(dist2; square=true)
    save("../assets/images/haar-dist.svg", plot2)
    plot2
end

# ╔═╡ 0fcb83fa-132d-4e96-93cb-2b5c03363370
mmd_distance(
    dist1.register.state,
    dist2.register.state,
)

# ╔═╡ Cell order:
# ╠═e5838e3c-a6b4-11f1-85d4-956007b80228
# ╠═96922a17-256b-4781-a358-61a6293496f6
# ╠═74533573-287d-481c-8b4b-9adfe581d633
# ╠═5d9a7b51-4287-4d30-a724-0ff1e40e328b
# ╠═0fcb83fa-132d-4e96-93cb-2b5c03363370
