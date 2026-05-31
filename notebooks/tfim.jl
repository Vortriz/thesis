### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 62021008-0938-432e-9f97-73c5e4d64513
begin
    import Pkg

    # activate the shared project environment
    Pkg.activate(@__DIR__)

    # This file must always be executed from notebooks/ dir
    base_path = ".."
    Pkg.develop(Pkg.PackageSpec(; path=base_path))
    Pkg.instantiate()

    using GQML
    using QuantumToolbox
    using LinearAlgebra
    using CairoMakie
end

# ╔═╡ 8bc42b3f-8bec-47c7-9158-0e671185af85
begin
    const n = 4
    const g = 1
end;

# ╔═╡ 09a3be29-1f6e-4dd3-84fe-3ebff853aee5
function ground(H::AbstractQuantumObject{Operator})
    evd = eigenstates(H)
    return real(evd.values[begin]), evd.vectors[:, begin]
end

# ╔═╡ c3ab47f6-89a6-4cce-8b3f-a12696f345cf
md"""
Magnetization is given by:

$$M = \frac{\sum_i \braket{Z_i}}{N}$$

In a true finite-size system, the ground state $\ket{\psi_0}$ is a symmetric superposition and hence the expectation value comes out to be zero.

So we calculate the classical magnetization of each individual basis state $c$, multiply it by its probability $\left\lvert \braket{c | \psi_0} \right\rvert^2$ and takes the absolute value before summing them up:

$$M = \sum_c \left\lvert \braket{c | \psi_0} \right\rvert^2 \cdot \left\lvert \frac1N \sum_{i=1}^{N} s_{i}^{(c)} \right\rvert$$
"""

# ╔═╡ 2604b2b4-d88d-44ff-adcb-c5715b1437f8
function plot_tfim_stats(n::Int64)
    ref_ground_energy, ref_ground_state =
        gen_tfim_hamiltonian(; n_qubits=n, g=0.0) |> ground

    N = 100
    g_vals = range(0.1, 10; length=N)
    energy_vals = zeros(Float64, N)
    fidelity_vals = zeros(Float64, N)
    magnetization_vals = zeros(Float64, N)

    for (i, gᵢ) in enumerate(g_vals)
        ground_energy, ground_state = gen_tfim_hamiltonian(; n_qubits=n, g=gᵢ) |> ground
        energy_vals[i] = ground_energy
        fidelity_vals[i] = abs2(ref_ground_state' * ground_state)
        magnetization_vals[i] = magnetization(ground_state)
    end

    fig = Figure(; size=(800, 1000))
    ax_e = Axis(
        fig[1, 1];
        ylabel=L"E",
    )
    ax_f = Axis(
        fig[2, 1];
        ylabel=L"F",
    )
    ax_m = Axis(
        fig[3, 1];
        ylabel=L"M",
    )

    lines!(ax_e, g_vals, energy_vals)
    lines!(ax_f, g_vals, fidelity_vals)
    lines!(ax_m, g_vals, magnetization_vals)

    for ax in [ax_e, ax_f, ax_m]
        ax.xlabel = L"g"
        ax.xgridvisible = false
        ax.ygridvisible = false
        ax.xscale = log10
        ax.xticks = LogTicks(-1:1)
        ax.xminorticksvisible = true
        ax.xminorticks = IntervalsBetween(10)
    end

    hidexdecorations!(ax_e)
    hidexdecorations!(ax_f)

    return fig
end

# ╔═╡ 5e5ff9a7-df48-460f-9033-32cc2b5707fb
plot_tfim_stats(n)

# ╔═╡ f8dc162d-21d5-430f-9ef0-fb894e968a34
tfim_dist = TFIMDist(;
    n_qubits=n,
    g=range(0.2, 0.4; length=10000) |> collect,
)

# ╔═╡ 0dcbb940-3446-4b8c-9cda-9061a4d379b4
scrambled_dist = scramble(;
    n_qubits=n,
    distribution=tfim_dist,
    weight_schedule=range(3, 4; length=4) |> collect,
)[end]

# ╔═╡ 0012a59f-0a5a-458b-bb9e-dbc53070923e
plot_tfim_magnetization_dist(tfim_dist.register)

# ╔═╡ 74c77d23-42c7-4759-b959-a65168eb9a58
plot_tfim_magnetization_dist(scrambled_dist.register)

# ╔═╡ Cell order:
# ╟─62021008-0938-432e-9f97-73c5e4d64513
# ╠═8bc42b3f-8bec-47c7-9158-0e671185af85
# ╟─09a3be29-1f6e-4dd3-84fe-3ebff853aee5
# ╟─c3ab47f6-89a6-4cce-8b3f-a12696f345cf
# ╠═2604b2b4-d88d-44ff-adcb-c5715b1437f8
# ╠═5e5ff9a7-df48-460f-9033-32cc2b5707fb
# ╠═f8dc162d-21d5-430f-9ef0-fb894e968a34
# ╠═0dcbb940-3446-4b8c-9cda-9061a4d379b4
# ╠═0012a59f-0a5a-458b-bb9e-dbc53070923e
# ╠═74c77d23-42c7-4759-b959-a65168eb9a58
