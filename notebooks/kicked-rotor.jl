### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 3814d626-327f-11f1-ade5-130603661e5c
# ╠═╡ show_logs = false
begin
    import Pkg

    # activate the shared project environment
    Pkg.activate(@__DIR__)

    # This file needs to be executed from notebooks/ dir
    cd(@__DIR__)
    base_path = ".."
    Pkg.develop(Pkg.PackageSpec(; path=base_path))
    Pkg.precompile()

    using GQML
    using CairoMakie
    using LinearAlgebra
    using FFTW
    using Statistics
end

# ╔═╡ 35e4f6ac-f3b7-497d-a997-4821d024eae8
md"# Classical Kicked Rotor"

# ╔═╡ 311e3ddf-25a0-4c28-ac5d-1f753c2cf7e4
begin
    const N_ckr = 500    # number of step to evolve for
    const K_ckr = 0.7    # coupling strength
end

# ╔═╡ 30c6cfbf-23fd-4faf-95e6-4786316603f2
function ckr_step!(x, p)
    for i in 1:(N_ckr-1)
        p[i+1] = mod((p[i] + K_ckr * sin(x[i])), 2pi)
        x[i+1] = mod((x[i] + p[i+1]), 2pi)
    end
end

# ╔═╡ 7361b49e-4f61-48ba-97e5-87612ae928fa
function plot_ckr_phase_space()
    fig = Figure()
    ax = Axis(fig[1, 1]; aspect=1, xlabel="x", ylabel="p")
    xlims!(ax, 0, 2pi)
    ylims!(ax, 0, 2pi)

    for i in range(0, 2pi, 20)
        x, p = zeros(N_ckr), zeros(N_ckr)
        x[begin] = i
        ckr_step!(x, p)
        scatter!(ax, x, p; markersize=2, color=:black)
    end

    for i in range(0, 2pi, 100)
        x, p = zeros(N_ckr), zeros(N_ckr)
        p[begin] = i
        ckr_step!(x, p)
        scatter!(ax, x, p; markersize=2, color=:black)
    end

    return fig
end

# ╔═╡ 1b4da5fb-b48f-44d5-b154-731c448d1d16
plot_ckr_phase_space()

# ╔═╡ 12323bd8-d246-4f41-b9cc-38bdfa78d172
md"# Quantum Kicked Rotor"

# ╔═╡ 92a62c51-a39e-4163-a981-5555c659a229
begin
    const qubits = 10
    const dims = 2^qubits
    const N_qkr = 1000
    const K_qkr = 12
    const ħₛ = 0.7
end;

# ╔═╡ ea8b11de-9faa-4ccd-9b97-3ec8bab50ee7
md"""
## Full simulation via FFT method

To show the localization of the final state
"""

# ╔═╡ 4ada6f4b-70c0-472c-8d75-c4b5ea89feb1
function QKR!(state)
    m_vec = [0:(dims/2-1); (-dims/2):-1]
    m_vec² = @. m_vec^2
    x_vec = @. cos(2π * (0:(dims-1)) / dims)

    E = zeros(N_qkr + 1)
    E[begin] = 1 / 2 * ħₛ^2 * sum(@. abs2(state) * m_vec²)

    for N in 1:N_qkr
        @. state *= exp(-im * ħₛ * m_vec² / 2)
        ifft!(state)
        @. state *= exp(-im * K_qkr / ħₛ * x_vec)
        fft!(state)
        E[N+1] = 1 / 2 * ħₛ^2 * sum(@. abs2(state) * m_vec²)
    end

    return E
end

# ╔═╡ a6abbf87-1b38-4d50-872e-eb8838999e92
function plot_qkr_energy_evolution(E)
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="N", ylabel="E")

    scatter!(
        ax, E;
        markersize=5, label=:none,
    )

    return fig
end

# ╔═╡ 68d89de7-befd-4bea-bfae-9339cbfe5be4
function plot_qkr_localization(ψ::Vector{ComplexF64})
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xlabel=L"m",
        ylabel=L"\left| \psi(p) \right|^2",
        yscale=log10,
        yticks=LogTicks(LinearTicks(5)),
        yminorticksvisible=true,
        yminorticks=IntervalsBetween(10),
    )

    dims = length(ψ)
    m_vec = [0:(dims/2-1); (-dims/2):-1]

    scatter!(
        ax,
        m_vec, get_centered_amplitudes(ψ);
        markersize=5,
        label=:none,
    )

    return fig
end

# ╔═╡ 4e092315-784a-41d9-ba0e-f30d4f67b5b3
begin
    ψ = zeros(ComplexF64, dims)
    ψ[1] = 1   # m₁ = 1

    E = QKR!(ψ)
end

# ╔═╡ 824f798b-437f-4110-8a94-3a5cd9c41574
plot_qkr_energy_evolution(E)

# ╔═╡ 240a503f-34ea-4396-a641-95555f40241a
plot_qkr_localization(ψ)

# ╔═╡ fea40b3b-dbc3-4059-b41c-b34d520efece
md"""
## Constructing the Floquet operator

To show the localization of the eigenstates
"""

# ╔═╡ 10adf359-d158-4911-88ff-e14af3c6b607
eigenstates = QKRLocalizedDist(;
    n_qubits=10,
    K=12.0,
    ħₛ=0.7,
)

# ╔═╡ 9e0f1db0-d94f-4212-a2f7-a9be795c8fe1
GQML.plot(
    eigenstates;
    title="Exponential localization",
)

# ╔═╡ Cell order:
# ╠═3814d626-327f-11f1-ade5-130603661e5c
# ╟─35e4f6ac-f3b7-497d-a997-4821d024eae8
# ╠═311e3ddf-25a0-4c28-ac5d-1f753c2cf7e4
# ╠═30c6cfbf-23fd-4faf-95e6-4786316603f2
# ╠═7361b49e-4f61-48ba-97e5-87612ae928fa
# ╠═1b4da5fb-b48f-44d5-b154-731c448d1d16
# ╟─12323bd8-d246-4f41-b9cc-38bdfa78d172
# ╠═92a62c51-a39e-4163-a981-5555c659a229
# ╟─ea8b11de-9faa-4ccd-9b97-3ec8bab50ee7
# ╠═4ada6f4b-70c0-472c-8d75-c4b5ea89feb1
# ╠═a6abbf87-1b38-4d50-872e-eb8838999e92
# ╠═68d89de7-befd-4bea-bfae-9339cbfe5be4
# ╠═4e092315-784a-41d9-ba0e-f30d4f67b5b3
# ╠═824f798b-437f-4110-8a94-3a5cd9c41574
# ╠═240a503f-34ea-4396-a641-95555f40241a
# ╟─fea40b3b-dbc3-4059-b41c-b34d520efece
# ╠═10adf359-d158-4911-88ff-e14af3c6b607
# ╠═9e0f1db0-d94f-4212-a2f7-a9be795c8fe1
