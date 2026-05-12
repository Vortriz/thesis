### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# ╔═╡ 3814d626-327f-11f1-ade5-130603661e5c
# Do not modify or remove this cell!
begin
    import Pkg

    # activate the shared project environment
    Pkg.activate(Base.current_project())
    Pkg.instantiate()
end

# ╔═╡ 6bf43aa5-1296-4880-880c-427c0d3dfea3
begin
    using Plots;
    default(; dpi=300)
    using Bessels
    using LinearAlgebra
    using ProgressLogging
    using FFTW
    using Statistics
end

# ╔═╡ 35e4f6ac-f3b7-497d-a997-4821d024eae8
md"# Classical Kicked Rotor"

# ╔═╡ 6e510e20-4600-40e7-beda-53b59e662931
begin
    const N_ckr = 500
    const K_ckr = 0.7
end

# ╔═╡ 30c6cfbf-23fd-4faf-95e6-4786316603f2
function rotor!(x, p)
    for i in 1:(N_ckr-1)
        p[i+1] = mod((p[i] + K_ckr * sin(x[i])), 2pi)
        x[i+1] = mod((x[i] + p[i+1]), 2pi)
    end
end

# ╔═╡ 1b4da5fb-b48f-44d5-b154-731c448d1d16
begin
    plt_ckr = plot(;
        xlims=(0, 2pi), ylims=(0, 2pi),
        legend=nothing,
        ratio=:equal,
    )

    for i in range(0, 2pi, 20)
        x, p = zeros(N_ckr), zeros(N_ckr)
        x[1] = i
        rotor!(x, p)
        scatter!(x, p; ms=0.4)
    end

    for i in range(0, 2pi, 100)
        x, p = zeros(N_ckr), zeros(N_ckr)
        p[1] = i
        rotor!(x, p)
        scatter!(x, p; ms=0.4)
    end
end

# ╔═╡ 44f61800-1f87-4a7f-b5d5-d969e52e1d45
plt_ckr

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
function QKR(state)
    m_vec = [0:(dims/2-1); (-dims/2):-1]
    m_vec² = @. m_vec^2
    x_vec = @. cos(2π * (0:(dims-1))/dims)

    E = zeros(N_qkr + 1)
    E[1] = 1/2 * ħₛ^2 * sum(@. abs2(state) * m_vec²)

    for N in 1:N_qkr
        @. state *= exp(-im * ħₛ * m_vec² / 2)
        ifft!(state)
        @. state *= exp(-im * K_qkr / ħₛ * x_vec)
        fft!(state)
        E[N+1] = 1/2 * ħₛ^2 * sum(@. abs2(state) * m_vec²)
    end

    return state, E
end

# ╔═╡ 4e092315-784a-41d9-ba0e-f30d4f67b5b3
begin
    state = zeros(ComplexF64, dims)
    pos = 700 #rand(1:dims)
    state[pos] = 1

    state, E = QKR(state)
end

# ╔═╡ 824f798b-437f-4110-8a94-3a5cd9c41574
scatter(
    E;
    ms=1.5, label=:none,
    xlabel="t", ylabel="E",
)

# ╔═╡ 4185c241-78ee-4d8f-bc0d-b1b1bda50458
let m_vec = [0:(dims/2-1); (-dims/2):-1]
    scatter(
        m_vec, abs2.(state);
        yscale=:log10, ms=1.5, label=:none,
        xlabel="m", ylabel="|ψ(p)|²",
    )
end

# ╔═╡ c9e3bc28-cfeb-46ad-82b2-b4fc366dcc59
begin
    m_vec = [0:(dims/2-1); (-dims/2):-1]
    amplitudes_final_state = zeros((dims, dims))
    n_init_states = 500
    for i in 1:n_init_states
        state = zeros(ComplexF64, dims);
        state[i] = 1
        state, _ = QKR(state)
        amplitudes_final_state[:, i] = abs2.(state)
    end
    amplitudes_mean = mean(amplitudes_final_state; dims=2) |> vec
    scatter(
        m_vec, amplitudes_mean;
        yscale=:log10, ms=1.5, label=:none,
        xlabel="m", ylabel="|ψ(p)|² (avg over $(n_init_states))",
    )
end

# ╔═╡ d96111d6-55d7-424d-b651-ddc15e8ad020
# ╠═╡ disabled = true
#=╠═╡
circshift(m_vec, 1024)[1025] # midpoint
  ╠═╡ =#

# ╔═╡ fea40b3b-dbc3-4059-b41c-b34d520efece
md"""
## Constructing the Floquet operator

To show the localization of the eigenstates
"""

# ╔═╡ 8cae7a0d-118a-4240-8eea-2dbbc11239c4
begin
    U = zeros(ComplexF64, (dims, dims))
    Threads.@threads for idx in CartesianIndices(U)
        i, j = idx.I
        m₁, m₂ = m_vec[i], m_vec[j]
        d = m₂ - m₁
        if d > dims/2
            d -= dims
        end
        if d < -dims/2
            d += dims
        end
        U[idx] = ℯ^(-im/2 * ħₛ * m₂^2) * im^d * besselj(d, K_qkr / ħₛ)
    end
end

# ╔═╡ 5d5988d7-d8fd-4390-a982-96a4b5427c21
eigenstates = eigen(U).vectors

# ╔═╡ 2585a5d8-ff6e-495f-ac65-93edd65c4573
begin
    avg_amplitudes = zeros(dims)
    for ϕ in eachcol(eigenstates)[4:4]
        amplitudes = abs2.(ϕ)
        _, idx = findmax(amplitudes)
        circshift!(amplitudes, dims-idx+1)
        avg_amplitudes += amplitudes
    end
    avg_amplitudes /= dims
    scatter(
        m_vec, avg_amplitudes;
        yscale=:log10,
        ms=1.5,
        label=:none, xlabel="m", ylabel="|ψ(p)|²",
    )
end

# ╔═╡ 79a2853d-cac3-42b5-b893-3ed8de1ae0a7
# ╠═╡ disabled = true
#=╠═╡
begin
	plt_test = plot()
	mid = dims÷2
	centered_m_vec = circshift(m_vec, mid)
	good_eigenstates = [3,4,5,9,10,13,16,18,20]
	for ϕ in eachcol(eigenstates)[1:1024]
		amplitudes = abs2.(ϕ)
		_, idx = findmax(amplitudes)
		circshift!(amplitudes, mid)
		circshift!(amplitudes, -idx)
		# amplitudes[mid:mid-idx], amplitudes[mid-idx:mid] .= 1e-30, 1e-30
		scatter!(
			abs2.(ϕ),
			# m_vec, circshift(abs2.(ϕ), -idx),
			# centered_m_vec, amplitudes,
			label=:none,
			yscale=:log10,
			ms=1.55, msw=0,
			xlabel="m", ylabel="|ψ(p)|²",
		)
	end
	plt_test
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─3814d626-327f-11f1-ade5-130603661e5c
# ╠═6bf43aa5-1296-4880-880c-427c0d3dfea3
# ╟─35e4f6ac-f3b7-497d-a997-4821d024eae8
# ╠═6e510e20-4600-40e7-beda-53b59e662931
# ╠═30c6cfbf-23fd-4faf-95e6-4786316603f2
# ╠═1b4da5fb-b48f-44d5-b154-731c448d1d16
# ╠═44f61800-1f87-4a7f-b5d5-d969e52e1d45
# ╟─12323bd8-d246-4f41-b9cc-38bdfa78d172
# ╠═92a62c51-a39e-4163-a981-5555c659a229
# ╟─ea8b11de-9faa-4ccd-9b97-3ec8bab50ee7
# ╠═4ada6f4b-70c0-472c-8d75-c4b5ea89feb1
# ╠═4e092315-784a-41d9-ba0e-f30d4f67b5b3
# ╠═824f798b-437f-4110-8a94-3a5cd9c41574
# ╠═4185c241-78ee-4d8f-bc0d-b1b1bda50458
# ╠═c9e3bc28-cfeb-46ad-82b2-b4fc366dcc59
# ╠═d96111d6-55d7-424d-b651-ddc15e8ad020
# ╟─fea40b3b-dbc3-4059-b41c-b34d520efece
# ╠═8cae7a0d-118a-4240-8eea-2dbbc11239c4
# ╠═5d5988d7-d8fd-4390-a982-96a4b5427c21
# ╠═2585a5d8-ff6e-495f-ac65-93edd65c4573
# ╠═79a2853d-cac3-42b5-b893-3ed8de1ae0a7
