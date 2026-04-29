### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ b25c644c-e852-4151-9a95-0a68a9299036
begin
    import Pkg

    # activate the shared project environment
    Pkg.activate(Base.current_project())
    Pkg.instantiate()
end

# ╔═╡ 9628d483-547c-49fa-b4ea-4a7625cded6e
begin
	using Yao
	using YaoPlots
	using Random
end

# ╔═╡ 0933f790-e99d-4b76-b18a-ed3eeb3516a0
rng = MersenneTwister(1234)

# ╔═╡ fad927dd-6a9a-4937-ab4e-49ea33477f72
begin
	YaoPlots.CircuitStyles.linecolor[] = "#ffffff"
	YaoPlots.CircuitStyles.textcolor[] = "#ffffff"
end

# ╔═╡ c925b384-b4cf-45a5-87db-2670cbfbcada
function gen_pqc(n_qubits::Int64, n_layers::Int64)::ChainBlock
	circuit = chain(n_qubits)
	allq = 1:n_qubits

	layer = chain(
		n_qubits,
		chain(
			n_qubits,
			put(i=>chain(Rx(0), Ry(0))) for i in allq
		),
		chain(
			n_qubits,
			chain(cz(i, j) for (i, j) in zip(allq, circshift(allq, -1)))
		),
	)

	push!(circuit, layer^n_layers)

	return circuit
end

# ╔═╡ c7557420-83c6-4e50-8132-d59fbabbf7c5
gen_pqc(3, 3) |> vizcircuit

# ╔═╡ 64f64e4f-469a-4149-9fa3-f613eaf98ecb
begin
	n_qubits = 12
	n_layers = 16
	dataset_size = 1000
	circ = gen_pqc(n_qubits, n_layers)
	reg = zero_state(n_qubits; nbatch=dataset_size)
	dcirc = dispatch(circ, randn(rng, 2*n_qubits*n_layers))
	res = apply(reg, dcirc)
end

# ╔═╡ 5e818cc3-2590-4105-8a85-47114e922737
res.state

# ╔═╡ Cell order:
# ╠═b25c644c-e852-4151-9a95-0a68a9299036
# ╠═9628d483-547c-49fa-b4ea-4a7625cded6e
# ╠═0933f790-e99d-4b76-b18a-ed3eeb3516a0
# ╠═fad927dd-6a9a-4937-ab4e-49ea33477f72
# ╠═c925b384-b4cf-45a5-87db-2670cbfbcada
# ╠═c7557420-83c6-4e50-8132-d59fbabbf7c5
# ╠═64f64e4f-469a-4149-9fa3-f613eaf98ecb
# ╠═5e818cc3-2590-4105-8a85-47114e922737
