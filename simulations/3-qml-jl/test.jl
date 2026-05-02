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
	using LinearAlgebra
	using CairoMakie
	using QuantumToolbox: Bloch, basis, expect, sigmax, sigmay, sigmaz, add_points!, render, rand_unitary
	import Zygote
	import Optimisers
	using StatsBase

	using BenchmarkTools
	using ProgressLogging
end

# ╔═╡ 8c3908e0-d000-494e-976e-66181a8766e2
begin
	include("src/base.jl")
	using .QDDPM
end

# ╔═╡ fad927dd-6a9a-4937-ab4e-49ea33477f72
begin
	YaoPlots.CircuitStyles.linecolor[] = "#ffffff"
	YaoPlots.CircuitStyles.textcolor[] = "#ffffff"
	rng = MersenneTwister(1234)
end

# ╔═╡ 2ef55b67-465f-4141-9cbd-93c3068485e4
begin
	T=4
	model = Model(
		n_data=1,
		n_ancilla=1,
		n_layers=4,

		dataset_size=1000,
		batch_size=100,

		target_schedule=:direct,
		epoch_schedule=fill(400, T),
	)

	target_ensemble = gen_dist(
		Val(clustered),
		rng;
		n_qubits=model.n_data,
		n_samples=model.dataset_size,
		spread=0.05,
	)
end

# ╔═╡ ca80deb6-521f-4ed1-8c72-05ecc2c9b0af
plot_bloch_sphere(target_ensemble)

# ╔═╡ 2d672ebc-7dd9-4c30-aff5-ddf277f4e3a8


# ╔═╡ ca871011-5c1a-4923-b7b1-667f3473583a
begin
	loss_history::Vector{Vector{Float64}} =
		[zeros(Float64, n) for n in model.epoch_schedule]
	params = randn(rng, Float64, (2 * model.n_qubits * model.n_layers, model.T))
	ansatz = hardware_efficient_ansatz(model.n_data, model.n_ancilla, model.n_layers)

	@progress for t in 1:T
		params_t = params[:, t]
		opt_state = Optimisers.setup(Optimisers.AdaGrad(0.005), params_t)
		current_ensemble = gen_dist(
			Val(haar);
			n_qubits=model.n_data,
			n_samples=model.batch_size,
		)
		
		# Pre-calculate dimensions and column offsets for fast gathering
		n_a_dim = 1 << model.n_ancilla
		n_d_dim = 1 << model.n_data
		batch_size = model.batch_size
		col_offsets = (0:batch_size-1) .* n_a_dim
		
		# Pre-extract target state matrix to prevent Zygote from tracking property access
		target_matrix = target_ensemble.state
		
		# Pre-join the constant input register with ancillas outside the loop
		current_ensemble_with_ancilla = join(
			current_ensemble,
			zero_state(model.n_ancilla; nbatch=model.batch_size),
		)
		
		@progress for epoch in 1:model.epoch_schedule[t]
			# Sample a batch from the target ensemble
			target_indices = sample(
				1:model.dataset_size,
				model.batch_size,
				replace=false,
			)
			target_batch = view(target_matrix, :, target_indices)

			loss, grads = Zygote.withgradient(params_t) do p
				output_ensemble = apply(
					current_ensemble_with_ancilla,
					dispatch(ansatz, p)
				)

				indices = Zygote.ignore() do
					res = measure(output_ensemble, 1:model.n_ancilla)
					vec(Int.(res)) .+ 1 .+ col_offsets
				end

				# Manual collapse (Optimized with reshape and slice)
				state_3d = reshape(output_ensemble.state, n_a_dim, n_d_dim, batch_size)
				state_permuted = permutedims(state_3d, (2, 1, 3))
				state_2d = reshape(state_permuted, n_d_dim, :)

				# Extract unnormalized collapsed state using pre-calculated offsets
				collapsed_state = state_2d[:, indices]
				probs = sum(abs2, collapsed_state, dims=1)
				
				# Pre-calculate dot products on unnormalized states for efficiency
				dot_products = target_batch' * collapsed_state
				
				# Normalize squared fidelity instead of states
				fidelity_matrix = abs2.(dot_products) ./ (probs .+ 1e-12)
				
				Γ = Zygote.ignore() do
	                ipot(1.0 .- fidelity_matrix)
	            end

				return -dot(Γ, fidelity_matrix)
			end

			opt_state, params_t = Optimisers.update!(opt_state, params_t, grads[1])
			loss_history[t][epoch] = loss
		end

		params[:, t] = params_t
	end
end

# ╔═╡ Cell order:
# ╟─b25c644c-e852-4151-9a95-0a68a9299036
# ╠═9628d483-547c-49fa-b4ea-4a7625cded6e
# ╠═8c3908e0-d000-494e-976e-66181a8766e2
# ╠═fad927dd-6a9a-4937-ab4e-49ea33477f72
# ╠═2ef55b67-465f-4141-9cbd-93c3068485e4
# ╠═ca80deb6-521f-4ed1-8c72-05ecc2c9b0af
# ╠═2d672ebc-7dd9-4c30-aff5-ddf277f4e3a8
# ╠═ca871011-5c1a-4923-b7b1-667f3473583a
