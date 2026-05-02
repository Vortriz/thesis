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
# ╠═╡ disabled = true
#=╠═╡
plot_bloch_sphere(target_ensemble)
  ╠═╡ =#

# ╔═╡ ca871011-5c1a-4923-b7b1-667f3473583a
begin
	loss_history::Vector{Vector{Float64}} =
		[zeros(Float64, n) for n in model.epoch_schedule]
	params = randn(rng, Float64, (2 * model.n_qubits * model.n_layers, model.T))
	ansatz = hardware_efficient_ansatz(model.n_data, model.n_ancilla, model.n_layers)

	current_ensemble_ref = Ref{Any}(gen_dist(
		Val(haar);
		n_qubits=model.n_data,
		n_samples=model.batch_size,
	))

	@progress for t in 1:T
		# Create the combined register fresh for this timestep to avoid Zygote closure boxing
		current_ensemble_with_ancilla = join(
			current_ensemble_ref[],
			zero_state(model.n_ancilla; nbatch=model.batch_size),
		)

		current_params = params[:, t]
		opt_state = Optimisers.setup(Optimisers.AdaGrad(0.005), current_params)
		
		# Pre-extract target state matrix to prevent Zygote from tracking property access
		target_matrix = target_ensemble.state
		
		@progress for epoch in 1:model.epoch_schedule[t]
			# Sample a batch from the target ensemble
			target_indices = sample(
				1:model.dataset_size,
				model.batch_size,
				replace=false,
			)
			target_batch = target_matrix[:, target_indices]

			loss, grads = Zygote.withgradient(current_params) do p
				output_ensemble_with_ancilla = apply(
					current_ensemble_with_ancilla,
					dispatch(ansatz, p)
				)

				# Collapse the ancilla qubits
				output_ensemble = Zygote.ignore() do
					copy(output_ensemble_with_ancilla) |> Measure(model.n_qubits; locs=1:model.n_ancilla, remove=true)
				end

				C = 1.0 .- abs2.(target_batch' * output_ensemble.state)
				
				Γ = Zygote.ignore() do
	                ipot(C)
	            end

				return dot(Γ, C)
			end

			opt_state, current_params = Optimisers.update!(opt_state, current_params, grads[1])
			loss_history[t][epoch] = loss
		end

		# Apply the trained circuit for this timestep to the full register
		apply!(
			current_ensemble_with_ancilla,
			dispatch(ansatz, current_params)
		)
		
		# Collapse and remove the ancilla qubits to get the data state
		current_ensemble_ref[] = current_ensemble_with_ancilla |> Measure(model.n_qubits; locs=1:model.n_ancilla, remove=true)
		
		params[:, t] = current_params
	end
end

# ╔═╡ 1984e8ba-e163-41b0-bc30-ed562a8443c2
plot_loss_history(loss_history, "hello")

# ╔═╡ Cell order:
# ╟─b25c644c-e852-4151-9a95-0a68a9299036
# ╠═9628d483-547c-49fa-b4ea-4a7625cded6e
# ╠═8c3908e0-d000-494e-976e-66181a8766e2
# ╠═fad927dd-6a9a-4937-ab4e-49ea33477f72
# ╠═2ef55b67-465f-4141-9cbd-93c3068485e4
# ╠═ca80deb6-521f-4ed1-8c72-05ecc2c9b0af
# ╠═ca871011-5c1a-4923-b7b1-667f3473583a
# ╠═1984e8ba-e163-41b0-bc30-ed562a8443c2
