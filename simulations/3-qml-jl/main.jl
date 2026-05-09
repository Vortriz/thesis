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

# ╔═╡ 8c3908e0-d000-494e-976e-66181a8766e2
begin
	include("src/base.jl")
	using .QML
end

# ╔═╡ 9628d483-547c-49fa-b4ea-4a7625cded6e
begin
	using Yao
	using YaoPlots
	using Random
	using LinearAlgebra
	using CairoMakie
	using StatsBase
	import Zygote
	import Optimisers

	using BenchmarkTools
	using JET
	using ProgressLogging
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
	arch = ModelArch(
		n_data=1,
		n_ancilla=2,
		n_layers=4,
		ansatz_builder=hardware_efficient_ansatz,
		collapse_method=normal
	)
	
	config = TrainConfig(
		dataset_size=1000,
		batch_size=400,
		target_schedule=:direct,
		epoch_schedule=fill(200, T),
		optimizer=Optimisers.AMSGrad(0.005)
	)

	target_ensemble = gen_dist(
		Val(circle);
		n_qubits=arch.n_data,
		n_samples=config.dataset_size,
	)
end

# ╔═╡ ca80deb6-521f-4ed1-8c72-05ecc2c9b0af
# ╠═╡ disabled = true
#=╠═╡
plot_bloch_sphere(target_ensemble)
  ╠═╡ =#

# ╔═╡ 1984e8ba-e163-41b0-bc30-ed562a8443c2
#=╠═╡
plot_loss_history(loss_history)
  ╠═╡ =#

# ╔═╡ 278ac457-999e-4d92-a2ac-38f46432f0a1
initial_ensemble = gen_dist(
	Val(haar);
	n_qubits=arch.n_data,
	n_samples=config.batch_size,
)

# ╔═╡ ca871011-5c1a-4923-b7b1-667f3473583a
begin
	function train(
		arch::ModelArch,
		config::TrainConfig,
		target_trajectory::Vector{CTBArrayReg},
	)
		params = randn(rng, Float64, (arch.n_params_ppb, config.T))
		loss_history = [zeros(Float64, n) for n in config.epoch_schedule]

		current_ensemble = target_trajectory[end] |> copy

		@progress for t in 1:config.T
			append_qubits!(current_ensemble, arch.n_ancilla)

			current_params = params[:, t]
			opt_state = Optimisers.setup(config.optimizer, current_params)
			
			target_idx = config.target_schedule[t]
			target_matrix = target_trajectory[target_idx].state

			@progress for epoch in 1:config.epoch_schedule[t]
				target_indices = sample(
					1:config.dataset_size,
					config.batch_size,
					replace=false,
				)
				target_batch = target_matrix[:, target_indices]

				loss, grads = loss_and_grads(
					arch,
					rng,
					current_params,
					current_ensemble,
					target_batch,
				)

				Optimisers.update!(opt_state, current_params, grads[1])
				loss_history[t][epoch] = loss
			end

			current_ensemble = apply_pqc(
				arch,
				rng,
				current_params,
				current_ensemble,
			) |> BatchedArrayReg |> transpose_storage
			
			params[:, t] = current_params
		end

		return loss_history, params
	end

	
	
	# Forward pass to create the scrambling trajectory
	# scramble_weight_schedule = fill(1.0, config.T)
	# target_trajectory = scramble(
	# 	arch,
	# 	config,
	# 	rng,
	# 	target_ensemble;
	# 	weight_schedule=scramble_weight_schedule
	# )
	target_trajectory = [target_ensemble, initial_ensemble]

	loss_history, trained_params = train(arch, config, target_trajectory)
end

# ╔═╡ 079dbcb4-d72e-459b-99ca-8ed601b99dde
generated_ensemble = inference(
	arch,
	config,
	rng,
	initial_ensemble,
	trained_params,
)

# ╔═╡ 5e0b724f-e1cc-4d4f-8594-312906c938a9
plot_bloch_sphere(generated_ensemble)

# ╔═╡ 00928b1d-acf6-401b-9215-12a22da65efd
hardware_efficient_ansatz |> typeof

# ╔═╡ fa6a85fe-11e9-4198-95aa-9277c17c0345
cl = gen_dist(
	Val(clustered),
	rng;
	n_qubits=arch.n_data,
	n_samples=config.dataset_size,
)

# ╔═╡ de4c8f20-f166-45b8-9998-61fc8123d381
trajectory = scramble(
	arch,
	config,
	rng,
	cl,
	weight_schedule=range(0.5, 3, config.T) |> collect
)

# ╔═╡ 0a7a1268-2da6-42ab-9bb5-9166291d0e14
trajectory[3] |> plot_bloch_sphere

# ╔═╡ ecccf125-c9cc-4309-9e92-cfba75683803
trajectory[1].state[:, 1:100]

# ╔═╡ 00449b2a-8935-40bc-9f03-d03612816a44
plot_scrambling_decay(
	arch,
	config;
	trajectory=trajectory,
	metric=mmd_distance
)

# ╔═╡ cad94911-7d01-42f8-aaed-a7c9d59db5e0
begin
	ensemble = gen_dist(
		Val(clustered),
		rng;
		n_qubits=arch.n_data,
		n_samples=config.dataset_size,
	)
	append_qubits!(ensemble, arch.n_ancilla)
	
	@code_warntype collapse(
		Val(normal),
		arch,
		rng,
		ensemble,
	)
end

# ╔═╡ Cell order:
# ╟─b25c644c-e852-4151-9a95-0a68a9299036
# ╠═8c3908e0-d000-494e-976e-66181a8766e2
# ╠═9628d483-547c-49fa-b4ea-4a7625cded6e
# ╟─fad927dd-6a9a-4937-ab4e-49ea33477f72
# ╠═2ef55b67-465f-4141-9cbd-93c3068485e4
# ╠═ca80deb6-521f-4ed1-8c72-05ecc2c9b0af
# ╠═ca871011-5c1a-4923-b7b1-667f3473583a
# ╠═1984e8ba-e163-41b0-bc30-ed562a8443c2
# ╠═278ac457-999e-4d92-a2ac-38f46432f0a1
# ╠═079dbcb4-d72e-459b-99ca-8ed601b99dde
# ╠═5e0b724f-e1cc-4d4f-8594-312906c938a9
# ╠═00928b1d-acf6-401b-9215-12a22da65efd
# ╠═fa6a85fe-11e9-4198-95aa-9277c17c0345
# ╠═de4c8f20-f166-45b8-9998-61fc8123d381
# ╠═0a7a1268-2da6-42ab-9bb5-9166291d0e14
# ╠═ecccf125-c9cc-4309-9e92-cfba75683803
# ╠═00449b2a-8935-40bc-9f03-d03612816a44
# ╠═cad94911-7d01-42f8-aaed-a7c9d59db5e0
