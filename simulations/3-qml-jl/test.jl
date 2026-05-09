### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ c77d87a0-4a3c-11f1-b15a-5db1dff58976
begin
    import Pkg

    # activate the shared project environment
    Pkg.activate(Base.current_project())
    Pkg.instantiate()
end

# ╔═╡ a761dda7-0017-4ca4-bd7e-c9d327df65d0
# ╠═╡ show_logs = false
begin
	include("src/base.jl")
	using .QML
end

# ╔═╡ 720275a3-0f9e-49a7-8b4d-2e96f70f1a78
begin
	using Yao
	using Random
	using LinearAlgebra
	using CairoMakie
	using StatsBase
	import Zygote
	import Optimisers

	using ProgressLogging
	using BenchmarkTools
	using JET
	using ProfilePerfetto
end

# ╔═╡ f87a8c67-a279-49d8-b20f-c2a5a07423a8
begin
	using YaoPlots
	YaoPlots.darktheme!()
end

# ╔═╡ 0dc82d60-ad67-4cd1-a501-b9ecea612984
const rng = MersenneTwister(1234)

# ╔═╡ 7e386bf9-3d9e-4852-b7cf-67f2b3ef2f61
begin
	T=2
	arch = ModelArch(
		n_data=5,
		n_ancilla=2,
		n_layers=3,
		ansatz_builder=HEA,
		collapse_method=alternate,
	)
	
	config = TrainConfig(
		dataset_size=1000,
		batch_size=100,
		target_schedule=:direct,
		epoch_schedule=fill(500, T),
		optimizer=Optimisers.AMSGrad(0.1),
	)

	target_ensemble = gen_dist(
		Val(clustered),
		rng;
		n_qubits=arch.n_data,
		n_samples=config.dataset_size,
	)
	initial_ensemble = gen_dist(
		Val(haar);
		n_qubits=arch.n_data,
		n_samples=config.batch_size,
	)
end;

# ╔═╡ 4c5794a9-c5e1-46f8-84b6-195d91fe63d7
plot_bloch_sphere(target_ensemble)

# ╔═╡ f845e549-9e6d-4d25-b379-55ecafa7c559
function train(
	arch::ModelArch,
	config::TrainConfig,
	target_trajectory::Vector{CTBArrayReg},
)
	params = rand(rng, Float64, (arch.n_params_ppb, config.T))
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
			target_batch = @view target_matrix[:, target_indices]

			loss, grads = loss_and_grads(
					arch,
					current_params,
					current_ensemble,
					target_batch,
				)

			Optimisers.update!(opt_state, current_params, grads[1])
			loss_history[t][epoch] = loss
		end

		current_ensemble = apply_pqc(
			arch,
			current_ensemble,
			current_params,
		) |> BatchedArrayReg |> transpose_storage
		
		params[:, t] = current_params
	end

	return loss_history, params
end

# ╔═╡ b5b829f9-931e-451c-b734-ff5c8bff8eec
begin
	target_trajectory = [target_ensemble, initial_ensemble]
	loss_history, trained_params = train(arch, config, target_trajectory)
end

# ╔═╡ 99df8fa2-8bff-4ba6-8dfd-4fe892815962
generated_ensemble = inference(
	arch,
	config,
	initial_ensemble,
	trained_params,
)

# ╔═╡ da5377db-aa38-4852-9062-ed7b888c0738
plot_bloch_sphere(generated_ensemble)

# ╔═╡ 55881a2c-598c-48f2-b4fa-eab7e96a9ba4
plot_loss_history(loss_history)

# ╔═╡ Cell order:
# ╟─c77d87a0-4a3c-11f1-b15a-5db1dff58976
# ╠═a761dda7-0017-4ca4-bd7e-c9d327df65d0
# ╠═720275a3-0f9e-49a7-8b4d-2e96f70f1a78
# ╟─f87a8c67-a279-49d8-b20f-c2a5a07423a8
# ╟─0dc82d60-ad67-4cd1-a501-b9ecea612984
# ╠═7e386bf9-3d9e-4852-b7cf-67f2b3ef2f61
# ╠═4c5794a9-c5e1-46f8-84b6-195d91fe63d7
# ╠═f845e549-9e6d-4d25-b379-55ecafa7c559
# ╠═b5b829f9-931e-451c-b734-ff5c8bff8eec
# ╠═99df8fa2-8bff-4ba6-8dfd-4fe892815962
# ╠═da5377db-aa38-4852-9062-ed7b888c0738
# ╠═55881a2c-598c-48f2-b4fa-eab7e96a9ba4
