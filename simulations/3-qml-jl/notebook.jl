### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# ╔═╡ b37d79fa-f9a8-11f0-8be3-37c0117d6175
begin
    import Pkg

    # activate the shared project environment
    Pkg.activate(Base.current_project())
    Pkg.instantiate()
end

# ╔═╡ 9ba0daf4-2b8e-47c8-beb4-796bfa13f4f5
begin
	include("src/base.jl")
	using .QDDPM
end

# ╔═╡ c9dfae33-aef0-42e5-b43c-87d542a1b428
using Random

# ╔═╡ 91c975f1-500d-4f56-8dc0-65075bc54974
using Optimisers

# ╔═╡ 5de59778-07cd-496b-84ec-4700db380ea6
using OffsetArrays

# ╔═╡ 3cc2773d-624a-40a1-8839-8e1efde82147
# ╠═╡ disabled = true
#=╠═╡
using JET, BenchmarkTools
  ╠═╡ =#

# ╔═╡ 06399d96-599f-40ab-b2f7-f8f6955617ca
begin
	const T = 4
	model = Model(
	    n_qubits = 6,
	    n_ancilla = 3,
	    T = T,
	    forward_ensemble_size = 1000,
	    n_layers = 18,
	    backward_ensemble_size = 100,
	    rng = MersenneTwister(124),
	)

	initialize_forward_ensemble!(model; spread=0.05)
	scramble!(model; weight_schedule=logrange(0.8, 2.4; length=T))
	# scramble!(model; weight_schedule=logrange(0.4, 1.3; length=T))
	# scramble!(model; weight_schedule=range(0.4, 1.2; length=T))

	training_strategy = GradZygote(
		loss_function = wasserstein_distance,
		optimizer = Optimisers.AMSGrad(0.042169650342858224),
		iter_schedule = [1000, 1000, 1000, 1000],
	)
	# training_strategy = GradEnzyme(
	# 	loss_function = wasserstein_distance,
	# 	iter_schedule = vcat(fill(1700, 2), fill(2000, T-3), fill(1800, 1)),
	# 	# iter_schedule = intrange(1200, 1100; length=T),
	# 	hyper_params = (lr=0.005,)
	# )
	# training_strategy = DirectGradZygote(
	# 	loss_function = wasserstein_distance,
	# 	optimizer = Optimisers.AMSGrad(0.01),
	# 	n_iters = 300
	# )
	# training_strategy = QNSPSA(;
	# 	loss_function = wasserstein_distance,
	# 	iter_schedule = fill(1500, T),
	# 	# iter_schedule = [1500, 1500, 1, 1, 1, 1],
	# 	# hyper_params = (η=2*1e-3, ϵ=5e-2, β=1e-2, history_length=3), # good
	# 	hyper_params = (η=1e-2, ϵ=5e-2, β=1e-2, history_length=5), # good (nq=2)
	# 	# hyper_params = (η=7e-2, ϵ=5e-2, β=1e-2, history_length=3),
	# )
	# training_strategy = DirectQNSPSA(
	# 	loss_function = wasserstein_distance,
	# 	n_iters = 2000,
 #    	# hyper_params = (η=1e-2, ϵ=5e-2, β=1e-2, history_length=5),
	# 	hyper_params = (η=7e-2, ϵ=5e-2, β=1e-1, history_length=5),
	# )
	# training_strategy = Rotosolve(
	# 	model;
	# 	loss_function = wasserstein_distance,
	# 	iter_schedule = intrange(40, 30; length=T),
	# )
	# training_strategy = SPSA(;
	# 	loss_function = wasserstein_distance,
	# 	iter_schedule = intrange(20, 4; length=T),
	# 	# hyper_params = (c=0.01, γ=1/6, α=1.0),
	# 	hyper_params = (c=0.01, γ=0.101, α=0.602),
	# )
	# training_strategy = ZygoteAD(
	# 	# model;
	# 	;
	# 	loss_function = wasserstein_distance_zygote,
	# 	iter_schedule = intrange(100, 400; length=T),
	# 	learning_rate = 0.005,
	# )
	# training_strategy = EnzymeAD(
	# 	model;
	# 	optimizer = Optimisers.Adam(0.005),
	# 	iter_schedule = intrange(1, 2; length=T)
	# )
	train!(model, training_strategy)
end;

# ╔═╡ 0d3e791f-4d38-4c2a-8e56-180cf0eac7cf
# ╠═╡ disabled = true
#=╠═╡
begin
	E1 = model.forward_ensembles[1:100] |> ensemble_to_matrix
	E2 = model.forward_ensembles[1:100] |> ensemble_to_matrix
	C = 1 .- abs2.(E1' * E2)
	@btime ipot($C)
end
  ╠═╡ =#

# ╔═╡ 013ab426-5d6d-4d38-b34a-ae8ecc8458dc
plot_forward_fidelity_decay(model)

# ╔═╡ f0e96743-6d6e-4358-89b9-d05996986386
plot_bloch_sphere(model.forward_ensembles[:, 0])

# ╔═╡ af235e27-5c57-4183-82d8-dad18a44a8aa
backward_ensembles = test(model, training_strategy);

# ╔═╡ 1c809324-66c5-49a7-9c07-beb576347223
plot_bloch_sphere(backward_ensembles[:, 0])

# ╔═╡ 9e9aca97-b122-4c0a-bc8a-e99628035874
# ╠═╡ show_logs = false
bplh = plot_training_loss_history(model, training_strategy)

# ╔═╡ 8f3e4157-cfa1-4e54-a281-09e90d328651
if !all(isempty, training_strategy.loss_history)
	record_run(model, training_strategy, bplh, "Original")
end

# ╔═╡ 38c35101-8806-4bf8-b5ca-e468f6012f1c
# ╠═╡ show_logs = false
elh = plot_eval_loss_history(model, training_strategy, backward_ensembles)

# ╔═╡ b4a074c3-dcd7-42ad-be85-c1b949e161e1
wasserstein_distance(
	backward_ensembles[:, 0] |> OffsetArrays.no_offset_view,
	model.forward_ensembles[:, 0] |> OffsetArrays.no_offset_view
)

# ╔═╡ fe436074-a580-4c57-8869-18f372571e60
# ╠═╡ disabled = true
#=╠═╡
if training_strategy.loss_history .|> sum |> sum != 0
	CairoMakie.save("$(model.n_qubits)q_eval_hist.png", elh)
end
  ╠═╡ =#

# ╔═╡ d9bb3bd3-ba26-4cff-91dc-da40d5341fa5
# ╠═╡ disabled = true
#=╠═╡
using StatsBase
  ╠═╡ =#

# ╔═╡ 8ea0e4d7-dbbf-4d18-a8a6-4c99fad55b55
# ╠═╡ disabled = true
#=╠═╡
begin
	params = rand(Float64, (2 * model.n_total * model.n_layers))
	t = 4
	losses = []
	for _ in 1:50
		indices = StatsBase.sample(
			1:model.forward_ensemble_size,
			model.backward_ensemble_size,
			replace = false,
		)

		input_ensemble::Ensemble = model.forward_ensembles[indices, t]
		target_ensemble::Ensemble = model.forward_ensembles[indices, 0]

		denoised_ensemble = denoise(model, training_strategy, input_ensemble |> ensemble_to_batch, params)
		loss = training_strategy.loss_function(
			target_ensemble |> ensemble_to_matrix,
			denoised_ensemble
		)
		push!(losses, loss)
	end
end
  ╠═╡ =#

# ╔═╡ 7600e501-5753-43d8-bc6a-db4d6102e1d2
#=╠═╡
losses |> std
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─b37d79fa-f9a8-11f0-8be3-37c0117d6175
# ╠═9ba0daf4-2b8e-47c8-beb4-796bfa13f4f5
# ╠═c9dfae33-aef0-42e5-b43c-87d542a1b428
# ╠═91c975f1-500d-4f56-8dc0-65075bc54974
# ╠═5de59778-07cd-496b-84ec-4700db380ea6
# ╠═3cc2773d-624a-40a1-8839-8e1efde82147
# ╠═06399d96-599f-40ab-b2f7-f8f6955617ca
# ╠═0d3e791f-4d38-4c2a-8e56-180cf0eac7cf
# ╠═013ab426-5d6d-4d38-b34a-ae8ecc8458dc
# ╠═f0e96743-6d6e-4358-89b9-d05996986386
# ╠═af235e27-5c57-4183-82d8-dad18a44a8aa
# ╠═1c809324-66c5-49a7-9c07-beb576347223
# ╠═9e9aca97-b122-4c0a-bc8a-e99628035874
# ╠═8f3e4157-cfa1-4e54-a281-09e90d328651
# ╠═38c35101-8806-4bf8-b5ca-e468f6012f1c
# ╠═b4a074c3-dcd7-42ad-be85-c1b949e161e1
# ╠═fe436074-a580-4c57-8869-18f372571e60
# ╠═d9bb3bd3-ba26-4cff-91dc-da40d5341fa5
# ╠═8ea0e4d7-dbbf-4d18-a8a6-4c99fad55b55
# ╠═7600e501-5753-43d8-bc6a-db4d6102e1d2
