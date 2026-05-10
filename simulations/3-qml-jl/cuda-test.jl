### A Pluto.jl notebook ###
# v0.20.25

using Markdown
using InteractiveUtils

# ╔═╡ 168a33fa-4be8-11f1-937a-99ef8733e91e
begin
    import Pkg

    # activate the shared project environment
    Pkg.activate(Base.current_project())
    Pkg.instantiate()
end

# ╔═╡ 7c3a8a85-b2b5-4dc7-9638-7b7d2d6f3a3e
# ╠═╡ show_logs = false
begin
	include("src/base.jl")
	using .QML
end

# ╔═╡ f0dd9925-d3d2-4cad-9b9f-bf11ac792953
begin
	using Yao
	using Random
	using LinearAlgebra
	using CairoMakie
	using StatsBase
	import Zygote
	import Optimisers

	using ProgressLogging
	# using BenchmarkTools
	# using JET
	# using ProfilePerfetto
end

# ╔═╡ 7479301c-85c0-4773-bc5d-1195c7cb47ad
begin
    const T = 3
	const rng = MersenneTwister(1234)
    arch = ModelArch(
        n_data=4,
        n_ancilla=4,
        n_layers=3,
        ansatz_builder=EHA,
        collapse_method=normal,
    )

    config = TrainConfig(
        dataset_size=1000,
        batch_size=100,
        target_schedule=:direct,
        epoch_schedule=fill(300, T),
        optimizer=Optimisers.AMSGrad(0.05),
    )

    target_ensemble = gen_dist(
        Val(clustered),
		rng;
        n_qubits=arch.n_data,
        n_samples=config.dataset_size,
    )
    initial_ensemble = gen_dist(
        Val(haar),
		rng;
        n_qubits=arch.n_data,
        n_samples=config.batch_size,
    )
end;

# ╔═╡ 0c83c042-7de8-4b61-a041-59d39f9d61bd
plot_bloch_sphere(target_ensemble)

# ╔═╡ 1af82bcf-1787-403e-a4c8-8bc59f1bd995
function train(
    arch::ModelArch,
    config::TrainConfig,
    target_trajectory::Vector{CBArrayReg},
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
            target_batch = target_matrix[:, target_indices]

            loss, grads = Zygote.withgradient(current_params) do p
	            collapsed_ensemble_matrix = apply_pqc(
	                arch,
	                current_ensemble,
	                p,
	            )
	
	            C = 1.0 .- abs2.(target_batch' * collapsed_ensemble_matrix)
	
	            Γ = Zygote.ignore() do
	                optimal_transport_plan(C)
	            end
	
	            return dot(Γ, C)
	        end

            Optimisers.update!(opt_state, current_params, grads[1])
            loss_history[t][epoch] = loss
        end

        current_ensemble = apply_pqc(
                               arch,
                               current_ensemble,
                               current_params,
                           ) |> batch_and_normalize

        params[:, t] = current_params
    end

    return loss_history, params
end

# ╔═╡ de779e94-a981-4020-a5d8-ef25ba701015
begin
    target_trajectory::Vector{CBArrayReg} = [target_ensemble, initial_ensemble]
    loss_history, params = train(arch, config, target_trajectory)
end

# ╔═╡ eca56e03-c725-4792-9032-7d4738706f0e
target_trajectory[1] |> typeof

# ╔═╡ e7010f58-8607-4bd6-97d2-b0a3a668bbb5
plot_loss_history(loss_history)

# ╔═╡ Cell order:
# ╟─168a33fa-4be8-11f1-937a-99ef8733e91e
# ╠═7c3a8a85-b2b5-4dc7-9638-7b7d2d6f3a3e
# ╠═f0dd9925-d3d2-4cad-9b9f-bf11ac792953
# ╠═7479301c-85c0-4773-bc5d-1195c7cb47ad
# ╠═0c83c042-7de8-4b61-a041-59d39f9d61bd
# ╠═1af82bcf-1787-403e-a4c8-8bc59f1bd995
# ╠═de779e94-a981-4020-a5d8-ef25ba701015
# ╠═eca56e03-c725-4792-9032-7d4738706f0e
# ╠═e7010f58-8607-4bd6-97d2-b0a3a668bbb5
