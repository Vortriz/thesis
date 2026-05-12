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
    include("../src/QML.jl")
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
    const T = 2
    const rng = MersenneTwister(1234)
    arch = ModelArch(;
        n_data=2,
        n_ancilla=1,
        n_layers=2,
        ansatz_builder=EHA,
        collapse_method=normal,
    )

    initial_ensemble = gen_dist(
        Val(haar),
        rng;
        n_qubits=arch.n_data,
        n_samples=100,
    )
    target_ensemble = gen_dist(
        Val(clustered),
        rng;
        n_qubits=arch.n_data,
        n_samples=1000,
    )

    config = TrainConfig(
        Val(direct);
        initial_ensemble=initial_ensemble,
        target_ensemble=target_ensemble,
        epoch_schedule=fill(300, T),
        optimizer=Optimisers.AMSGrad(0.01),
    )

end;

# ╔═╡ 0c83c042-7de8-4b61-a041-59d39f9d61bd
plot_bloch_sphere(target_ensemble)

# ╔═╡ 1af82bcf-1787-403e-a4c8-8bc59f1bd995
function train(
    arch::ModelArch,
    config::TrainConfig,
)
    params = rand(rng, Float64, (arch.n_params_ppb, config.T))
    loss_history = [zeros(Float64, n) for n in config.epoch_schedule]

    model_state = ModelState()
    model_state.current_ensemble = config.initial_ensemble |> copy

    @progress for t in 1:config.T
        append_qubits!(model_state.current_ensemble, arch.n_ancilla)

        model_state.current_params = params[:, t]
        opt_state = Optimisers.setup(config.optimizer, model_state.current_params)

        target_idx = config.target_schedule[t]
        target_matrix = config.target_trajectory[target_idx].state

        @progress for epoch in 1:config.epoch_schedule[t]
            target_indices = sample(
                1:config.dataset_size,
                config.batch_size,
                replace=false,
            )
            model_state.target_matrix = @view target_matrix[:, target_indices]

            loss, grads = loss_and_grads(arch, model_state)

            Optimisers.update!(opt_state, model_state.current_params, grads[1])
            loss_history[t][epoch] = loss
        end

        model_state.current_ensemble =
            apply_pqc(
                arch,
                model_state.current_ensemble,
                model_state.current_params,
            ) |> batch_and_normalize

        params[:, t] = model_state.current_params
    end

    return loss_history, params
end

# ╔═╡ de779e94-a981-4020-a5d8-ef25ba701015
loss_history, params = train(arch, config)

# ╔═╡ 4f4c1a18-5b37-4f4a-9d78-a07f3cc51c68
arch.collapse_method

# ╔═╡ e7010f58-8607-4bd6-97d2-b0a3a668bbb5
plot_loss_history(loss_history; yscale=log10)

# ╔═╡ c0c82792-abd4-48b4-8fe2-d913c60d1e92
generated_trajectory = inference(
    arch,
    config,
    gen_dist(
        Val(haar),
        rng;
        n_qubits=arch.n_data,
        n_samples=config.batch_size,
    ),
    params,
);

# ╔═╡ dcdba0ce-8b94-4d38-8e9e-980070e155e0
plot_bloch_sphere(generated_trajectory[end])

# ╔═╡ Cell order:
# ╟─168a33fa-4be8-11f1-937a-99ef8733e91e
# ╠═7c3a8a85-b2b5-4dc7-9638-7b7d2d6f3a3e
# ╠═f0dd9925-d3d2-4cad-9b9f-bf11ac792953
# ╠═7479301c-85c0-4773-bc5d-1195c7cb47ad
# ╠═0c83c042-7de8-4b61-a041-59d39f9d61bd
# ╟─1af82bcf-1787-403e-a4c8-8bc59f1bd995
# ╠═de779e94-a981-4020-a5d8-ef25ba701015
# ╠═4f4c1a18-5b37-4f4a-9d78-a07f3cc51c68
# ╠═e7010f58-8607-4bd6-97d2-b0a3a668bbb5
# ╠═c0c82792-abd4-48b4-8fe2-d913c60d1e92
# ╠═dcdba0ce-8b94-4d38-8e9e-980070e155e0
