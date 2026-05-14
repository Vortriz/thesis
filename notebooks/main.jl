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
    using Random
    using LinearAlgebra
    import Dates

    using Yao
    using CairoMakie
    using StatsBase
    import Zygote
    import Optimisers

    using ProgressLogging
    using TensorBoardLogger, Logging
    # using BenchmarkTools
    # using JET
    # using ProfilePerfetto
end

# ╔═╡ 7479301c-85c0-4773-bc5d-1195c7cb47ad
begin
    const T = 2
    const rng = MersenneTwister(1234)
    const TB_LOGGING = true

    arch = ModelArch(;
        n_data=1,
        n_ancilla=1,
        n_layers=1,
        ansatz_builder=HEA,
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

# ╔═╡ d5d5bd83-1e77-4d5f-a24c-a0e576fe1eff
begin
    const save_path = joinpath(
        dirname(Base.current_project()),
        "data",
        Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS"),
    )
    tbl = TB_LOGGING ? TBLogger(
        save_path;
        min_level=Logging.Info,
    ) : nothing
end

# ╔═╡ 0c83c042-7de8-4b61-a041-59d39f9d61bd
target_bloch = plot_bloch_sphere(target_ensemble)

# ╔═╡ 1af82bcf-1787-403e-a4c8-8bc59f1bd995
function train(
    arch::ModelArch,
    config::TrainConfig;
    callback=(loss, step) -> nothing,
)
    params = rand(rng, Float64, (arch.n_params_ppb, config.T))
    loss_history = [zeros(Float64, n) for n in config.epoch_schedule]

    model_state = ModelState()
    model_state.current_ensemble = config.initial_ensemble |> copy

    global_step = 1

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

            callback(loss, global_step)
            global_step += 1
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
loss_history, params = train(
    arch,
    config;
    callback=(loss, step) -> begin
        if !isnothing(tbl)
            log_value(tbl, "loss", loss; step=step)
        end
    end,
)

# ╔═╡ e7010f58-8607-4bd6-97d2-b0a3a668bbb5
loss_history_fig = plot_loss_history(loss_history; yscale=log10)

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
generated_bloch = plot_bloch_sphere(generated_trajectory[end])

# ╔═╡ d95bf7db-e4d9-4964-9019-69ac019dd7fb
if TB_LOGGING == true
    log_and_save(
        tbl, save_path,
        arch, config, rng,
        loss_history, loss_history_fig, params,
        target_bloch, generated_bloch,
    )
end

# ╔═╡ Cell order:
# ╟─168a33fa-4be8-11f1-937a-99ef8733e91e
# ╠═7c3a8a85-b2b5-4dc7-9638-7b7d2d6f3a3e
# ╠═f0dd9925-d3d2-4cad-9b9f-bf11ac792953
# ╠═7479301c-85c0-4773-bc5d-1195c7cb47ad
# ╠═d5d5bd83-1e77-4d5f-a24c-a0e576fe1eff
# ╠═0c83c042-7de8-4b61-a041-59d39f9d61bd
# ╠═1af82bcf-1787-403e-a4c8-8bc59f1bd995
# ╠═de779e94-a981-4020-a5d8-ef25ba701015
# ╠═e7010f58-8607-4bd6-97d2-b0a3a668bbb5
# ╠═c0c82792-abd4-48b4-8fe2-d913c60d1e92
# ╠═dcdba0ce-8b94-4d38-8e9e-980070e155e0
# ╠═d95bf7db-e4d9-4964-9019-69ac019dd7fb
