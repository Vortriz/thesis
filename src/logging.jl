export log_hyperparams, save_run

function log_hyperparams(
    logger::TBLogger,
    arch::ModelArch,
    config::TrainConfig,
    rng::AbstractRNG,
)
    for field in [:n_data, :n_ancilla, :n_qubits, :n_layers, :ansatz_name, :n_params_ppb]
        log_text(
            logger,
            field |> string,
            getfield(arch, field);
            step=0,
        )
    end

    log_text(
        logger,
        "collapse_method",
        typeof(arch.collapse_method).parameters[1] |> string;
        step=0,
    )

    for field in
        [:dataset_size, :batch_size, :T, :target_schedule, :epoch_schedule, :optimizer]
        log_text(
            logger,
            field |> string,
            getfield(config, field);
            step=0,
        )
    end

    log_text(
        logger,
        "target_trajectory_type",
        typeof(config.target_trajectory_type).parameters[1] |> string;
        step=0,
    )

    log_text(logger, "rng", rng; step=0)

    return
end

function save_run(
    tbl::TBLogger, save_path::String,
    arch::ModelArch, config::TrainConfig, rng::AbstractRNG,
    loss_history::Vector{Vector{Float64}}, loss_history_fig, params::Matrix{Float64},
    target_bloch=nothing, generated_bloch=nothing,
)
    @save joinpath(save_path, "model.jld2") arch config rng
    @save joinpath(save_path, "results.jld2") loss_history params

    save(joinpath(save_path, "loss_history.svg"), loss_history_fig)

    if target_bloch !== nothing && generated_bloch !== nothing
        save(joinpath(save_path, "target_bloch.svg"), target_bloch)
        save(joinpath(save_path, "generated_bloch.svg"), generated_bloch)
    end

    return
end
