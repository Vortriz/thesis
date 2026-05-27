export log_hyperparams, save_run

macro store(path, data)
    return :( open(f -> serialize(f, $(esc(data))), $(esc(path)), "w") )
end

function log_hyperparams(
    tbl::TBLogger,
    arch::ModelArch,
    config::TrainConfig,
    rng::AbstractRNG,
)
    for field in [:n_data, :n_ancilla, :n_qubits, :n_layers, :ansatz_name, :n_params_ppb]
        log_text(
            tbl,
            field |> string,
            getfield(arch, field);
            step=0,
        )
    end

    log_text(
        tbl,
        "collapse_method",
        arch.collapse_method |> typeof |> string;
        step=0,
    )

    for field in [:dataset_size, :batch_size, :T, :target_schedule, :epoch_schedule, :optimizer]
        log_text(
            tbl,
            field |> string,
            getfield(config, field);
            step=0,
        )
    end

    for field in [:initial_ensemble_type, :target_ensemble_type]
        log_text(
            tbl,
            field |> string,
            getfield(config, field) |> string;
            step=0,
        )
    end

    log_text(
        tbl,
        "target_trajectory_type",
        config.target_trajectory_type |> typeof |> string;
        step=0,
    )

    log_text(tbl, "rng", rng; step=0)

    return
end

function save_run(
    save_path::String,
    arch::ModelArch, config::TrainConfig, rng::AbstractRNG,
    loss_history::Vector{Vector{Float64}}, loss_history_fig, params::Matrix{Float64},
)
    @store joinpath(save_path, "model.jls") (arch=arch, config=config, rng=rng)
    @store joinpath(save_path, "results.jls") (loss_history=loss_history, params=params)

    save(joinpath(save_path, "loss_history.svg"), loss_history_fig)

    if target_bloch !== nothing && generated_bloch !== nothing
        save(joinpath(save_path, "target_bloch.svg"), target_bloch)
        save(joinpath(save_path, "generated_bloch.svg"), generated_bloch)
    end

    return
end
