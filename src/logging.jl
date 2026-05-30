export log_hyperparams, log_optim, save_run

macro store(path, data)
    return :(open(f -> serialize(f, $(esc(data))), $(esc(path)), "w"))
end

function log_hyperparams(
    tbl::TBLogger,
    ansatz::AbstractAnsatz,
    config::TrainConfig,
)
    for field in [:n_data, :n_ancilla, :n_qubits, :n_layers]
        log_text(
            tbl,
            field |> string,
            getfield(ansatz, field);
            step=0,
        )
    end

    log_text(
        tbl,
        "measurement",
        ansatz.measurement |> typeof |> string;
        step=0,
    )

    for field in [:dataset_size, :batch_size, :T, :epoch_schedule]
        log_text(
            tbl,
            field |> string,
            getfield(config, field);
            step=0,
        )
    end

    log_text(
        tbl,
        "trajectory",
        config.trajectory |> typeof |> string;
        step=0,
    )
end

function log_optim(
    tbl::TBLogger,
    optimizer::Optimisers.AbstractRule,
)
    log_text(
        tbl,
        "optimizer",
        string(optimizer);
        step=0,
    )
end

function save_run(
    save_path::String,
    ansatz::AbstractAnsatz, config::TrainConfig,
    params::Matrix{Float64}, loss_history_fig,
)
    @store joinpath(save_path, "model.jls") (ansatz=ansatz, config=config)
    @store joinpath(save_path, "params.jls") params

    save(joinpath(save_path, "loss_history.svg"), loss_history_fig)
    return
end
