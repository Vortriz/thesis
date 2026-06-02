module GQMLTensorBoardLoggerExt

using GQML: GQML, AbstractAnsatz, TrainConfig
import Optimisers
using TensorBoardLogger: TBLogger, log_text

function GQML.log_hyperparams(
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
    log_text(
        tbl,
        "initial_dist",
        config.trajectory.steps[begin] |> typeof |> string;
        step=0,
    )
    log_text(
        tbl,
        "target_dist",
        config.trajectory.steps[end] |> typeof |> string;
        step=0,
    )

    return
end

function GQML.log_optim(
    tbl::TBLogger,
    optimizer::Optimisers.AbstractRule,
)
    log_text(
        tbl,
        "optimizer",
        string(optimizer);
        step=0,
    )

    return
end

end
