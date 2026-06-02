export save

macro store(path, data)
    return :(open(f -> serialize(f, $(esc(data))), $(esc(path)), "w"))
end

function save(
    save_path::String,
    ansatz::AbstractAnsatz, config::TrainConfig,
    params::Matrix{Float64}, loss_history_fig,
)
    @store joinpath(save_path, "model.jls") (ansatz=ansatz, config=config)
    @store joinpath(save_path, "params.jls") params

    CairoMakie.save(joinpath(save_path, "loss_history.svg"), loss_history_fig)
    return
end

function save(
    save_path::String,
    plots::Dict{String, CairoMakie.Figure},
)
    for (fname, fig) in plots
        CairoMakie.save(joinpath(save_path, "$fname.svg"), fig)
    end
end

# Empty stubs for TensorBoardLogger extension
function log_hyperparams end
function log_optim end
