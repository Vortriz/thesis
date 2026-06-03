export save

macro store(path, data)
    return :(open(f -> serialize(f, $(esc(data))), $(esc(path)), "w"))
end

function save(
    save_path::String,
    ansatz::AbstractAnsatz,
    config::TrainConfig,
    params::Matrix{Float64},
)
    @store joinpath(save_path, "model.jls") (ansatz=ansatz, config=config)
    @store joinpath(save_path, "params.jls") params
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
