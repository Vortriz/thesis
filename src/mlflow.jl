export get_hyperparams

function get_hyperparams(
    arch::ModelArch,
    config::TrainConfig,
    rng::AbstractRNG,
)
    hyperparams::Vector{Tuple{String, String}} = []

    for field in [:n_data, :n_ancilla, :n_qubits, :n_layers, :ansatz_name, :n_params_ppb, :collapse_method]
        push!(
            hyperparams,
            (field |> string, getfield(arch, field) |> string)
        )
    end

    for field in [:dataset_size, :batch_size, :T, :target_trajectory_type, :target_schedule, :epoch_schedule, :optimizer]
        push!(
            hyperparams,
            (field |> string, getfield(config, field) |> string)
        )
    end

    push!(hyperparams, ("rng", rng |> string))

    return hyperparams
end
