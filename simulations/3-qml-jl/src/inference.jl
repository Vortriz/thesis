export inference

function inference(model::Model, params::Matrix{Float64})

end

function denoise(model::Model, strategy::TrainingStrategy, input_reg::ConcreteBatchedArrayReg, params::Vector{Float64})
    input_with_ancilla = join(
        input_reg,
        zero_state(model.n_ancilla; nbatch = input_reg.nbatch),
    )

    circuit = dispatch(model.backward_circuit, params)
    apply!(input_with_ancilla, circuit)

    measure!(RemoveMeasured(), input_with_ancilla, (model.n_qubits+1):model.n_total)
    return input_with_ancilla.state
end

function denoise(model::Model, strategy::TrainingStrategy, input_reg::ConcreteArrayReg, params::Vector{Float64})
    input_with_ancilla = join(
        input_reg,
        zero_state(model.n_ancilla),
    )

    circuit = dispatch(model.backward_circuit, params)
    apply!(input_with_ancilla, circuit)

    measure!(RemoveMeasured(), input_with_ancilla, (model.n_qubits+1):model.n_total)
    return input_with_ancilla.state
end

function initialize_backward_ensemble(model::Model)
    ensemble = Vector{ArrayReg}(undef, (model.batch_size))
    for i in eachindex(ensemble)
        ensemble[i] =
            arrayreg(rand_unitary(2^model.n_qubits, Val(:haar))[1, :])
    end
    return ensemble
end

function test(model::Model, strategy::TrainingStrategy; weights::Union{Nothing, Matrix{Float64}} = nothing)
    eval_weights = isnothing(weights) ? model.params : weights
    T_eval = size(eval_weights, 2)

    backward_states = OffsetArrays.Origin(1, 0)(fill(zero_state(model.n_qubits), (model.batch_size, T_eval + 1)))
    backward_states[:, T_eval] = initialize_backward_ensemble(model)

    for t in range(T_eval, 1; step = -1)
        # The denoise function returns a matrix of states
        output_matrix = denoise(
                model,
                strategy,
                backward_states[:, t] |> OffsetArrays.no_offset_view |> ensemble_to_batch,
                eval_weights[:, t],
            )

        # Convert the matrix of states to an ensemble of ArrayRegs
        backward_states[:, t-1] = matrix_to_ensemble(output_matrix)
    end

    return backward_states
end
