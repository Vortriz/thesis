export test, denoise

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
    ensemble = Vector{ArrayReg}(undef, (model.backward_ensemble_size))
    for i in eachindex(ensemble)
        ensemble[i] =
            arrayreg(rand_unitary(2^model.n_qubits, Val(:haar))[1, :])
    end
    return ensemble
end

function test(model::Model, strategy::TrainingStrategy)
    backward_states = OffsetArrays.Origin(1, 0)(fill(zero_state(model.n_qubits), (model.backward_ensemble_size, model.T + 1)))
    backward_states[:, model.T] = initialize_backward_ensemble(model)

    for t in range(model.T, 1; step = -1)
        # The denoise function returns a matrix of states
        output_matrix = denoise(
                model,
                strategy,
                backward_states[:, t] |> OffsetArrays.no_offset_view |> ensemble_to_batch,
                model.trained_params[:, t],
            )

        # Convert the matrix of states to an ensemble of ArrayRegs
        backward_states[:, t-1] = [output_matrix[:, i] for i in 1:model.backward_ensemble_size] .|> ArrayReg
    end

    return backward_states
end
