export inference

function inference(model::Model, anzatz::ChainBlock, initial_ensemble::BatchedArrayReg, params::Matrix{Float64})
    for t in 1:model.T
        ensemble_with_ancilla = join(initial_ensemble, zero_state(model.n_ancilla; nbatch=model.batch_size))
        dispatch!(ansatz, params[:, t])
        apply!(ensemble_with_ancilla, ansatz)
        ensemble = stochastic_collapse(ensemble_with_ancilla, model.n_ancilla, model.n_data)
    end
end
