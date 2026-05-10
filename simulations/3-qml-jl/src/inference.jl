export inference

function inference(
    arch::ModelArch,
    config::TrainConfig,
    initial_ensemble::CBArrayReg,
    params::Matrix{Float64}
)::CBArrayReg

    current_ensemble = copy(initial_ensemble)

    for t in 1:config.T
        append_qubits!(current_ensemble, arch.n_ancilla)

        current_ensemble = apply_pqc(
            arch,
            current_ensemble,
            params[:, t],
        ) |> batch_and_normalize
    end

    return current_ensemble
end
