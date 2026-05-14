export inference

function inference(
    arch::ModelArch,
    config::TrainConfig,
    initial_ensemble::CBArrayReg,
    params::Matrix{Float64},
)
    trajectory = Vector{CBArrayReg}(undef, config.T + 1)
    trajectory[1] = copy(initial_ensemble)
    current_ensemble = copy(initial_ensemble)

    for t in 1:config.T
        append_qubits!(current_ensemble, arch.n_ancilla)

        current_ensemble =
            apply_pqc(
                arch,
                current_ensemble,
                params[:, t],
            ) |> batch_and_normalize

        trajectory[t+1] = copy(current_ensemble)
    end

    return trajectory
end
