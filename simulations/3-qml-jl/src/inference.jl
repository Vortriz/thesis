export inference

function inference(
	arch::ModelArch,
	config::TrainConfig,
	rng::AbstractRNG,
	initial_ensemble::CTBArrayReg,
	params::Matrix{Float64}
)::CTBArrayReg

    current_ensemble = copy(initial_ensemble)

    for t in 1:config.T
        append_qubits!(current_ensemble, arch.n_ancilla)

        current_ensemble = apply_pqc(
			arch,
			rng,
			params[:, t],
			current_ensemble,
		) |> BatchedArrayReg |> transpose_storage
    end

    return current_ensemble
end
