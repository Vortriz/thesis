export collapse

function collapse(
	::Val{alternate},
    arch::ModelArch,
	rng::AbstractRNG,
	ensemble::CTBArrayReg,
)::Matrix{ComplexF64}

    n_data = arch.n_data
    n_ancilla = arch.n_ancilla
    batch_size = ensemble.nbatch
    n_a_dim = 1 << n_ancilla
    n_d_dim = 1 << n_data

    indices = Zygote.ignore() do
        col_offsets = (0:batch_size-1) .* n_a_dim
        res = measure(ensemble, 1:n_ancilla; rng=rng)
        vec(Int.(res)) .+ 1 .+ col_offsets
    end

    state_3d = reshape(ensemble.state, n_a_dim, n_d_dim, batch_size)
    state_permuted = permutedims(state_3d, (2, 1, 3))
    state_2d = reshape(state_permuted, n_d_dim, :)

    collapsed_state = state_2d[:, indices]
    probs = sum(abs2, collapsed_state, dims=1)

    return collapsed_state ./ sqrt.(probs .+ 1e-12)
end

function collapse(
	::Val{normal},
	arch::ModelArch,
	rng::AbstractRNG,
	ensemble::CTBArrayReg,
)::Matrix{ComplexF64}

    n_data = arch.n_data
    n_ancilla = arch.n_ancilla
	batch_size = ensemble.nbatch
	n_a_dim = 1 << n_ancilla
	n_d_dim = 1 << n_data

	indices = Zygote.ignore() do
		col_offsets = (0:batch_size-1) .* n_a_dim
		# Measure HIGHER bits (the data bits)
		res = measure(ensemble, (n_data+1):(n_data+n_ancilla); rng=rng)
		vec(Int.(res)) .+ 1 .+ col_offsets
	end

	state_2d = reshape(ensemble.state, n_d_dim, :)
	collapsed_state = state_2d[:, indices]

	probs = sum(abs2, collapsed_state, dims=1)
	return collapsed_state ./ sqrt.(probs .+ 1e-12)
end


export scramble

function scramble(
    arch::ModelArch,
    config::TrainConfig,
    rng::AbstractRNG,
    ensemble::CTBArrayReg;
    weight_schedule::Vector{Float64},
)::Vector{CTBArrayReg}

    n_qubits = arch.n_data
    circuit = scramble_circuit(n_qubits)

    trajectory = Vector{CTBArrayReg}(undef, config.T + 1)
    trajectory[1] = copy(ensemble)

    for t in 1:config.T
        current_ensemble = copy(ensemble)

        for s in 1:config.dataset_size
            reg = viewbatch(current_ensemble, s)
            # Run through all steps up to the current timestep t
            for prev_t in 1:t
                # Generate random parameters scaled by the weight schedule for this step
                params = vcat(
                    weight_schedule[prev_t] .* (rand(rng, n_qubits * 3) .* (pi/4) .- (pi/8)),
                    weight_schedule[prev_t] .* (rand(rng, binomial(n_qubits, 2)) .* 0.2 .+ 0.4) ./ (2.0 * sqrt(n_qubits))
                )

                dispatch!(circuit, params)
                apply!(reg, circuit)
            end
        end

        trajectory[t+1] = current_ensemble
    end

    return trajectory
end


export batch_and_normalize

function batch_and_normalize(ensemble::Matrix{ComplexF64})::CTBArrayReg
    reg = ensemble |> BatchedArrayReg |> transpose_storage
    normalize!(reg)

    return reg
end


export apply_pqc

function apply_pqc(
	arch::ModelArch,
	rng::AbstractRNG,
	params::Vector{Float64},
	input_ensemble::CTBArrayReg,
)::CTBMatrix

	output_ensemble = apply(
		input_ensemble,
		dispatch(arch.ansatz, params),
	)

	collapsed_ensemble_matrix = collapse(
		Val(arch.collapse_method),
		arch,
		rng,
		output_ensemble,
	)

	return collapsed_ensemble_matrix
end


export loss_and_grads

function loss_and_grads(
	arch::ModelArch,
	rng::AbstractRNG,
	params::Vector{Float64},
	input_ensemble::CTBArrayReg,
	target_matrix::CTBMatrix,
)
	return (
		Zygote.withgradient(params) do p
			collapsed_ensemble_matrix = apply_pqc(
				arch,
				rng,
				p,
				input_ensemble,
			)

			C = 1.0 .- abs2.(target_matrix' * collapsed_ensemble_matrix)

			Γ = Zygote.ignore() do
				optimal_transport_plan(C)
			end

			return dot(Γ, C)
		end
	)
end
