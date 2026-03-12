export Model

mutable struct Model
    # --- System Configuration ---
    const n_qubits::Int64
    const n_ancilla::Int64
    const n_total::Int64

    # --- Diffusion Configuration ---
    const T::Int64

    # --- Forward Process Configuration ---
    const forward_ensemble_size::Int64

    # --- Backward Process Configuration ---
    const n_layers::Int64
    const backward_ensemble_size::Int64

    const rng::MersenneTwister

    # Misc ---
    forward_ensembles::OffsetEnsembleCollection

    # Diagnostics
    forward_fidelity_decay::OffsetVector{Float64}

    # Trained Model Parameters
    trained_params::Matrix{Float64}

    # Circuits
    forward_circuit::ChainBlock
    backward_circuit::ChainBlock

    RZZ(n::Int64, i::Int64, j::Int64)::ChainBlock =
        chain(n, control(i, j=>X), put(j=>Rz(0)), control(i, j=>X))

    function gen_forward_circuit(n_qubits::Int64)::ChainBlock
        circuit = chain(n_qubits)

        push!(
            circuit,
            chain(n_qubits, put(i=>chain(Rx(0), Ry(0), Rz(0))) for i = 1:n_qubits),
        )

        RZZ_combinations = combinations(1:n_qubits, 2)
        push!(
			circuit,
			chain(RZZ(n_qubits, i, j) for (i, j) in collect(RZZ_combinations))
		)

        return circuit
    end

    function gen_backward_circuit(n_total::Int64, n_layers::Int64)::ChainBlock
        circuit = chain(n_total)

        layer = chain(
            n_total,
            chain(n_total, put(i=>chain(Rx(0), Ry(0))) for i = 1:n_total),
            chain(n_total, chain(cz(i, i+1) for i in range(1, n_total-1; step = 2))),
            chain(n_total, chain(cz(i, i+1) for i in range(2, n_total-1; step = 2))),
        )

        push!(circuit, layer^n_layers)

        return circuit
    end

    # --- Constructor ---
    function Model(;
        n_qubits,
        n_ancilla,
        T,
        forward_ensemble_size,
        n_layers,
        backward_ensemble_size,
        rng,
    )
        n_total = n_qubits + n_ancilla
		n_params = 2 * n_total * n_layers

        new(
            n_qubits,
            n_ancilla,
            n_total,
            T,
            forward_ensemble_size,
            n_layers,
            backward_ensemble_size,
            rng,
			OffsetArrays.Origin(1, 0)(fill(zero_state(n_qubits), (forward_ensemble_size, T + 1))),
            OffsetArrays.Origin(0)(zeros(Float64, (T + 1))),
            zeros(Float64, (n_params, T)),
            gen_forward_circuit(n_qubits),
            gen_backward_circuit(n_total, n_layers),
        )
    end
end
