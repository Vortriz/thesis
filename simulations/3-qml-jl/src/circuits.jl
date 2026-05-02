export scramble_circuit, hardware_efficient_ansatz

RZZ(n::Int64, i::Int64, j::Int64)::ChainBlock =
    chain(n, control(i, j=>X), put(j=>Rz(0)), control(i, j=>X))

function scramble_circuit(n_qubits::Int64)::ChainBlock
    register = 1:n_qubits
    circuit = chain(n_qubits)

    push!(
        circuit,
        chain(n_qubits, put(i=>chain(Rx(0), Ry(0), Rx(0))) for i in register),
    )

    RZZ_combinations = combinations(register, 2)
    push!(
			circuit,
			chain(RZZ(n_qubits, i, j) for (i, j) in collect(RZZ_combinations))
		)

    return circuit
end

function hardware_efficient_ansatz(n_data::Int64, n_ancilla::Int64, n_layers::Int64)::ChainBlock
	n_qubits = n_data + n_ancilla
	register = 1:n_qubits
	entangle_pairs = if n_qubits == 2
		[(1,2)]
	else
		[(i, mod1(i+1, n_qubits)) for i in register]
	end

	circuit = chain(n_qubits)
	layer = chain(
		n_qubits,
		chain(
			n_qubits,
			put(i=>chain(Rx(0), Ry(0))) for i in register
		),
		chain(
			n_qubits,
			chain(cz(i, j) for (i, j) in entangle_pairs)
		),
	)

	push!(circuit, layer^n_layers)

	# Measuring and removing ancilla qubits (note that locs considers reversed indexing)
	# push!(circuit, Measure(n_qubits; locs=1:n_ancilla, remove=true))

	return circuit
end
