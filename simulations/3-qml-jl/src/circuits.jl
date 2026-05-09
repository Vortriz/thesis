export scramble_circuit

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


# Hardware Efficient Ansatz
export HEA

function HEA(n_qubits::Int64, n_layers::Int64)::ChainBlock
	register = 1:n_qubits
	entangle_pairs = if n_qubits == 2
		[(1,2)]
	else
		[(i, mod1(i+1, n_qubits)) for i in register]
	end

	layer = chain(
		n_qubits,
		chain(put(i=>chain(Rx(0), Ry(0))) for i in register),
		chain(cz(i, j) for (i, j) in entangle_pairs),
	)

	return layer^n_layers
end


# Entanglement-variational Hardware-efficient Ansatz
export EHA

@const_gate Rxp::ComplexF64 = Rx(pi/2) |> mat
@const_gate Rxn::ComplexF64 = Rx(-pi/2) |> mat

XX(i::Int64) = chain(cnot(i, i+1), put(i=>Rx(0)), cnot(i, i+1))
ZZ(i::Int64) = chain(cnot(i, i+1), put(i+1=>Rz(0)), cnot(i, i+1))
YY(i::Int64) = chain(
    repeat(Rxp, (i, i+1)),
    cnot(i, i+1), put(i+1=>Rz(0)), cnot(i, i+1),
    repeat(Rxn, (i, i+1)),
)

Uₑₙₜ(i::Int64) = chain(XX(i), YY(i), ZZ(i))

function EHA(n_qubits::Int64, n_layers::Int64)::ChainBlock{2}
    register = 1:n_qubits

    layer = chain(
    	n_qubits,
    	chain(put(i=>chain(Rz(0), Ry(0), Rz(0))) for i in register),
    	chain(Uₑₙₜ(i) for i in 1:n_qubits-1),
    )

    return layer^n_layers
end
