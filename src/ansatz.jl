export ansatz

# Hardware Efficient Ansatz
function create_layer(::Type{HEA{M}}, n_qubits::Int64) where {M <: AbstractMeasurement}
    register = 1:n_qubits
    entangle_pairs = if n_qubits == 2
        [(1, 2)]
    else
        [(i, mod1(i + 1, n_qubits)) for i in register]
    end

    layer = chain(n_qubits)
    for i in register
        push!(layer, put(i => Rx(0)))
        push!(layer, put(i => Ry(0)))
    end

    for (i, j) in entangle_pairs
        push!(layer, cz(i, j))
    end

    return layer
end

# Entanglement-variational Hardware-efficient Ansatz
@const_gate Rxp::ComplexF64 = Rx(π/2) |> mat
@const_gate Rxn::ComplexF64 = Rx(-π/2) |> mat

XX(i::Int64) = chain(cnot(i, i + 1), put(i => Rx(0)), cnot(i, i + 1))
ZZ(i::Int64) = chain(cnot(i, i + 1), put(i + 1 => Rz(0)), cnot(i, i + 1))
YY(i::Int64) = chain(
    repeat(Rxp, (i, i + 1)),
    cnot(i, i + 1), put(i + 1 => Rz(0)), cnot(i, i + 1),
    repeat(Rxn, (i, i + 1)),
)

Uₑₙₜ(i::Int64) = chain(XX(i), YY(i), ZZ(i))

function create_layer(::Type{EHA{M}}, n_qubits::Int64) where {M <: AbstractMeasurement}
    register = 1:n_qubits

    layer = chain(n_qubits)
    for i in register
        push!(layer, put(i => Rz(0)))
        push!(layer, put(i => Ry(0)))
        push!(layer, put(i => Rz(0)))
    end
    for i in 1:(n_qubits-1)
        push!(layer, Uₑₙₜ(i))
    end

    return layer
end

function ansatz(
    A::Type{<:AbstractAnsatz},
    n_qubits::Int64,
    n_layers::Int64,
)::ChainBlock{2}
    circuit = create_layer(A, n_qubits)^n_layers
    return Optimise.canonicalize(circuit)
end

function ansatz(
    A::Type{<:AbstractAnsatz},
    n_qubits::Int64,
    n_layers::Vector{Int64},
)::ChainBlock{2}
    subblocks = chain(n_qubits)
    for subblock_size in n_layers
        layer = create_layer(A, n_qubits)
        push!(subblocks, layer^subblock_size)
        push!(subblocks, layer'^subblock_size)
    end

    return Optimise.canonicalize(subblocks)
end


export identity_params

function identity_params(ansatz::A) where {A <: AbstractAnsatz}
    params = Vector{Float64}()
    n_params_per_layer = create_layer(typeof(ansatz), ansatz.n_qubits) |> nparameters
    for subblock_size in ansatz.n_layers
        subblock_params = rand(RNG, Float64, n_params_per_layer * subblock_size)
        append!(params, subblock_params)
        append!(params, reverse(-subblock_params))
    end

    return params
end
