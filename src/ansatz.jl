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
@const_gate Rxp::ComplexF64 = Rx(π / 2) |> mat
@const_gate Rxn::ComplexF64 = Rx(-π / 2) |> mat

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


export IdentityParams, RandParams

struct IdentityParams <: AbstractParams
    params::Matrix{Float64}

    function IdentityParams(ansatz::A, T::Int64) where {A <: AbstractAnsatz}
        T <= 0 && throw(DomainError(T, "T should be a positive integer"))

        params = Vector{Matrix{Float64}}()
        n_params_per_layer = ansatz.n_params ÷ sum(ansatz.n_layers)
        for subblock_size in ansatz.n_layers
            half_subblock_params =
                rand(RNG, Float64, (n_params_per_layer * subblock_size ÷ 2, T))
            push!(params, half_subblock_params)
            push!(params, reverse(-half_subblock_params))
        end

        return new(reduce(vcat, params))
    end
end

struct RandParams <: AbstractParams
    params::Matrix{Float64}

    function RandParams(ansatz::A, T::Int64) where {A <: AbstractAnsatz}
        T <= 0 && throw(DomainError(T, "T should be a positive integer"))

        return new(rand(RNG, Float64, (ansatz.n_params, T)))
    end
end
