export AbstractDist
abstract type AbstractDist end

macro dist_struct(name)
    return quote
        struct $name <: AbstractDist
            n_qubits::Int64
            n_samples::Int64
            register::Register

            function $name(register::Register)
                n_qubits = size(register.state, 1) |> log2 |> Int
                n_samples = register.nbatch
                return new(n_qubits, n_samples, register)
            end

            $name(ensemble::BatchState) = $name(convert(Register, ensemble))
        end
    end |> esc
end


export ArbitraryDist
@dist_struct ArbitraryDist


export ClusteredDist
@dist_struct ClusteredDist

function ClusteredDist(;
    n_qubits::Int64,
    n_samples::Int64,
    spread::Float64=0.05,
)
    ensemble = (
        randn(RNG, ComplexF64, (2^n_qubits, 1)) .+
        spread * randn(RNG, ComplexF64, (2^n_qubits, n_samples))
    )

    return ClusteredDist(ensemble)
end


export CircleDist
@dist_struct CircleDist

function CircleDist(;
    n_qubits::Int64,
    n_samples::Int64,
)
    @assert n_qubits == 1 "Circle distribution is defined only for 1 qubit."

    phis = rand(Float64, n_samples) * 2π
    ensemble_gen = ([cos(phis[i]), sin(phis[i])] .|> ComplexF64 for i in 1:n_samples)
    ensemble = reduce(hcat, ensemble_gen)

    return CircleDist(ensemble)
end


export QKRLocalizedDist, gen_qkr_operator
@dist_struct QKRLocalizedDist

function gen_qkr_operator(;
    n_qubits::Int64,
    K::Float64,
    ħₛ::Float64,
)::AbstractQuantumObject{Operator}
    dims = 2^n_qubits
    m_vec = [0:(dims/2-1); (-dims/2):-1]
    U = zeros(ComplexF64, (dims, dims))

    Threads.@threads for idx in CartesianIndices(U)
        i, j = idx.I
        m₁, m₂ = m_vec[i], m_vec[j]
        d = m₂ - m₁
        if d > dims / 2
            d -= dims
        end
        if d < -dims / 2
            d += dims
        end
        U[idx] = ℯ^(-im / 2 * ħₛ * m₂^2) * im^d * besselj(d, K / ħₛ)
    end

    return QT.Qobj(U)
end

function QKRLocalizedDist(;
    n_qubits::Int64,
    K::Union{Float64, Vector{Float64}},
    ħₛ::Union{Float64, Vector{Float64}},
)
    if typeof(K) == Float64
        K = fill(K, length(ħₛ))
    end

    if typeof(ħₛ) == Float64
        ħₛ = fill(ħₛ, length(K))
    end

    ensemble_gen = (
        gen_qkr_operator(;
            n_qubits=n_qubits,
            K=K[i],
            ħₛ=ħₛ[i],
        ) |> QT.eigenstates |> evd -> evd.vectors
        for i in eachindex(K)
    )
    ensemble = reduce(hcat, ensemble_gen)

    return QKRLocalizedDist(ensemble)
end


export TFIMDist, gen_tfim_hamiltonian
@dist_struct TFIMDist

function gen_tfim_hamiltonian(;
    n_qubits::Int64,
    g::Float64,
)::AbstractQuantumObject{Operator}
    H = QT.Qobj(
        zeros(ComplexF64, (2^n_qubits, 2^n_qubits));
        dims=Tuple(fill(2, n_qubits)),
    )

    partial_term_1 = vcat(
        [QT.sigmaz(), QT.sigmaz()],
        fill(QT.eye(2), n_qubits - 2),
    )
    partial_term_2 = vcat(
        [QT.sigmax()],
        fill(QT.eye(2), n_qubits - 1),
    )

    for i in 0:(n_qubits-2)
        H -= reduce(QT.kron, circshift(partial_term_1, i))
    end
    for i in 0:(n_qubits-1)
        H -= g * reduce(QT.kron, circshift(partial_term_2, i))
    end

    return H
end

function TFIMDist(;
    n_qubits::Int64,
    g::Vector{Float64},
)
    @assert n_qubits >= 2 "TFIM distribution is defined only for more than 2 qubits."

    ensemble = zeros(ComplexF64, (2^n_qubits, length(g)))

    for (i, gᵢ) in enumerate(g)
        H = gen_tfim_hamiltonian(; n_qubits=n_qubits, g=gᵢ)
        ensemble[:, i] = QT.eigenstates(H).vectors[:, 1]
    end

    return TFIMDist(ensemble)
end


export HaarDist
@dist_struct HaarDist

function HaarDist(;
    n_qubits::Int64,
    n_samples::Int64,
)
    ensemble = randn(RNG, ComplexF64, (2^n_qubits, n_samples))

    return HaarDist(ensemble)
end
