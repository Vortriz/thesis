export ClusteredDist

struct ClusteredDist <: AbstractDist
    register::Register
end

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


export QKRLocalizedDist

struct QKRLocalizedDist <: AbstractDist
    register::Register
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
        ) |> eigenstates |> evd -> evd.vectors
        for i in eachindex(K)
    )
    ensemble = reduce(hcat, ensemble_gen)

    return QKRLocalizedDist(ensemble)
end


export CircleDist

struct CircleDist <: AbstractDist
    register::Register
end

function CircleDist(;
    n_samples::Int64,
)
    phis = rand(Float64, n_samples) * 2pi
    ensemble_gen = ([cos(phis[i]), sin(phis[i])] .|> ComplexF64 for i in 1:n_samples)
    ensemble = reduce(hcat, ensemble_gen)

    return CircleDist(ensemble)
end


export TFIMDist

struct TFIMDist <: AbstractDist
    register::Register
end

function TFIMDist(;
    n_qubits::Int64,
    g::Vector{Float64},
)
    ensemble = zeros(ComplexF64, (2^n_qubits, length(g)))

    for (i, gᵢ) in enumerate(g)
        H = gen_tfim_hamiltonian(; n_qubits=n_qubits, g=gᵢ)
        ensemble[:, i] = QT.eigenstates(H).vectors[:, 1]
    end

    return TFIMDist(ensemble)
end


export HaarDist

struct HaarDist <: AbstractDist
    register::Register
end

function HaarDist(;
    n_qubits::Int64,
    n_samples::Int64,
)
    ensemble = randn(RNG, ComplexF64, (2^n_qubits, n_samples))

    return HaarDist(ensemble)
end
