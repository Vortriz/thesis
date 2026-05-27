export gen_dist

function gen_dist(
    ::Clustered,
    rng::AbstractRNG;
    n_qubits::Int64,
    n_samples::Int64,
    spread::Float64=0.05,
)::CBArrayReg
    ensemble =
        (
            randn(rng, ComplexF64, (2^n_qubits, 1)) .+
            spread * randn(rng, ComplexF64, (2^n_qubits, n_samples))
        ) |> StorageType

    return ensemble |> batch_and_normalize
end

function gen_dist(
    ::QKRLocalized;
    n_qubits::Int64,
    K::Union{Float64, Vector{Float64}},
    ħₛ::Union{Float64, Vector{Float64}},
)::CBArrayReg
    if typeof(K) == Float64
        K = fill(K, length(ħₛ))
    end

    if typeof(ħₛ) == Float64
        ħₛ = fill(ħₛ, length(K))
    end

    return reduce(
        hcat,
        [gen_qkrlocalized_states(n_qubits, K[i], ħₛ[i]) for i in eachindex(K)],
    ) |> batch_and_normalize
end

function gen_dist(
    ::Circle;
    n_qubits::Int64,
    n_samples::Int64,
)::CBArrayReg
    @assert n_qubits == 1 "Circle distribution is only defined for 1 qubit system."

    ensemble = zeros(ComplexF64, (2^n_qubits, n_samples))

    phis = rand(Float64, n_samples) * 2pi
    for i in 1:n_samples
        ensemble[:, i] = (
            [cos(phis[i]), sin(phis[i])]
            .|>
            ComplexF64
        )
    end

    return ensemble |> StorageType |> batch_and_normalize
end

function gen_dist(
    ::TFIM;
    n_qubits::Int64,
    g::Vector{Float64},
)::CBArrayReg
    ensemble = zeros(ComplexF64, (2^n_qubits, length(g)))
    for (i, gᵢ) in enumerate(g)
        ensemble[:, i] = eigenstates(gen_tfim_hamiltonian(n_qubits, gᵢ)).vectors[:, 1]
    end

    return ensemble |> StorageType |> batch_and_normalize
end

function gen_dist(
    ::Haar,
    rng::AbstractRNG;
    n_qubits::Int64,
    n_samples::Int64,
)::CBArrayReg
    return randn(rng, ComplexF64, (2^n_qubits, n_samples)) |> StorageType |>
           batch_and_normalize
end
