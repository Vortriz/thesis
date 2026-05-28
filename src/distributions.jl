export Clustered, QKRLocalized, XZCircle, TFIM, Haar

struct Clustered <: Distribution
    ensemble::CBArrayReg

    function Clustered(
        rng::AbstractRNG;
        n_qubits::Int64,
        n_samples::Int64,
        spread::Float64=0.05,
    )
        ensemble = (
            randn(rng, ComplexF64, (2^n_qubits, 1)) .+
            spread * randn(rng, ComplexF64, (2^n_qubits, n_samples))
        )

        return new(ensemble)
    end
end

struct QKRLocalized <: Distribution
    ensemble::CBArrayReg

    function QKRLocalized(;
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

        ensemble = reduce(
            hcat,
            [gen_qkrlocalized_states(n_qubits, K[i], ħₛ[i]) for i in eachindex(K)],
        )

        return new(ensemble)
    end
end

struct XZCircle <: Distribution
    ensemble::CBArrayReg

    function XZCircle(; n_samples::Int64)
        ensemble = zeros(ComplexF64, (2, n_samples))
        phis = rand(Float64, n_samples) * 2pi

        for i in 1:n_samples
            ensemble[:, i] = [cos(phis[i]), sin(phis[i])] .|> ComplexF64
        end

        return new(ensemble)
    end
end

struct TFIM <: Distribution
    ensemble::CBArrayReg

    function TFIM(; n_qubits::Int64, g::Vector{Float64})
        ensemble = zeros(ComplexF64, (2^n_qubits, length(g)))
        for (i, gᵢ) in enumerate(g)
            H = gen_tfim_hamiltonian(n_qubits, gᵢ)
            ensemble[:, i] = eigenstates(H).vectors[:, 1]
        end

        return new(ensemble)
    end
end

struct Haar <: Distribution
    ensemble::CBArrayReg

    function Haar(rng::AbstractRNG; n_qubits::Int64, n_samples::Int64)
        return new(randn(rng, ComplexF64, (2^n_qubits, n_samples)))
    end
end
