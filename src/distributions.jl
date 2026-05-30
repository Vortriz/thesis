export ClusteredDist

struct ClusteredDist <: AbstractDist
    register::Register

    function ClusteredDist(;
        n_qubits::Int64,
        n_samples::Int64,
        spread::Float64=0.05,
    )
        ensemble = (
            randn(RNG, ComplexF64, (2^n_qubits, 1)) .+
            spread * randn(RNG, ComplexF64, (2^n_qubits, n_samples))
        )

        return new(ensemble)
    end
end


export QKRLocalizedDist

struct QKRLocalizedDist <: AbstractDist
    register::Register

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

        ensemble = reduce(
            hcat,
            [
                gen_qkrlocalized_states(; n_qubits=n_qubits, K=K[i], ħₛ=ħₛ[i]) for
                i in eachindex(K)
            ],
        )

        return new(ensemble)
    end
end


export CircleDist

struct CircleDist <: AbstractDist
    register::Register

    function CircleDist(; n_samples::Int64)
        ensemble = zeros(ComplexF64, (2, n_samples))
        phis = rand(RNG, Float64, n_samples) * 2pi

        for i in 1:n_samples
            ensemble[:, i] = [cos(phis[i]), sin(phis[i])] .|> ComplexF64
        end

        return new(ensemble)
    end
end


export TFIMDist

struct TFIMDist <: AbstractDist
    register::Register

    function TFIMDist(;
        n_qubits::Int64,
        g::Vector{Float64},
    )
        ensemble = zeros(ComplexF64, (2^n_qubits, length(g)))

        for (i, gᵢ) in enumerate(g)
            H = gen_tfim_hamiltonian(; n_qubits=n_qubits, g=gᵢ)
            ensemble[:, i] = eigenstates(H).vectors[:, 1]
        end

        return new(ensemble)
    end
end


export HaarDist

struct HaarDist <: AbstractDist
    register::Register

    function HaarDist(;
        n_qubits::Int64,
        n_samples::Int64,
    )
        ensemble = randn(RNG, ComplexF64, (2^n_qubits, n_samples))

        return new(ensemble)
    end
end
