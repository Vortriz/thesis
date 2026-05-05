export gen_dist

# [MARK] should we have all BatchedArrayReg with transposed storage?

function gen_dist(
    ::Val{clustered},
    rng::AbstractRNG;
    n_qubits::Int64,
    n_samples::Int64,
    spread::Float64=0.05,
)::BatchedArrayReg
    ensemble = (
        randn(rng, ComplexF64, (2^n_qubits, 1))
            .+ spread * randn(rng, ComplexF64, (2^n_qubits, n_samples))
    )
	ensemble = ensemble |> BatchedArrayReg
	normalize!(ensemble)

	return ensemble
end

# [MARK] try using QuantumToolbox.jl
function gen_dist(
    ::Val{qkrlocalized};
    n_qubits::Int64,
    n_samples::Int64,
    K::Float64=12.0,
    ħₛ::Float64=0.7,
)::BatchedArrayReg
    dims = 2^n_qubits
    @assert n_samples <= dims "Number of samples cannot exceed the dimension of the Hilbert space."

    m_vec = [0:dims/2-1; -dims/2:-1]
    U = zeros(ComplexF64, (dims, dims))

	Threads.@threads for idx in CartesianIndices(U)
		i, j = idx.I
		m₁, m₂ = m_vec[i], m_vec[j]
		d = m₂ - m₁
		if d > dims/2  d -= dims end
		if d < -dims/2 d += dims end
		U[idx] = ℯ^(-im/2 * ħₛ * m₂^2) * im^d * besselj(d, K / ħₛ)
	end

	eigenstates = eigen(U).vectors[:, 1:n_samples]
	eigenstates = eigenstates |> BatchedArrayReg
	normalize!(eigenstates)

	return eigenstates
end

function gen_dist(
    ::Val{circle};
    n_qubits::Int64,
    n_samples::Int64,
)::BatchedArrayReg
    @assert n_qubits == 1 "Circle distribution is only defined for 1 qubit system."

    ensemble = zeros(ComplexF64, (2^n_qubits, n_samples))

    phis = rand(Float64, n_samples) * 2pi
    for i in 1:n_samples
        ensemble[:, i] = (
            [cos(phis[i]), sin(phis[i])]
            .|> ComplexF64
        )
    end

    ensemble = ensemble |> BatchedArrayReg
    normalize!(ensemble)

    return ensemble
end

function gen_dist(
    ::Val{haar};
    n_qubits::Int64,
    n_samples::Int64,
)::BatchedArrayReg
	return rand_state(ComplexF64, n_qubits; nbatch=n_samples, no_transpose_storage=true)
end
