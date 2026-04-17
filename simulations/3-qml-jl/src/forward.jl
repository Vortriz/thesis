export initialize_forward_ensemble!, scramble!

function initialize_forward_ensemble!(model::Model, ::Clustered; spread::Float64=0.05)
	base_state = randn(model.rng, ComplexF64, 2^model.n_qubits)
    for i in 1:model.forward_ensemble_size
        model.forward_ensembles[i, 0] = (
				base_state .+ spread * randn(model.rng, ComplexF64, 2^model.n_qubits)
				# vcat(model.spread * randn(ComplexF64, 2^model.n_qubits - 1), 1)
				|> v -> reshape(v, :, 1)
				|> ConcreteArrayReg
				|> normalize!
		)
    end
end

function initialize_forward_ensemble!(model::Model, ::QKRLocalized; K::Float64=12.0, ħₛ::Float64=0.7)
    dims = 2^model.n_qubits
    @assert model.forward_ensemble_size <= dims "For QKRLocalized distribution, the forward ensemble size must be less than or equal to the Hilbert space dimension (2^n_qubits)."

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

	eigenstates = eigen(U).vectors

	for i in 1:model.forward_ensemble_size
	    model.forward_ensembles[i, 0] = (
				eigenstates[:, i]
				|> v -> reshape(v, :, 1)
				|> ConcreteArrayReg
		)
	end
end

function scramble!(
    model::Model;
    weight_schedule
)
    for t in 1:model.T
        for s in 1:model.forward_ensemble_size
            params = vcat(
                weight_schedule[t] * (rand(model.rng, model.n_qubits * 3) * pi/4 .- pi/8),
                weight_schedule[t] * (rand(model.rng, binomial(model.n_qubits, 2)) * 0.2 .+ 0.4) /
                (2.0 * sqrt(model.n_qubits)),
            )
            circuit = dispatch(model.forward_circuit, params)
            model.forward_ensembles[s, t] = apply(model.forward_ensembles[s, t-1], circuit)
        end
    end
end
