export initialize_forward_ensemble!, scramble!

function initialize_forward_ensemble!(model::Model; spread::Float64)
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
