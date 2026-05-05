export stochastic_collapse, wrong_collapse

function stochastic_collapse(ensemble::BatchedArrayReg, n_ancilla::Int, n_data::Int)
    batch_size = ensemble.nbatch
    n_a_dim = 1 << n_ancilla
    n_d_dim = 1 << n_data

    indices = Zygote.ignore() do
        col_offsets = (0:batch_size-1) .* n_a_dim
        # note that `measure` takes reversed order of qubits
        res = measure(ensemble, 1:n_ancilla)
        vec(Int.(res)) .+ 1 .+ col_offsets
    end

    state_3d = reshape(ensemble.state, n_a_dim, n_d_dim, batch_size)
    state_permuted = permutedims(state_3d, (2, 1, 3))
    state_2d = reshape(state_permuted, n_d_dim, :)

    collapsed_state = state_2d[:, indices]
    probs = sum(abs2, collapsed_state, dims=1)

    return collapsed_state ./ sqrt.(probs .+ 1e-12)
end

function wrong_collapse(ensemble::BatchedArrayReg, n_ancilla::Int, n_data::Int)
	batch_size = ensemble.nbatch
	n_a_dim = 1 << n_ancilla
	n_d_dim = 1 << n_data

	indices = Zygote.ignore() do
		col_offsets = (0:batch_size-1) .* n_a_dim
		# Measure HIGHER bits (the data bits)
		res = measure(ensemble, (n_data+1):(n_data+n_ancilla))
		vec(Int.(res)) .+ 1 .+ col_offsets
	end

	# Keep lower bits (the ancilla bits) by reshaping and slicing directly using data dim
	state_2d = reshape(ensemble.state, n_d_dim, :)
	collapsed_state = state_2d[:, indices]

	probs = sum(abs2, collapsed_state, dims=1)
	return collapsed_state ./ sqrt.(probs .+ 1e-12)
end

# function scramble!(

#     weight_schedule
# )
#     for t in 1:model.T
#         for s in 1:model.dataset_size
#             params = vcat(
#                 weight_schedule[t] * (rand(model.rng, model.n_qubits * 3) * pi/4 .- pi/8),
#                 weight_schedule[t] * (rand(model.rng, binomial(model.n_qubits, 2)) * 0.2 .+ 0.4) /
#                 (2.0 * sqrt(model.n_qubits)),
#             )
#             circuit = dispatch(model.forward_circuit, params)
#             model.forward_ensembles[s, t] = apply(model.forward_ensembles[s, t-1], circuit)
#         end
#     end
# end
