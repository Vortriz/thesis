export collapse

function collapse(
    ::Val{alternate},
    n_data::Int64,
    n_ancilla::Int64,
    ensemble::CBArrayReg,
)::CBMatrix

    batch_size = ensemble.nbatch
    n_a_dim = 1 << n_ancilla
    n_d_dim = 1 << n_data

    indices::Vector{Int64} = Zygote.ignore() do
        col_offsets = (0:(batch_size-1)) .* n_a_dim
        res = measure(ensemble, 1:n_ancilla)
        return vec(Int.(res)) .+ 1 .+ col_offsets
    end

    state_3d = reshape(ensemble.state, n_a_dim, n_d_dim, batch_size)
    state_permuted = permutedims(state_3d, (2, 1, 3))
    state_2d = reshape(state_permuted, n_d_dim, :)

    collapsed_state = state_2d[:, indices]
    probs = sum(abs2, collapsed_state; dims=1)

    return collapsed_state ./ sqrt.(probs .+ 1e-12)
end

function collapse(
    ::Val{normal},
    n_data::Int64,
    n_ancilla::Int64,
    ensemble::CBArrayReg,
)::CBMatrix

    batch_size = ensemble.nbatch
    n_a_dim = 1 << n_ancilla
    n_d_dim = 1 << n_data

    indices::Vector{Int64} = Zygote.ignore() do
        col_offsets = (0:(batch_size-1)) .* n_a_dim
        # Measure HIGHER bits (the data bits)
        res = measure(ensemble, (n_data+1):(n_data+n_ancilla))
        return vec(Int.(res)) .+ 1 .+ col_offsets
    end

    state_2d = reshape(ensemble.state, n_d_dim, :)
    collapsed_state = state_2d[:, indices]

    probs = sum(abs2, collapsed_state; dims=1)
    return collapsed_state ./ sqrt.(probs .+ 1e-12)
end


export scramble

function scramble(
    rng::AbstractRNG;
    n_qubits::Int64,
    ensemble::CBArrayReg,
    weight_schedule::Vector{Float64},
)::Vector{CBArrayReg}

    T = weight_schedule |> length
    circuit = scramble_circuit(n_qubits)

    trajectory = Vector{CBArrayReg}(undef, T + 1)
    trajectory[1] = copy(ensemble)

    for t in 1:T
        current_ensemble = copy(ensemble)

        for s in 1:ensemble.nbatch
            reg = viewbatch(current_ensemble, s)
            # Run through all steps up to the current timestep t
            for prev_t in 1:t
                # Generate random parameters scaled by the weight schedule for this step
                params = vcat(
                    weight_schedule[prev_t] .*
                    (rand(rng, Float64, n_qubits * 3) .* (pi / 4) .- (pi / 8)),
                    weight_schedule[prev_t] .*
                    (rand(rng, Float64, binomial(n_qubits, 2)) .* 0.2 .+ 0.4) ./
                    (2.0 * sqrt(n_qubits)),
                )

                dispatch!(circuit, params)
                apply!(reg, circuit)
            end
        end

        trajectory[t+1] = current_ensemble
    end

    return trajectory
end


export batch_and_normalize

function batch_and_normalize(ensemble::Matrix{ComplexF64})::CBArrayReg
    reg = ensemble |> BatchedArrayReg |> transpose_storage
    normalize!(reg)

    return reg
end

function batch_and_normalize(ensemble::CuMatrix{ComplexF64})::CBArrayReg
    reg = ensemble |> BatchedArrayReg
    normalize!(reg)

    return reg
end


export get_final_training_loss

get_final_training_loss(loss_history::Vector{Vector{Float64}})::Float64 = last(loss_history[end], 10) |> mean
