export collapse

function collapse(
    ::Alternate,
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
    ::Normal,
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

    return trajectory |> reverse
end


export batch_and_normalize

function batch_and_normalize(ensemble::Matrix{ComplexF64})::CBArrayReg
    return ensemble |> BatchedArrayReg |> transpose_storage |> normalize!
end

function batch_and_normalize(ensemble::CuMatrix{ComplexF64})::CBArrayReg
    return ensemble |> BatchedArrayReg |> normalize!
end


export get_final_training_loss

function get_final_training_loss(loss_history::Vector{Vector{Float64}})::Float64
    return last(loss_history[end], 10) |> mean
end


export get_centered_amplitudes

function get_centered_amplitudes(ψ::CState)
    dims = length(ψ)
    amplitudes = abs2.(ψ)
    _, idx = findmax(amplitudes)
    circshift!(amplitudes, dims - idx + 1)

    return amplitudes
end


export gen_qkrlocalized_states

# [MARK] try using QuantumToolbox.jl
function gen_qkrlocalized_states(
    n_qubits::Int64,
    K::Float64,
    ħₛ::Float64,
)::CBMatrix
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

    return eigen(U).vectors
end


export gen_tfim_hamiltonian

function gen_tfim_hamiltonian(
    n_qubits::Int64,
    g::Float64,
)::AbstractQuantumObject{Operator}
    H = Qobj(
        zeros(ComplexF64, (2^n_qubits, 2^n_qubits));
        dims=Tuple(fill(2, n_qubits)),
    )
    partial_term_1 = vcat(
        [sigmaz(), sigmaz()],
        fill(eye(2), n_qubits-2),
    )
    partial_term_2 = vcat(
        [sigmax()],
        fill(eye(2), n_qubits-1),
    )
    for i in 0:(n_qubits-2)
        H -= reduce(kron, circshift(partial_term_1, i))
    end
    for i in 0:(n_qubits-1)
        H -= g * reduce(kron, circshift(partial_term_2, i))
    end

    return H
end


export magnetization

function magnetization(ψ::CState)
    n = ψ |> length |> log2 |> Int64
    M = 0
    for (i, ψᵢ) in enumerate(ψ)
        ψᵢ_M = 0
        for spin in digits(i-1; base=2, pad=n) |> reverse |> BitVector
            ψᵢ_M += abs2(ψᵢ) * (spin ? 1 : -1)
        end
        ψᵢ_M /= n
        @assert abs(ψᵢ_M) <= 1
        M += abs(ψᵢ_M)
    end

    return M
end
