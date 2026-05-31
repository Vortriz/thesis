export measure

function GQML.measure(
    ::Normal;
    n_data::Int64,
    n_ancilla::Int64,
    register::Register,
)::BatchState
    batch_size = register.nbatch
    n_a_dim = 1 << n_ancilla
    n_d_dim = 1 << n_data

    indices::Vector{Int64} = Zygote.ignore() do
        col_offsets = (0:(batch_size-1)) .* n_a_dim
        # Measure HIGHER bits (the data bits)
        res = measure(register, (n_data+1):(n_data+n_ancilla); rng=RNG)
        return vec(Int.(res)) .+ 1 .+ col_offsets
    end

    state_2d = reshape(register.state, n_d_dim, :)
    collapsed_state = state_2d[:, indices]

    probs = sum(abs2, collapsed_state; dims=1)
    return collapsed_state ./ sqrt.(probs .+ 1e-12)
end

function GQML.measure(
    ::Alternate;
    n_data::Int64,
    n_ancilla::Int64,
    register::Register,
)::BatchState
    batch_size = register.nbatch
    n_a_dim = 1 << n_ancilla
    n_d_dim = 1 << n_data

    indices::Vector{Int64} = Zygote.ignore() do
        col_offsets = (0:(batch_size-1)) .* n_a_dim
        res = measure(register, 1:n_ancilla; rng=RNG)
        return vec(Int.(res)) .+ 1 .+ col_offsets
    end

    state_3d = reshape(register.state, n_a_dim, n_d_dim, batch_size)
    state_permuted = permutedims(state_3d, (2, 1, 3))
    state_2d = reshape(state_permuted, n_d_dim, :)

    collapsed_state = state_2d[:, indices]
    probs = sum(abs2, collapsed_state; dims=1)

    return collapsed_state ./ sqrt.(probs .+ 1e-12)
end


export scramble

function scramble(;
    n_qubits::Int64,
    distribution::D,
    weight_schedule::Vector{Float64},
) where {D <: AbstractDist}

    T = length(weight_schedule)
    circuit = scramble_circuit(n_qubits)

    trajectory = Vector{AbstractDist}(undef, T + 1)
    trajectory[begin] = deepcopy(distribution)

    for t in 1:T
        reg = deepcopy(distribution.register)

        for r in 1:reg.nbatch
            reg_view = viewbatch(reg, r)
            # Run through all steps up to the current timestep t
            for prev_t in 1:t
                # Generate random parameters scaled by the weight schedule for this step
                params = vcat(
                    weight_schedule[prev_t] .*
                    (rand(RNG, Float64, n_qubits * 3) .* (pi / 4) .- (pi / 8)),
                    weight_schedule[prev_t] .*
                    (rand(RNG, Float64, binomial(n_qubits, 2)) .* 0.2 .+ 0.4) ./
                    (2.0 * sqrt(n_qubits)),
                )

                dispatch!(circuit, params)
                apply!(reg_view, circuit)
            end
        end

        trajectory[t+1] = ArbitraryDist(reg)
    end

    return reverse(trajectory)
end


export get_centered_amplitudes

function get_centered_amplitudes(ψ::State)
    dims = length(ψ)
    amplitudes = abs2.(ψ)
    _, idx = findmax(amplitudes)
    circshift!(amplitudes, dims - idx + 1)

    return amplitudes
end


export gen_qkr_operator

function gen_qkr_operator(;
    n_qubits::Int64,
    K::Float64,
    ħₛ::Float64,
)::AbstractQuantumObject{Operator}
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

    return Qobj(U)
end


export gen_tfim_hamiltonian

function gen_tfim_hamiltonian(;
    n_qubits::Int64,
    g::Float64,
)::AbstractQuantumObject{Operator}
    H = QT.Qobj(
        zeros(ComplexF64, (2^n_qubits, 2^n_qubits));
        dims=Tuple(fill(2, n_qubits)),
    )

    partial_term_1 = vcat(
        [QT.sigmaz(), QT.sigmaz()],
        fill(QT.eye(2), n_qubits - 2),
    )
    partial_term_2 = vcat(
        [QT.sigmax()],
        fill(QT.eye(2), n_qubits - 1),
    )

    for i in 0:(n_qubits-2)
        H -= reduce(QT.kron, circshift(partial_term_1, i))
    end
    for i in 0:(n_qubits-1)
        H -= g * reduce(QT.kron, circshift(partial_term_2, i))
    end

    return H
end

export magnetization

function magnetization(ψ::State)
    n = ψ |> length |> log2 |> Int64
    M = 0
    for (i, ψᵢ) in enumerate(ψ)
        ψᵢ_M = 0
        for spin in digits(i - 1; base=2, pad=n) |> reverse |> BitVector
            ψᵢ_M += abs2(ψᵢ) * (spin ? 1 : -1)
        end
        ψᵢ_M /= n
        @assert abs(ψᵢ_M) <= 1
        M += abs(ψᵢ_M)
    end

    return M
end
