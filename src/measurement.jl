export AbstractMeasurement, Normal, Alternate, measure
abstract type AbstractMeasurement end

struct Normal <: AbstractMeasurement end

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


struct Alternate <: AbstractMeasurement end

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
