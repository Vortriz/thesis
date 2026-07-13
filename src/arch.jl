export AbstractAnsatz, HEA, EHA
abstract type AbstractAnsatz end

struct HEA{M <: AbstractMeasurement} <: AbstractAnsatz
    n_data::Int64
    n_ancilla::Int64
    n_qubits::Int64
    n_layers::Int64
    n_subblocks::Int64
    n_params::Int64
    circuit::ChainBlock{2}
    measurement::M
end

struct EHA{M <: AbstractMeasurement} <: AbstractAnsatz
    n_data::Int64
    n_ancilla::Int64
    n_qubits::Int64
    n_layers::Int64
    n_subblocks::Int64
    n_params::Int64
    circuit::ChainBlock{2}
    measurement::M
end

function init_ansatz(
    ::Type{T},
    circuit_builder::Function;
    n_data::Int64,
    n_ancilla::Int64,
    n_layers::Int64,
    n_subblocks::Int64,
    measurement::M,
) where {T <: AbstractAnsatz, M <: AbstractMeasurement}
    n_data <= 0 && throw(DomainError("Number of data qubits should be a positive integer."))
    n_ancilla <= 0 && throw(DomainError("Number of ancilla qubits should be a positive integer."))

    n_qubits = n_data + n_ancilla

    if n_subblocks == 0
        circuit = circuit_builder(n_qubits, n_layers)
    elseif n_subblocks > 0
        if n_layers % n_subblocks == 0
            circuit = circuit_builder(n_qubits, n_layers, n_subblocks)
        else
            error("Number of layers should be an integer multiple of number of subblocks.")
        end
    elseif n_subblocks < 0
        throw(DomainError("Number of data qubits should be a positive integer."))
    end

    n_params = circuit |> parameters |> length

    return T{M}(
        n_data,
        n_ancilla,
        n_qubits,
        n_layers,
        n_subblocks,
        n_params,
        circuit,
        measurement,
    )
end

HEA(; kwargs...) = init_ansatz(HEA, HEA_circuit; kwargs...)
EHA(; kwargs...) = init_ansatz(EHA, EHA_circuit; kwargs...)


export TrainConfig

struct TrainConfig{TT <: AbstractTrajectory}
    dataset_size::Int64
    batch_size::Int64
    T::Int64
    trajectory::TT
    epoch_schedule::Vector{Int64}
end

function TrainConfig(
    trajectory::Direct;
    batch_size::Int64,
    epoch_schedule::Vector{Int64},
)
    T = length(epoch_schedule)
    dataset_size = trajectory.steps[end].register.nbatch

    return TrainConfig{Direct}(
        dataset_size,
        batch_size,
        T,
        trajectory,
        epoch_schedule,
    )
end

function TrainConfig(
    trajectory::Diffusion;
    batch_size::Int64,
    epoch_schedule::Vector{Int64},
)
    T = length(epoch_schedule)
    @assert length(trajectory.steps) == T + 1 "Diffusion trajectory must have length(epoch_schedule)+1 steps."

    dataset_size = trajectory.steps[end].register.nbatch

    return TrainConfig{Diffusion}(
        dataset_size,
        batch_size,
        T,
        trajectory,
        epoch_schedule,
    )
end
