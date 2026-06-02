export AbstractAnsatz, HEA, EHA
abstract type AbstractAnsatz end

struct HEA{M <: AbstractMeasurement} <: AbstractAnsatz
    n_data::Int64
    n_ancilla::Int64
    n_qubits::Int64
    n_layers::Int64
    n_params::Int64
    circuit::ChainBlock{2}
    measurement::M

    function HEA(;
        n_data::Int64,
        n_ancilla::Int64,
        n_layers::Int64,
        measurement::M,
    ) where {M <: AbstractMeasurement}
        n_qubits = n_data + n_ancilla
        circuit = HEA_circuit(n_qubits, n_layers)
        n_params = circuit |> parameters |> length

        return new{M}(
            n_data,
            n_ancilla,
            n_qubits,
            n_layers,
            n_params,
            circuit,
            measurement,
        )
    end
end

struct EHA{M <: AbstractMeasurement} <: AbstractAnsatz
    n_data::Int64
    n_ancilla::Int64
    n_qubits::Int64
    n_layers::Int64
    n_params::Int64
    circuit::ChainBlock{2}
    measurement::M

    function EHA(;
        n_data::Int64,
        n_ancilla::Int64,
        n_layers::Int64,
        measurement::M,
    ) where {M <: AbstractMeasurement}
        n_qubits = n_data + n_ancilla
        circuit = EHA_circuit(n_qubits, n_layers)
        n_params = circuit |> parameters |> length

        return new{M}(
            n_data,
            n_ancilla,
            n_qubits,
            n_layers,
            n_params,
            circuit,
            measurement,
        )
    end
end


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
