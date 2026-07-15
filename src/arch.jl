export AbstractAnsatz, HEA, EHA
abstract type AbstractAnsatz end

macro ansatz(name)
    return quote
        struct $name{M <: AbstractMeasurement} <: AbstractAnsatz
            n_data::Int64
            n_ancilla::Int64
            n_qubits::Int64
            n_layers::Union{Int64, Vector{Int64}}
            n_params::Int64
            circuit::ChainBlock{2}
            measurement::M

            function $name(;
                n_data::Int64,
                n_ancilla::Int64,
                n_layers::Union{Int64, Vector{Int64}},
                measurement::M,
            ) where {M <: AbstractMeasurement}
                n_data <= 0 && throw(
                    DomainError(
                        n_data,
                        "Number of data qubits should be a positive integer",
                    ),
                )
                n_ancilla <= 0 && throw(
                    DomainError(
                        n_ancilla,
                        "Number of ancilla qubits should be a positive integer",
                    ),
                )
                n_qubits = n_data + n_ancilla

                if n_layers isa Int64
                    n_layers <= 0 && throw(
                        DomainError(
                            n_layers,
                            "Number of layers should be a positive integer",
                        ),
                    )
                elseif n_layers isa Vector{Int64}
                    any(i -> i <= 1, n_layers) && throw(
                        DomainError(
                            n_layers,
                            "Number of layers for each subblock should be a positive even integer >= 4",
                        ),
                    )
                end

                circuit = ansatz($name{typeof(measurement)}, n_qubits, n_layers)
                n_params = nparameters(circuit)

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
    end |> esc
end

@ansatz HEA
@ansatz EHA


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
    dataset_size = trajectory[end].register.nbatch

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
    @assert length(trajectory.steps) == T + 1 "Diffusion trajectory must have length(epoch_schedule)+1 steps"

    dataset_size = trajectory[end].register.nbatch

    return TrainConfig{Diffusion}(
        dataset_size,
        batch_size,
        T,
        trajectory,
        epoch_schedule,
    )
end
