export Distribution, clustered, qkrlocalized, circle, haar

@enum Distribution begin
    clustered
    circle
    qkrlocalized
    haar
end

export TargetSchedule, diffusion, direct

@enum TargetSchedule begin
    diffusion
    direct
end

export Model

struct Model
    n_data::Int64
    n_ancilla::Int64
    n_qubits::Int64

    n_layers::Int64
    T::Int64

    dataset_size::Int64
    batch_size::Int64

    target_schedule::Union{Vector{Int64}, Val{TargetSchedule}}
    epoch_schedule::Vector{Int64}

    function Model(;
        n_data,
        n_ancilla,
        n_layers,
        dataset_size,
        batch_size,
        target_schedule,
        epoch_schedule,
    )
        T = epoch_schedule |> length

        if target_schedule == :diffusion
            target_schedule = range(start=T, stop=1, step=-1) |> collect
        elseif target_schedule == :direct
            target_schedule = ones(Int64, T)
        elseif target_schedule isa Vector{Int64}
            if length(target_schedule) != length(epoch_schedule)
                throw(ArgumentError("target_schedule and epoch_schedule must have the same length"))
            end
        else
            throw(ArgumentError("target_schedule must be either a Vector{Int64} or an instance of TargetSchedule"))
        end

        n_qubits = n_data + n_ancilla

        new(
            n_data,
            n_ancilla,
            n_qubits,
            n_layers,
            T,
            dataset_size,
            batch_size,
            target_schedule,
            epoch_schedule,
        )
    end
end


export TrainingStrategy, SequentialStrategy, DirectStrategy

abstract type TrainingStrategy end
abstract type SequentialStrategy <: TrainingStrategy end
abstract type DirectStrategy <: TrainingStrategy end


export ConcreteArrayReg, ConcreteBatchedArrayReg, Ensemble, OffsetEnsemble, OffsetEnsembleCollection

const ConcreteArrayReg = ArrayReg{2, ComplexF64, Matrix{ComplexF64}}
const ConcreteBatchedArrayReg = BatchedArrayReg{2, ComplexF64, Matrix{ComplexF64}}
const Ensemble = Vector{ConcreteArrayReg}
# const OffsetEnsemble = OffsetVector{ConcreteArrayReg, Ensemble}
# const OffsetEnsembleCollection = OffsetMatrix{ConcreteArrayReg, Matrix{ConcreteArrayReg}}
