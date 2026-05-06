export Distribution, clustered, qkrlocalized, circle, haar

@enum Distribution begin
    clustered
    circle
    qkrlocalized
    haar
end

export CollapseMethod, normal, alternate

@enum CollapseMethod begin
    normal
    alternate
end

export TargetSchedule, diffusion, direct

@enum TargetSchedule begin
    diffusion
    direct
end


export ModelArch, TrainConfig

struct ModelArch
    n_data::Int64
    n_ancilla::Int64
    n_qubits::Int64
    n_layers::Int64
    ansatz::ChainBlock{2}
    n_params_ppb::Int64 # Number of parameters per PQC block
    collapse_method::CollapseMethod

    function ModelArch(;
        n_data::Int64,
        n_ancilla::Int64,
        n_layers::Int64,
        ansatz_builder::Function,
        collapse_method::CollapseMethod = normal
    )
        n_qubits = n_data + n_ancilla
        ansatz = ansatz_builder(n_qubits, n_layers)
        n_params_ppb = ansatz |> parameters |> length
        new(n_data, n_ancilla, n_qubits, n_layers, ansatz, n_params_ppb, collapse_method)
    end
end

struct TrainConfig
    dataset_size::Int64
    batch_size::Int64
    T::Int64
    target_schedule::Vector{Int64}
    epoch_schedule::Vector{Int64}
    optimizer::Optimisers.AbstractRule

    function TrainConfig(;
        dataset_size::Int64,
        batch_size::Int64,
        target_schedule::Union{Symbol, Vector{Int64}},
        epoch_schedule::Vector{Int64},
        optimizer::Optimisers.AbstractRule
    )
        T = length(epoch_schedule)

        if target_schedule == :diffusion
            target_schedule_vec = range(start=T, stop=1, step=-1) |> collect
        elseif target_schedule == :direct
            target_schedule_vec = ones(Int64, T)
        elseif target_schedule isa Vector{Int64}
            if length(target_schedule) != length(epoch_schedule)
                throw(ArgumentError("target_schedule and epoch_schedule must have the same length"))
            end
            target_schedule_vec = target_schedule
        else
            throw(ArgumentError("target_schedule must be either a Vector{Int64} or a recognized Symbol"))
        end

        new(dataset_size, batch_size, T, target_schedule_vec, epoch_schedule, optimizer)
    end
end


export CTBArrayReg, CTBMatrix

# ConcreteTransposedBatchedArrayReg
const CTBArrayReg = BatchedArrayReg{2, ComplexF64, Transpose{ComplexF64, Matrix{ComplexF64}}}
const CTBMatrix = Union{Matrix{ComplexF64}, LinearAlgebra.Transpose{ComplexF64, Matrix{ComplexF64}}}
