export Distribution, clustered, qkrlocalized, circle, tfim, haar

@enum Distribution begin
    clustered
    qkrlocalized
    circle
    tfim
    haar
end

export CollapseMethod, normal, alternate

@enum CollapseMethod begin
    normal
    alternate
end

export TargetTrajectoryType, diffusion, direct

@enum TargetTrajectoryType begin
    diffusion
    direct
end


export CBArrayReg, CBMatrix

# ConcreteBatchedArrayReg
const CBArrayReg{T, MT} = BatchedArrayReg{2, T, MT}
const CBMatrix = AbstractMatrix{ComplexF64}


export ModelArch, TrainConfig, ModelState

struct ModelArch{CM}
    n_data::Int64
    n_ancilla::Int64
    n_qubits::Int64
    n_layers::Int64
    ansatz::ChainBlock{2}
    ansatz_name::String
    n_params_ppb::Int64 # Number of parameters per PQC block
    collapse_method::CM

    function ModelArch(;
        n_data::Int64,
        n_ancilla::Int64,
        n_layers::Int64,
        ansatz_builder::Function,
        collapse_method::CollapseMethod=normal,
    )
        n_qubits = n_data + n_ancilla
        ansatz = ansatz_builder(n_qubits, n_layers)
        ansatz_name = ansatz_builder |> nameof |> string
        n_params_ppb = ansatz |> parameters |> length
        CM = collapse_method |> Val |> typeof

        return new{CM}(
            n_data,
            n_ancilla,
            n_qubits,
            n_layers,
            ansatz,
            ansatz_name,
            n_params_ppb,
            Val(collapse_method),
        )
    end
end

struct TrainConfig{TT}
    dataset_size::Int64
    batch_size::Int64
    T::Int64
    initial_ensemble::CBArrayReg
    target_trajectory_type::TT
    target_trajectory::Vector{CBArrayReg}
    target_schedule::Vector{Int64}
    epoch_schedule::Vector{Int64}
    optimizer::Optimisers.AbstractRule
end

function TrainConfig(
    ::Val{direct};
    batch_size::Int64,
    initial_ensemble::CBArrayReg,
    target_ensemble::CBArrayReg,
    epoch_schedule::Vector{Int64},
    optimizer::Optimisers.AbstractRule,
)
    T = length(epoch_schedule)
    target_schedule = Device.ones(Int64, T)

    dataset_size = target_ensemble.nbatch
    target_trajectory = [target_ensemble]
    TT = direct |> Val |> typeof

    return TrainConfig{TT}(
        dataset_size,
        batch_size,
        T,
        initial_ensemble,
        Val(direct),
        target_trajectory,
        target_schedule,
        epoch_schedule,
        optimizer,
    )
end

function TrainConfig(
    ::Val{diffusion};
    batch_size::Int64,
    initial_ensemble::CBArrayReg,
    target_trajectory::Vector{CBArrayReg},
    epoch_schedule::Vector{Int64},
    optimizer::Optimisers.AbstractRule,
)
    T = length(target_trajectory) - 1
    target_schedule = Device.range(; start=T, stop=1, step=-1) |> collect

    dataset_size = target_trajectory[begin].nbatch

    @assert length(epoch_schedule) == T "epoch_schedule must have the same length as the target trajectory (minus one)"
    TT = diffusion |> Val |> typeof

    return TrainConfig{TT}(
        dataset_size,
        batch_size,
        T,
        initial_ensemble,
        Val(diffusion),
        target_trajectory,
        target_schedule,
        epoch_schedule,
        optimizer,
    )
end

mutable struct ModelState
    current_params::Vector{Float64}
    current_ensemble_batch::CBArrayReg
    target_matrix_batch::CBMatrix

    ModelState() = new()
end
