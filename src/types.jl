export CBArrayReg, CBMatrix, CState

const CBArrayReg{T, MT} = BatchedArrayReg{2, T, MT} # ConcreteBatchedArrayReg
const CBMatrix = AbstractMatrix{ComplexF64}
const CState = AbstractVector{ComplexF64}

Base.convert(::Type{CBArrayReg}, x::CBMatrix) = x |> StorageType |> batch_and_normalize


export Distribution
abstract type Distribution end


export CollapseMethod, Normal, Alternate

abstract type CollapseMethod end
struct Normal <: CollapseMethod end
struct Alternate <: CollapseMethod end


export TargetTrajectory, Diffusion, Direct

abstract type TargetTrajectory end
struct Diffusion <: TargetTrajectory end
struct Direct <: TargetTrajectory end


export ModelArch, TrainConfig, ModelState

struct ModelArch{CM <: CollapseMethod}
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
        collapse_method::CM=Normal(),
    ) where {CM <: CollapseMethod}
        n_qubits = n_data + n_ancilla
        ansatz = ansatz_builder(n_qubits, n_layers)
        ansatz_name = ansatz_builder |> nameof |> string
        n_params_ppb = ansatz |> parameters |> length

        return new{CM}(
            n_data,
            n_ancilla,
            n_qubits,
            n_layers,
            ansatz,
            ansatz_name,
            n_params_ppb,
            collapse_method,
        )
    end
end

struct TrainConfig{TT <: TargetTrajectory, IE <: Distribution, TE <: Distribution}
    dataset_size::Int64
    batch_size::Int64
    T::Int64
    initial_ensemble_type::Type{IE}
    initial_ensemble::CBArrayReg
    target_ensemble_type::Type{TE}
    target_trajectory_type::TT
    target_trajectory::Vector{CBArrayReg}
    target_schedule::Vector{Int64}
    epoch_schedule::Vector{Int64}
    optimizer::Optimisers.AbstractRule
end

function TrainConfig(
    target_trajectory_type::Direct;
    batch_size::Int64,
    initial_ensemble::IE,
    target_ensemble::TE,
    epoch_schedule::Vector{Int64},
    optimizer::Optimisers.AbstractRule,
) where {IE <: Distribution, TE <: Distribution}
    T = length(epoch_schedule)
    target_schedule = Device.ones(Int64, T)

    dataset_size = target_ensemble.ensemble.nbatch
    target_trajectory = [target_ensemble.ensemble]

    return TrainConfig{Direct, IE, TE}(
        dataset_size,
        batch_size,
        T,
        typeof(initial_ensemble),
        initial_ensemble.ensemble,
        typeof(target_ensemble),
        target_trajectory_type,
        target_trajectory,
        target_schedule,
        epoch_schedule,
        optimizer,
    )
end

function TrainConfig(
    target_trajectory_type::Diffusion;
    batch_size::Int64,
    initial_ensemble::IE,
    target_ensemble::TE,
    target_trajectory::Vector{CBArrayReg},
    epoch_schedule::Vector{Int64},
    optimizer::Optimisers.AbstractRule,
) where {IE <: Distribution, TE <: Distribution}
    T = length(target_trajectory) - 1
    target_schedule = Device.range(; start=1, stop=T, step=1) |> collect

    dataset_size = target_trajectory[begin].nbatch

    @assert length(epoch_schedule) == T "epoch_schedule must have the same length as the target trajectory (minus one)"
    @assert typeof(initial_ensemble) == Haar "Diffusion based training must always start from Haar random ensemble"

    return TrainConfig{Diffusion, IE, TE}(
        dataset_size,
        batch_size,
        T,
        typeof(initial_ensemble),
        initial_ensemble.ensemble,
        typeof(target_ensemble),
        target_trajectory_type,
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
