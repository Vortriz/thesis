export Register, BatchState, State

const Register{MT <: AbstractMatrix{ComplexF64}} = BatchedArrayReg{2, ComplexF64, MT}
const BatchState = AbstractMatrix{ComplexF64}
const State = AbstractVector{ComplexF64}

Base.convert(::Type{Register}, x::Matrix{ComplexF64}) =
    x |> BatchedArrayReg |> transpose_storage |> normalize!
Base.convert(::Type{Register}, x::LinearAlgebra.Transpose{ComplexF64, Matrix{ComplexF64}}) =
    x |> BatchedArrayReg |> normalize!


export AbstractParams
abstract type AbstractParams <: AbstractVector{Float64} end

Base.getindex(p::AbstractParams, xs...) = Base.getindex(p.params, xs...)
Base.size(p::AbstractParams) = Base.size(p.params)


export AbstractTrajectory, AbstractDist
abstract type AbstractDist end
abstract type AbstractTrajectory <: AbstractVector{AbstractDist} end

Base.getindex(traj::AbstractTrajectory, xs...) = Base.getindex(traj.steps, xs...)
Base.size(traj::AbstractTrajectory) = Base.size(traj.steps)
