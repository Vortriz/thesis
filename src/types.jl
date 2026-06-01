export Register, BatchState, State

const Register{MT <: AbstractMatrix{ComplexF64}} = BatchedArrayReg{2, ComplexF64, MT}
const BatchState = AbstractMatrix{ComplexF64}
const State = AbstractVector{ComplexF64}

Base.convert(::Type{Register}, x::Matrix{ComplexF64}) =
    x |> BatchedArrayReg |> transpose_storage |> normalize!
Base.convert(::Type{Register}, x::LinearAlgebra.Transpose{ComplexF64, Matrix{ComplexF64}}) =
    x |> BatchedArrayReg |> normalize!
