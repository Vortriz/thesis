module GQMLCUDAExt

using CUDA
using Yao: BatchedArrayReg, normalize!, cpu
using LinearAlgebra: mul!
using CairoMakie: GridPosition
using GQML: GQML, Register, AbstractDist, BatchState

CUDA.allowscalar(false)

Base.convert(::Type{Register}, x::CuMatrix{ComplexF64}) = x |> BatchedArrayReg |> normalize!
CUDA.cu(dist::D) where {D <: AbstractDist} = dist.register |> CUDA.cu |> typeof(dist)

function GQML.optimal_transport_plan(
    C::CuMatrix{Float64};
    β::Float64=0.05,
    max_iter::Int=500,
    L::Int=3,
)::CuMatrix{Float64}
    N1, N2 = size(C)

    a1 = CUDA.fill(1.0 / N1, N1)
    a2 = CUDA.fill(1.0 / N2, N2)

    P = CUDA.fill(1.0 / (N1 * N2), N1, N2)
    K = @. exp(-C / β)
    Q = CUDA.similar(P)

    u = CUDA.ones(Float64, N1)
    v = CUDA.ones(Float64, N2)

    Qv_buffer = CUDA.similar(u)
    QTu_buffer = CUDA.similar(v)

    for _ in 1:max_iter
        @. Q = K * P
        for _ in 1:L
            mul!(Qv_buffer, Q, v)
            @. u = a1 / (Qv_buffer + 1e-300)

            mul!(QTu_buffer, Q', u)
            @. v = a2 / (QTu_buffer + 1e-300)
        end
        @. P = u * Q * v'
    end

    return P
end

# [TODO] port all dist generation natively to CUDA (maybe?)

function _qkr_helper!(
    pos::GridPosition,
    reg::BatchedArrayReg{2, ComplexF64, CuMatrix{ComplexF64}},
    title::String,
)
    _qkr_helper!(pos, reg |> cpu, title)
end

function _bloch_helper!(
    pos::GridPosition,
    reg::BatchedArrayReg{2, ComplexF64, CuMatrix{ComplexF64}},
)
    _bloch_helper!(pos, reg |> cpu)
end

end
