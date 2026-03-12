export mmd_distance, wasserstein_distance, sinkhorn_distance, ipot

# Maximum Mean Discrepancy (MMD)
function mmd_distance(
	ensemble1::Matrix{ComplexF64},
	ensemble2::Matrix{ComplexF64}
)::Float64
    ensemble1_c = ensemble1'
    ensemble2_c = ensemble2'

    r11 = 1.0 - mean(abs2.(ensemble1_c * ensemble1))
    r22 = 1.0 - mean(abs2.(ensemble2_c * ensemble2))
    r12 = 1.0 - mean(abs2.(ensemble1_c * ensemble2))

    return 2.0 * r12 - r11 - r22
end

function mmd_distance(
    ensemble1::Ensemble,
    ensemble2::Ensemble
)::Float64
    return mmd_distance(Matrix(ensemble1), Matrix(ensemble2))
end

# Wasserstein (IPOT)
# Based on https://github.com/xieyujia/IPOT/blob/master/ipot.py
function wasserstein_distance(
	ensemble1::Union{Matrix{ComplexF64}, SubArray},
	ensemble2::Matrix{ComplexF64};
    beta::Float64 = 0.01,
    max_iter::Int = 1000,
    L::Int = 1,
    return_map::Bool = false,
)::Union{Float64, Matrix{Float64}}
    N1 = size(ensemble1, 2)
	N2 = size(ensemble2, 2)
	a1 = ones(Float64, (N1)) ./ N1
	a2 = ones(Float64, (N2)) ./ N2
	C = 1.0 .- abs2.(ensemble1' * ensemble2)

    P = ones(Float64, N1, N2) ./ (N1 * N2)
    K = exp.(-(C ./ beta))
    Q = similar(P)
    u = ones(Float64, N1)
    v = ones(Float64, N2)
    Qv_buffer = similar(u)
    QTu_buffer = similar(v)

    for _ in 1:max_iter
        @. Q = K * P
        for _ in 1:L
            mul!(Qv_buffer, Q, v)
            @. u = a1 / Qv_buffer

            mul!(QTu_buffer, Q', u)
            @. v = a2 / QTu_buffer
        end
        @. P = u * Q * v'
    end

    if return_map
        return P
    else
        W = sum(P .* C)
        return W
    end
end

function wasserstein_distance(
    ensemble1::Ensemble,
    ensemble2::Ensemble; kwargs...
)::Union{Float64, Matrix{Float64}}
    return wasserstein_distance(Matrix(ensemble1), Matrix(ensemble2); kwargs...)
end

# Specialized IPOT that takes a cost matrix directly
function ipot(
	C::Matrix{Float64};
    beta::Float64 = 0.01,
    max_iter::Int = 100, # Lower default for inner loops
    L::Int = 1,
)::Matrix{Float64}
    N1, N2 = size(C)
	a1 = fill(1.0 / N1, N1)
	a2 = fill(1.0 / N2, N2)

    P = fill(1.0 / (N1 * N2), N1, N2)
    K = exp.(-(C ./ beta))
    Q = similar(P)
    u = ones(Float64, N1)
    v = ones(Float64, N2)
    Qv_buffer = similar(u)
    QTu_buffer = similar(v)

    for _ in 1:max_iter
        @. Q = K * P
        for _ in 1:L
            mul!(Qv_buffer, Q, v)
            @. u = a1 / Qv_buffer

            mul!(QTu_buffer, Q', u)
            @. v = a2 / QTu_buffer
        end
        @. P = u * Q * v'
    end

    return P
end


function sinkhorn_distance(
	ensemble1::Matrix{ComplexF64},
	ensemble2::Matrix{ComplexF64}
)::Float64

	N1 = size(ensemble1, 2)
	N2 = size(ensemble2, 2)
	a1 = ones(Float64, (N1)) / N1
	a2 = ones(Float64, (N2)) / N2
	C = 1.0 .- abs2.(ensemble1' * ensemble2)

	return sinkhorn_divergence(
        a1, a2, C, 0.03; maxiter=1000, atol=rtol = 0, regularization=true
    )
end

function sinkhorn_distance(ensemble1::Ensemble, ensemble2::Ensemble)::Float64
    return sinkhorn_distance(Matrix(ensemble1), Matrix(ensemble2))
end
