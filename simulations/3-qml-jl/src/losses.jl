export mmd_distance, wasserstein_distance, optimal_transport_plan, sinkhorn_distance


# Maximum Mean Discrepancy (MMD)
function mmd_distance(
    ensemble1::CBMatrix,
    ensemble2::CBMatrix,
)::Float64

    ensemble1_c = ensemble1'
    ensemble2_c = ensemble2'

    r11 = 1.0 - mean(abs2.(ensemble1_c * ensemble1))
    r22 = 1.0 - mean(abs2.(ensemble2_c * ensemble2))
    r12 = 1.0 - mean(abs2.(ensemble1_c * ensemble2))

    return 2.0 * r12 - r11 - r22
end


# Based on https://github.com/xieyujia/IPOT/blob/master/ipot.py
function optimal_transport_plan(
    C::AbstractMatrix{Float64};
    beta::Float64=0.01,
    max_iter::Int=500,
    L::Int=1,
)::AbstractMatrix{Float64}

    N1, N2 = size(C)
    a1 = Device.fill(1.0 / N1, N1)
    a2 = Device.fill(1.0 / N2, N2)

    P = Device.fill(1.0 / (N1 * N2), N1, N2)
    K = exp.(-C ./ beta)
    u = Device.ones(Float64, N1)
    v = Device.ones(Float64, N2)

    for _ in 1:max_iter
        Q = K .* P
        for _ in 1:L
            u = a1 ./ (Q * v)
            v = a2 ./ (Q' * u)
        end
        P = u .* Q .* v'
    end

    return P
end


# Wasserstein (IPOT)
function wasserstein_distance(
    ensemble1::CBMatrix,
    ensemble2::CBMatrix;
    beta::Float64=0.01,
    max_iter::Int=500,
    L::Int=1,
)::Float64

    C = 1.0 .- abs2.(ensemble1' * ensemble2)
    P = optimal_transport_plan(C; beta, max_iter, L)

    return sum(P .* C)
end


function sinkhorn_distance(
    ensemble1::CBMatrix,
    ensemble2::CBMatrix,
)::Float64

    N1 = size(ensemble1, 2)
    N2 = size(ensemble2, 2)
    a1 = ones(Float64, (N1)) / N1
    a2 = ones(Float64, (N2)) / N2
    C = 1.0 .- abs2.(ensemble1' * ensemble2)

    # [MARK] does not work with CUDA
    return sinkhorn_divergence(
        a1, a2, C, 0.03; maxiter=1000, atol=rtol = 0, regularization=true,
    )
end
