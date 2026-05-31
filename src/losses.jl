export mmd_distance, wasserstein_distance, optimal_transport_plan, sinkhorn_distance


# Maximum Mean Discrepancy (MMD)
function mmd_distance(
    ensemble1::BatchState,
    ensemble2::BatchState,
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
    C::Matrix{Float64};
    β::Float64=0.05,
    max_iter::Int=500,
    L::Int=3,
)::Matrix{Float64}
    N1, N2 = size(C)

    a1 = fill(1.0 / N1, N1)
    a2 = fill(1.0 / N2, N2)

    P = fill(1.0 / (N1 * N2), N1, N2)
    K = @. exp(-C / β)
    Q = similar(P)

    u = ones(Float64, N1)
    v = ones(Float64, N2)

    Qv_buffer = similar(u)
    QTu_buffer = similar(v)

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


# Wasserstein (IPOT)
function wasserstein_distance(
    ensemble1::BatchState,
    ensemble2::BatchState;
    β::Float64=0.05,
    max_iter::Int=500,
    L::Int=3,
)::Float64

    C = 1.0 .- abs2.(ensemble1' * ensemble2)
    P = optimal_transport_plan(C; β, max_iter, L)

    return sum(P .* C)
end
