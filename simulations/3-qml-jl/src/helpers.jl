export apply_pqc

function apply_pqc(
    arch::ModelArch,
    input_ensemble::CBArrayReg,
    params::Vector{Float64},
)
    output_ensemble = apply(
        input_ensemble,
        dispatch(arch.ansatz, params),
    )

    collapsed_ensemble_matrix = collapse(
        arch.collapse_method,
        arch,
        output_ensemble,
    )

    return collapsed_ensemble_matrix
end


export loss_and_grads

function loss_and_grads(
    arch::ModelArch,
    params::Vector{Float64},
    input_ensemble::CBArrayReg,
    target_matrix::AbstractMatrix{ComplexF64},
)
    return (
        Zygote.withgradient(params) do p
            collapsed_ensemble_matrix = apply_pqc(
                arch,
                input_ensemble,
                p,
            )

            C = 1.0 .- abs2.(target_matrix' * collapsed_ensemble_matrix)

            Γ = Zygote.ignore() do
                optimal_transport_plan(C)
            end

            return dot(Γ, C)
        end
    )
end
