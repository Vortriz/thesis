# [TODO] remove exports once training loop is internalized

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
        arch.n_data,
        arch.n_ancilla,
        output_ensemble,
    )

    return collapsed_ensemble_matrix
end


export loss_and_grads

function loss_and_grads(
    arch::ModelArch,
    model_state::ModelState,
)
    params = model_state.current_params
    target_matrix = model_state.target_matrix_batch
    input_ensemble = model_state.current_ensemble_batch

    return (
        Zygote.withgradient(params) do p
            collapsed_ensemble_matrix = apply_pqc(
                arch,
                input_ensemble,
                p,
            )

            C = 1.0 .- abs2.(target_matrix' * collapsed_ensemble_matrix)

            Γ = Zygote.ignore() do
                return optimal_transport_plan(C; β=0.01, max_iter=500, L=1)
            end

            return dot(Γ, C)
        end
    )
end


export sample_batch!

function sample_batch!(
    config::TrainConfig,
    model_state::ModelState,
    current_ensemble::CBArrayReg,
    target_matrix::CBMatrix,
)
    model_state.current_ensemble_batch =
        reduce(
            hcat,
            sample(
                eachcol(current_ensemble.state),
                config.batch_size;
                replace=false,
            ),
        ) |> BatchedArrayReg

    return model_state.target_matrix_batch = reduce(
        hcat,
        sample(
            eachcol(target_matrix),
            config.batch_size;
            replace=false,
        ),
    )
end
