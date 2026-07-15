function apply_pqc(
    ansatz::A,
    register::Register,
    params::Vector{Float64},
) where {A <: AbstractAnsatz}
    output_register = apply(
        register,
        dispatch(ansatz.circuit, params),
    )

    return GQML.measure(
        ansatz.measurement;
        n_data=ansatz.n_data,
        n_ancilla=ansatz.n_ancilla,
        register=output_register,
    )
end


export train

function train(
    ansatz::A,
    config::TrainConfig;
    params::Matrix{Float64},
    optimizer::Optimisers.AbstractRule,
    callback=(loss, step) -> nothing,
) where {A <: AbstractAnsatz}
    params = deepcopy(params)
    loss_history = [zeros(Float64, n) for n in config.epoch_schedule]
    current_reg = deepcopy(config.trajectory[begin].register)

    @progress for t in 1:config.T
        current_params = params[:, t]
        opt_state = Optimisers.setup(optimizer, current_params)

        append_qubits!(current_reg, ansatz.n_ancilla)

        if typeof(config) == TrainConfig{Direct}
            target_matrix = config.trajectory[end].register.state
        elseif typeof(config) == TrainConfig{Diffusion}
            target_matrix = config.trajectory[t+1].register.state
        end

        @progress for epoch in 1:config.epoch_schedule[t]
            # Sample a batch from current register and target matrix
            current_reg_batch::Register = reduce(
                hcat,
                sample(
                    eachcol(current_reg.state),
                    config.batch_size;
                    replace=false,
                ),
            )

            target_matrix_batch = reduce(
                hcat,
                sample(
                    eachcol(target_matrix),
                    config.batch_size;
                    replace=false,
                ),
            )

            loss, grads = Zygote.withgradient(current_params) do p
                output_matrix_batch = apply_pqc(
                    ansatz,
                    current_reg_batch,
                    p,
                )

                C = 1.0 .- abs2.(target_matrix_batch' * output_matrix_batch)

                Γ = Zygote.ignore() do
                    return optimal_transport_plan(C; β=0.01, max_iter=500, L=1)
                end

                return dot(Γ, C)
            end

            Optimisers.update!(opt_state, current_params, grads[1])
            loss_history[t][epoch] = loss

            callback(loss)
        end

        current_reg::Register = apply_pqc(
            ansatz,
            current_reg,
            current_params,
        )

        params[:, t] = copy(current_params)
    end

    return loss_history, params
end


export inference

function inference(
    ansatz::A,
    config::TrainConfig,
    distribution::E,
    params::P,
) where {A <: AbstractAnsatz, E <: AbstractDist, P <: AbstractParams}
    trajectory = Vector{AbstractDist}(undef, config.T + 1)
    trajectory[begin] = deepcopy(distribution)
    current_reg = deepcopy(distribution.register)

    for t in 1:config.T
        append_qubits!(current_reg, ansatz.n_ancilla)

        current_reg::Register = apply_pqc(
            ansatz,
            current_reg,
            params[:, t],
        )

        trajectory[t+1] = ArbitraryDist(current_reg)
    end

    return ArbitraryTrajectory(trajectory)
end
