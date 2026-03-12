export DirectGradZygote

struct DirectGradZygote{F, O} <: DirectStrategy
    loss_function::F
    optimizer::O
    iter_schedule::Vector{Int}
    loss_history::Vector{Vector{Float64}}

    function DirectGradZygote(;loss_function::F, optimizer::O, n_iters::Int) where {F, O}
        loss_history = [zeros(Float64, n_iters)]
        new{F, O}(loss_function, optimizer, [n_iters], loss_history)
    end
end

function train!(model::Model, strategy::DirectGradZygote)
    @info "Strategy: $(typeof(strategy)) (Direct / Simultaneous Training)"

    # The parameters are stored in `model.trained_params` which is of size (n_params_per_step, T).
    params = rand(model.rng, Float64, size(model.trained_params))

    opt_state = Optimisers.setup(strategy.optimizer, params)

    n_batch = model.backward_ensemble_size
    d = 2^model.n_qubits
    a = 2^model.n_ancilla

    # Pre-calculate column offsets for efficient state gathering (non-differentiable)
    col_offsets = (0:n_batch-1) .* a

    full_target_matrix = model.forward_ensembles[begin:end, 0] |> ensemble_to_matrix

    n_iters = strategy.iter_schedule[1]

    @progress for k in 1:n_iters
        indices = sample(
            1:model.forward_ensemble_size,
            model.backward_ensemble_size,
            replace = false,
        )
        target_matrix = view(full_target_matrix, :, indices)

        # Generate fresh initial states at t=T for each training iteration
        initial_reg = generate_rand_ensemble(model.n_qubits, model.backward_ensemble_size) |> ensemble_to_batch

        surrogate_loss_val, grads = Zygote.withgradient(params) do p
            current_reg_state = initial_reg.state

            # Forward pass through all time steps T down to 1
            for t in model.T:-1:1
                batched_reg = ConcreteBatchedArrayReg(current_reg_state, n_batch)

                in_reg = join(
                    batched_reg,
                    zero_state(model.n_ancilla; nbatch = n_batch)
                )

                circuit_t = dispatch(model.backward_circuit, p[:, t])
                out_reg_ad = apply(in_reg, circuit_t)

                state_2d_sliceable = reshape(out_reg_ad.state, d, :)

                outcomes = Zygote.ignore() do
                    res = measure(out_reg_ad, (model.n_qubits+1):model.n_total; nshots=1)
                    vec(Int.(res)) .+ 1
                end

                gen_sampled = state_2d_sliceable[:, outcomes .+ col_offsets]

                probs = sum(abs2, gen_sampled, dims=1)

                # Normalize states for the next iteration (or final output)
                current_reg_state = gen_sampled ./ sqrt.(probs .+ 1e-12)
            end

            # Calculate the fidelity between target_matrix and final current_reg_state (t=0)
            dot_products = target_matrix' * current_reg_state
            fidelity_matrix = abs2.(dot_products)

            Γ = Zygote.ignore() do
                ipot(1.0 .- fidelity_matrix; max_iter=500)
            end

            return -dot(Γ, fidelity_matrix)
        end

        opt_state, params = Optimisers.update!(opt_state, params, grads[1])
        strategy.loss_history[1][k] = 1.0 + surrogate_loss_val
    end

    model.trained_params .= params
end
