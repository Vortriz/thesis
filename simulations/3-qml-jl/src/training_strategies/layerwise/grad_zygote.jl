export GradZygote

struct GradZygote{F, O} <: StepwiseStrategy
    loss_function::F
    optimizer::O
    iter_schedule::Vector{Int}
    loss_history::Vector{Vector{Float64}}

    function GradZygote(;loss_function::F, optimizer::O, iter_schedule) where {F, O}
        reverse!(iter_schedule) # We train from T down to 1, so reverse schedule for easier indexing
        loss_history = [zeros(Float64, n) for n in iter_schedule]
        new{F, O}(loss_function, optimizer, iter_schedule, loss_history)
    end
end

function train_step!(model::Model, strategy::GradZygote, t::Int, autoregressive_reg::ConcreteBatchedArrayReg)
    n_params = 2 * model.n_total * model.n_layers
    params = rand(model.rng, Float64, n_params)

    opt_state = Optimisers.setup(strategy.optimizer, params)

    n_batch = autoregressive_reg.nbatch
    d = 2^model.n_qubits
    a = 2^model.n_ancilla

    # Pre-calculate column offsets for efficient state gathering (non-differentiable)
    col_offsets = (0:n_batch-1) .* a

    # Pre-join the constant input register with ancillas outside the loop
    in_reg = join(
        autoregressive_reg,
        zero_state(model.n_ancilla; nbatch = n_batch)
    )

    full_target_matrix = model.forward_ensembles[begin:end, 0] |> ensemble_to_matrix

    n_iters = strategy.iter_schedule[t]
    base_lr = strategy.optimizer.eta

    @progress for k in 1:n_iters
        # Cosine Annealing Learning Rate Schedule with η_min
        # We decay to 1% of the base learning rate to prevent freezing
        η_min = 0.01
        lr_scale = η_min + (1 - η_min) * 0.5 * (1 + cos(pi * k / n_iters))
        current_lr = base_lr * lr_scale
        Optimisers.adjust!(opt_state, current_lr)

        indices = sample(
            1:model.forward_ensemble_size,
            model.backward_ensemble_size,
            replace = false,
        )
        # Use view for efficiency
        target_matrix = view(full_target_matrix, :, indices)

        # Unified Pass: Calculate Loss and Gradient simultaneously
        surrogate_loss_val, grads = Zygote.withgradient(params) do p
            # Forward pass (differentiable)
            out_reg_ad = apply(in_reg, dispatch(model.backward_circuit, p))

            # Efficient gather operation: reshape to 2D for fast column slicing
            state_2d = reshape(out_reg_ad.state, d, :)

            # Sample measurement outcomes (non-differentiable)
            outcomes = Zygote.ignore() do
                res = measure(out_reg_ad, (model.n_qubits+1):model.n_total; nshots=1)
                # Ensure outcomes is a 1D vector to avoid 3D array broadcasting (Matrix + Vector)
                vec(Int.(res)) .+ 1
            end

            # Extract the sampled states using pre-calculated column offsets
            # Slicing with a 1D vector for columns ensures gen_sampled is a 2D Matrix (d, n_batch)
            gen_sampled = state_2d[:, outcomes .+ col_offsets]

            # Pre-calculate probabilities for normalization (differentiable)
            # probs[j] = ||gen_sampled_j||^2
            probs = sum(abs2, gen_sampled, dims=1)

            # Weighted fidelity loss: L = -Σ Γ_ij * |<target_i | gen_normalized_j>|^2
            # |<target_i | gen_normalized_j>|^2 = |target_i' * gen_sampled_j|^2 / probs_j
            dot_products = target_matrix' * gen_sampled
            fidelity_matrix = abs2.(dot_products) ./ (probs .+ 1e-12)

            # Calculate Optimal Transport Plan (non-differentiable)
            # Reuse fidelity_matrix to avoid re-calculating C = 1 - F
            Γ = Zygote.ignore() do
                ipot(1.0 .- fidelity_matrix; max_iter=500)
            end

            return -dot(Γ, fidelity_matrix)
        end

        # Parameter Update
        opt_state, params = Optimisers.update!(opt_state, params, grads[1])

        # Logging: W ≈ 1 + surrogate_loss
        strategy.loss_history[t][k] = 1.0 + surrogate_loss_val
    end

    model.trained_params[:, t] = params
end
