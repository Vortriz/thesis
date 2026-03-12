export DirectQNSPSA

struct DirectQNSPSA <: DirectStrategy
    loss_function::Function
    iter_schedule::Vector{Int}
    hyper_params::NamedTuple
    loss_history::Vector{Vector{Float64}}

    function DirectQNSPSA(;
        loss_function,
        n_iters::Int,
        hyper_params,
    )
        loss_history = [zeros(Float64, n_iters)]
        new(loss_function, [n_iters], hyper_params, loss_history)
    end
end

function run_direct_process(model::Model, params_matrix::Matrix{Float64}, initial_reg::ConcreteBatchedArrayReg; fix_outcomes=nothing)
    n_batch = initial_reg.nbatch
    d = 2^model.n_qubits
    a = 2^model.n_ancilla
    col_offsets = (0:n_batch-1) .* a

    current_reg_state = initial_reg.state
    outcomes_history = Vector{Vector{Int}}()

    for t in model.T:-1:1
        batched_reg = ConcreteBatchedArrayReg(current_reg_state, n_batch)
        in_reg = join(batched_reg, zero_state(model.n_ancilla; nbatch = n_batch))
        circuit_t = dispatch(model.backward_circuit, params_matrix[:, t])
        out_reg = apply(in_reg, circuit_t)

        state_2d = reshape(out_reg.state, d, :)

        if fix_outcomes !== nothing
            idx = model.T - t + 1
            outcomes = fix_outcomes[idx]
        else
            res = measure(out_reg, (model.n_qubits+1):model.n_total; nshots=1)
            outcomes = vec(Int.(res)) .+ 1
            push!(outcomes_history, outcomes)
        end

        gen_sampled = state_2d[:, outcomes .+ col_offsets]
        probs = sum(abs2, gen_sampled, dims=1)
        current_reg_state = gen_sampled ./ sqrt.(probs .+ 1e-12)
    end

    return current_reg_state, outcomes_history
end

function get_direct_state_overlap(model::Model, initial_reg::ConcreteBatchedArrayReg, params1::Matrix{Float64}, params2::Matrix{Float64})
    state1, outcomes = run_direct_process(model, params1, initial_reg)
    state2, _ = run_direct_process(model, params2, initial_reg; fix_outcomes=outcomes)

    overlaps = sum(conj.(state1) .* state2, dims=1)
    fidelities = abs2.(overlaps)
    return mean(fidelities)
end

function get_direct_raw_metric_tensor(
    model::Model,
    strategy::DirectQNSPSA,
    input_reg::ConcreteBatchedArrayReg,
    params::Matrix{Float64}
)
    N = length(params)
    ϵ = strategy.hyper_params.ϵ

    Δ1_vec = rand([-1.0, 1.0], N)
    Δ2_vec = rand([-1.0, 1.0], N)

    Δ1 = reshape(Δ1_vec, size(params))
    Δ2 = reshape(Δ2_vec, size(params))

    p_pp = params + ϵ * (Δ1 + Δ2)
    p_p  = params + ϵ * Δ1
    p_mp = params - ϵ * Δ1 + ϵ * Δ2
    p_m  = params - ϵ * Δ1

    tasks = Vector()
    for i in 1:4
        push!(tasks, Threads.@spawn get_direct_state_overlap(model, input_reg, params, [p_pp, p_p, p_mp, p_m][i]))
    end
    o = fetch.(tasks)

    tensor_finite_diff = o[1] - o[2] - o[3] + o[4]

    term = tensor_finite_diff / (8 * ϵ^2)

    outer = Δ1_vec * Δ2_vec' + Δ2_vec * Δ1_vec'

    return -outer * term
end

function train!(model::Model, strategy::DirectQNSPSA)
    @info "Strategy: $(typeof(strategy)) (Direct / Simultaneous Training with QNSPSA)"

    params = rand(model.rng, Float64, size(model.trained_params))
    N = length(params)

    metric_tensor = Matrix{Float64}(I, N, N)
    ϵ = strategy.hyper_params.ϵ
    n_iters = strategy.iter_schedule[1]

    full_target_matrix = model.forward_ensembles[begin:end, 0] |> ensemble_to_matrix

    @progress for k in 1:n_iters
        # 1. Estimate Gradient
        indices = sample(1:model.forward_ensemble_size, model.backward_ensemble_size, replace=false)
        target_matrix = view(full_target_matrix, :, indices)

        initial_reg = generate_rand_ensemble(model.n_qubits, model.backward_ensemble_size) |> ensemble_to_batch

        Δ_vec = rand([-1.0, 1.0], N)
        Δ = reshape(Δ_vec, size(params))

        p_plus = params + ϵ * Δ
        p_minus = params - ϵ * Δ

        d_plus_task = Threads.@spawn run_direct_process(model, p_plus, initial_reg)
        d_minus_task = Threads.@spawn run_direct_process(model, p_minus, initial_reg)

        d_plus_state, _ = fetch(d_plus_task)
        d_minus_state, _ = fetch(d_minus_task)

        l_plus = strategy.loss_function(target_matrix, d_plus_state)
        l_minus = strategy.loss_function(target_matrix, d_minus_state)

        grad_vec = (l_plus - l_minus) / (2 * ϵ) .* Δ_vec

        # 2. Estimate Metric Tensor
        metric_raw = get_direct_raw_metric_tensor(model, strategy, initial_reg, params)

        tensor_avg = (metric_raw + k * metric_tensor) / (k + 1)

        # Robust Regularization
        F = eigen(Symmetric(tensor_avg))
        M_reg = F.vectors * Diagonal(abs.(F.values)) * F.vectors'

        reg_val = strategy.hyper_params.β
        metric_tensor = (M_reg + reg_val * I) / (1 + reg_val)

        # 3. Update Parameters
        update_step_vec = metric_tensor \ (strategy.hyper_params.η * grad_vec)
        update_step = reshape(update_step_vec, size(params))
        params_next = params - update_step

        # Log loss
        indices_val = sample(1:model.forward_ensemble_size, model.backward_ensemble_size, replace=false)
        target_val = view(full_target_matrix, :, indices_val)

        d_curr_task = Threads.@spawn run_direct_process(model, params, initial_reg)
        d_next_task = Threads.@spawn run_direct_process(model, params_next, initial_reg)

        d_curr_state, _ = fetch(d_curr_task)
        d_next_state, _ = fetch(d_next_task)

        loss_curr = strategy.loss_function(target_val, d_curr_state)
        loss_next = strategy.loss_function(target_val, d_next_state)

        hist_start = max(1, k - strategy.hyper_params.history_length + 1)
        history = vcat(strategy.loss_history[1][hist_start:k-1], loss_curr)
        tol = 2 * std(history; corrected=false)

        if loss_curr + tol < loss_next
            # Reject update
            val_loss = loss_curr
        else
            # Accept update
            params .= params_next
            val_loss = loss_next
        end

        strategy.loss_history[1][k] = val_loss
    end

    model.trained_params .= params
end
