# QNSPSA
struct QNSPSA <: TrainingStrategy
    loss_function::Function
    iter_schedule::Vector{Int}
    hyper_params::NamedTuple
    loss_history::Vector{Vector{Float64}}

    # learning_rate::Float64
    # finite_diff_step::Float64
    # regularization::Float64
    # resamplings::Int

    function QNSPSA(;
        loss_function,
        iter_schedule,
        hyper_params,
        # η=1e-2,
        # ϵ=1e-2,
        # β=1e-3,
        # resamplings=1,
    )
        reverse!(iter_schedule) # We train from T down to 1, so reverse schedule for easier indexing
        loss_history = iter_schedule .|> zeros
        new(loss_function, iter_schedule, hyper_params, loss_history)
    end
end

function denoise(model::Model, strategy::QNSPSA, input_reg::ConcreteBatchedArrayReg, params::Vector{Float64})
    # Join input register (from AR loop) with ancillas
    input_with_ancilla::ConcreteBatchedArrayReg = join(
        input_reg,
        zero_state(model.n_ancilla; nbatch = input_reg.nbatch),
    )

    circuit = dispatch(model.backward_circuit, params)
    apply!(input_with_ancilla, circuit)

    # Measurement collapse
    measure!(RemoveMeasured(), input_with_ancilla, (model.n_qubits+1):model.n_total)

    return input_with_ancilla.state
end

# Helper to get state before measurement for Metric Tensor calculation
function get_unitary_output(model::Model, input_reg::ConcreteBatchedArrayReg, params::Vector{Float64})
    reg = join(
        input_reg,
        zero_state(model.n_ancilla; nbatch = input_reg.nbatch),
    )
    circuit = dispatch(model.backward_circuit, params)
    apply!(reg, circuit)
    return reg
end

function get_perturbation(n_params::Int)
    return rand((-1.0, 1.0), n_params)
end

function compute_overlap(reg1::ConcreteBatchedArrayReg, reg2::ConcreteBatchedArrayReg)
    # Compute fidelity per batch item: |<psi1|psi2>|^2
    # state is Matrix (dim x batch)
    # We want dot product of columns

    # dot(A, B) computes sum(conj(A) .* B).
    # But we want column-wise dot products.
    # sum(conj(S1) .* S2, dims=1)

    overlaps = sum(conj.(reg1.state) .* reg2.state, dims=1)
    fidelities = abs2.(overlaps)
    return mean(fidelities)
end

function get_state_overlap(model::Model, input_reg::ConcreteBatchedArrayReg, params1::Vector{Float64}, params2::Vector{Float64})
    # Note: We reuse input_reg. Since apply! is in-place, we must be careful.
    # get_unitary_output creates a NEW register via join. So input_reg is safe.
    reg1 = get_unitary_output(model, input_reg, params1)
    reg2 = get_unitary_output(model, input_reg, params2)
    return compute_overlap(reg1, reg2)
end

function get_raw_metric_tensor(
    model::Model,
    strategy::QNSPSA,
    input_reg::ConcreteBatchedArrayReg,
    params::Vector{Float64}
)
    n_params = length(params)
    ϵ = strategy.hyper_params.ϵ

    Δ1 = get_perturbation(n_params)
    Δ2 = get_perturbation(n_params)

    # 4 evaluations
    p_pp = params + ϵ * (Δ1 + Δ2)
    p_p  = params + ϵ * Δ1
    p_mp = params - ϵ * Δ1 + ϵ * Δ2
    p_m  = params - ϵ * Δ1

    # Overlaps w.r.t base params is not needed?
    # Notebook formula:
    # overlap(p + e(d1+d2)) - overlap(p + e d1) - overlap(p + e(-d1+d2)) + overlap(p - e d1)
    # All overlaps are implicitly w.r.t a reference state?
    # In the notebook: get_overlap_tape(qnode, params1, params2)
    # It calculates |<psi(p1)|psi(p2)>|^2.
    # Here the reference point is crucial.
    # The paper (Gacon et al) formula (8) defines:
    # g_ij = ...
    # The approximation uses:
    # 1/2 (F(th, th+d1+d2) - F(th, th+d1) - F(th, th-d1+d2) + F(th, th-d1)) ?
    # Let's re-read notebook carefully.
    # "tapes = [get_overlap_tape(..., params_curr, params_curr + ...), ...]"
    # So reference is always params_curr!

    o1 = get_state_overlap(model, input_reg, params, p_pp)
    o2 = get_state_overlap(model, input_reg, params, p_p)
    o3 = get_state_overlap(model, input_reg, params, p_mp)
    o4 = get_state_overlap(model, input_reg, params, p_m)

    tensor_finite_diff = o1 - o2 - o3 + o4

    # metric = - (d1*d2^T + d2*d1^T) * diff / (8 * eps^2)
    term = tensor_finite_diff / (8 * ϵ^2)

    # Outer products
    # Δ1 * Δ2' is a matrix
    outer = Δ1 * Δ2' + Δ2 * Δ1'

    return -outer * term
end

function train_step!(model::Model, strategy::QNSPSA, t::Int, current_reg::ConcreteBatchedArrayReg)
    n_params = 2 * model.n_total * model.n_layers
    params = rand(Float64, n_params) # Initialize random params

    # Initialize metric tensor as Identity
    metric_tensor = Matrix{Float64}(I, n_params, n_params)

    # Hyperparameters
    ϵ = strategy.hyper_params.ϵ

    for k in 1:strategy.iter_schedule[t]
        # 1. Estimate Gradient (Standard SPSA)
        indices = sample(1:model.forward_ensemble_size, model.backward_ensemble_size, replace=false)
        target_ensemble = model.forward_ensembles[indices, t-1]
        target_matrix = target_ensemble |> ensemble_to_matrix

        Δ = get_perturbation(n_params)

        # We assume loss is computed on current_reg (AR input)
        # Note: In standard SPSA we might resample input/target for robustness
        # But for gradient calc, we should use same batch for +/- to reduce variance

        p_plus = params + ϵ * Δ
        p_minus = params - ϵ * Δ

        d_plus = denoise(model, strategy, current_reg, p_plus)
        d_minus = denoise(model, strategy, current_reg, p_minus)

        l_plus = strategy.loss_function(target_matrix, d_plus)
        l_minus = strategy.loss_function(target_matrix, d_minus)

        grad = (l_plus - l_minus) / (2 * ϵ) .* Δ

        # 2. Estimate Metric Tensor
        # Note: We can use a different batch or same batch. Notebook re-samples direction.
        # We reuse current_reg for state definition.

        metric_raw = get_raw_metric_tensor(model, strategy, current_reg, params)

        # Update running average. Note: The notebook accumulates the REGULARIZED tensor.
        tensor_avg = (metric_raw + k * metric_tensor) / (k + 1)

        # Robust Regularization (Equation 11/Sqrtm trick)
        # Instead of `real(sqrt(M * M))` which can fail or produce NaNs due to
        # floating point asymmetry, we use eigen decomposition of a forced Symmetric matrix.
        # Since M_reg = sqrt(M^2), M_reg = U * |Λ| * U' where M = U * Λ * U'
        F = eigen(Symmetric(tensor_avg))
        M_reg = F.vectors * Diagonal(abs.(F.values)) * F.vectors'

        # Normalize/Regularize
        reg_val = strategy.hyper_params.β
        metric_tensor = (M_reg + reg_val * I) / (1 + reg_val)

        # 3. Update Parameters
        # Solve (g + lambda I) d = -lr * grad ?
        # Or just: p_new = p - lr * inv(g) * grad
        # Notebook: solve(metric, -lr * grad + metric * params) => new_params
        # equivalent to: metric * new = -lr * grad + metric * old
        # metric * (new - old) = -lr * grad
        # new - old = -lr * inv(metric) * grad
        # new = old - lr * inv(metric) * grad
        # Yes.

        update_step = metric_tensor \ (strategy.hyper_params.η * grad)
        params_next = params - update_step

        # Log loss (at current params)
        # We can reuse l_plus/l_minus estimate or recompute. Recomputing is safer/standard.
        # Use new random target batch for validation
        indices_val = sample(1:model.forward_ensemble_size, model.backward_ensemble_size, replace=false)
        target_val = model.forward_ensembles[indices_val, t-1] |> ensemble_to_matrix

        d_curr = denoise(model, strategy, current_reg, params)
        loss_curr = strategy.loss_function(target_val, d_curr)

        d_next = denoise(model, strategy, current_reg, params_next)
        loss_next = strategy.loss_function(target_val, d_next)

        hist_start = max(1, k - strategy.hyper_params.history_length + 1)
        history = vcat(strategy.loss_history[t][hist_start:k-1], loss_curr)
        tol = 2 * std(history; corrected=false)

        if loss_curr + tol < loss_next
            # Reject update
            val_loss = loss_curr
        else
            # Accept update
            params .= params_next
            val_loss = loss_next
        end

        strategy.loss_history[t][k] = val_loss
    end

    model.trained_params[:, t] = params
end
