struct GradEnzyme{F, O} <: TrainingStrategy
    loss_function::F
    optimizer::O
    iter_schedule::Vector{Int}
    loss_history::Vector{Vector{Float64}}

    function GradEnzyme(;loss_function::F, optimizer::O, iter_schedule) where {F, O}
        reverse!(iter_schedule) # We train from T down to 1, so reverse schedule for easier indexing
        loss_history = [zeros(Float64, n) for n in iter_schedule]
        new{F, O}(loss_function, optimizer, iter_schedule, loss_history)
    end
end

@inline function apply_rx_enzyme!(state::Matrix{ComplexF64}, q::Int, θ::Float64, n_total::Int)
    stride = 1 << (q - 1)
    mask = stride - 1
    c = ComplexF64(cos(0.5 * θ), 0.0)
    s = ComplexF64(0.0, -sin(0.5 * θ))
    n_batch = size(state, 2)
    @inbounds @fastmath for b in 1:n_batch
        @simd ivdep for i in 0:(1 << (n_total - 1)) - 1
            i0 = (i & ~mask) << 1 | (i & mask)
            i1 = i0 | stride
            v0 = state[i0+1, b]
            v1 = state[i1+1, b]
            state[i0+1, b] = c * v0 + s * v1
            state[i1+1, b] = s * v0 + c * v1
        end
    end
end

@inline function apply_ry_enzyme!(state::Matrix{ComplexF64}, q::Int, θ::Float64, n_total::Int)
    stride = 1 << (q - 1)
    mask = stride - 1
    c = ComplexF64(cos(0.5 * θ), 0.0)
    s = ComplexF64(sin(0.5 * θ), 0.0)
    n_batch = size(state, 2)
    @inbounds @fastmath for b in 1:n_batch
        @simd ivdep for i in 0:(1 << (n_total - 1)) - 1
            i0 = (i & ~mask) << 1 | (i & mask)
            i1 = i0 | stride
            v0 = state[i0+1, b]
            v1 = state[i1+1, b]
            state[i0+1, b] = c * v0 - s * v1
            state[i1+1, b] = s * v0 + c * v1
        end
    end
end

@inline function apply_cz_enzyme!(state::Matrix{ComplexF64}, q1::Int, q2::Int, n_total::Int)
    m1 = 1 << (q1 - 1)
    m2 = 1 << (q2 - 1)
    mask = m1 | m2
    n_batch = size(state, 2)
    @inbounds @fastmath for b in 1:n_batch
        @simd ivdep for i in 0:(1 << n_total) - 1
            state[i+1, b] = ifelse((i & mask) == mask, -state[i+1, b], state[i+1, b])
        end
    end
end

@inline function apply_backward_circuit_enzyme!(state::Matrix{ComplexF64}, params::Vector{Float64}, model::Model)
    n_total = model.n_total
    p_idx = 1
    @inbounds for _ in 1:model.n_layers
        for q in 1:n_total
            apply_rx_enzyme!(state, q, params[p_idx], n_total)
            p_idx += 1
            apply_ry_enzyme!(state, q, params[p_idx], n_total)
            p_idx += 1
        end
        for q in 1:2:n_total-1
            apply_cz_enzyme!(state, q, q+1, n_total)
        end
        for q in 2:2:n_total-1
            apply_cz_enzyme!(state, q, q+1, n_total)
        end
    end
end

function manual_measure_ancillas!(outcomes::Vector{Int}, probs_buf::Vector{Float64}, state::Matrix{ComplexF64}, d::Int, a::Int, n_batch::Int)
    @inbounds for b in 1:n_batch
        fill!(probs_buf, 0.0)
        @fastmath for m in 0:a-1
            offset = m * d
            p = 0.0
            @simd for k in 1:d
                s = state[offset + k, b]
                p += real(s)^2 + imag(s)^2
            end
            probs_buf[m+1] = p
        end

        r = rand() * sum(probs_buf)
        cum = 0.0
        outcomes[b] = a
        for m in 1:a
            cum += probs_buf[m]
            if r <= cum
                outcomes[b] = m
                break
            end
        end
    end
end

function denoise(model::Model, strategy::GradEnzyme, input_reg::ConcreteBatchedArrayReg, params::Vector{Float64})
    d = 2^model.n_qubits
    a = 2^model.n_ancilla
    n_batch = input_reg.nbatch

    full_state = zeros(ComplexF64, d * a, n_batch)
    full_state[1:d, :] .= input_reg.state

    apply_backward_circuit_enzyme!(full_state, params, model)

    outcomes = zeros(Int, n_batch)
    probs_buf = zeros(Float64, a)
    manual_measure_ancillas!(outcomes, probs_buf, full_state, d, a, n_batch)

    state_2d = reshape(full_state, d, :)
    gen = state_2d[:, outcomes .+ (0:n_batch-1) .* a]
    gen ./= (sqrt.(sum(abs2, gen, dims=1)) .+ 1e-12)

    return gen
end

function surrogate_loss_kernel(
    params::Vector{Float64},
    model::Model,
    in_matrix::Matrix{ComplexF64},
    full_target_matrix::Matrix{ComplexF64}, # PRE-OPTIMIZATION: Pass full matrix
    target_indices::Vector{Int},            # PRE-OPTIMIZATION: Pass indices
    Γ::Matrix{Float64},
    outcomes::Vector{Int},
    n_batch::Int,
    full_state::Matrix{ComplexF64}
)
    d = 2^model.n_qubits
    a = 2^model.n_ancilla

    @inbounds @fastmath for b in 1:n_batch
        @simd ivdep for i in 1:d
            full_state[i, b] = in_matrix[i, b]
        end
        @simd ivdep for i in d+1:(d*a)
            full_state[i, b] = zero(ComplexF64)
        end
    end

    apply_backward_circuit_enzyme!(full_state, params, model)

    n_targets = length(target_indices)
    loss = 0.0

    @inbounds @fastmath for j in 1:n_batch
        ancilla_offset = (outcomes[j] - 1) * d

        branch_norm_sq = 1e-24
        @simd for k in 1:d
            s = full_state[ancilla_offset + k, j]
            branch_norm_sq += real(s)^2 + imag(s)^2
        end

        for i in 1:n_targets
            target_idx = target_indices[i]
            re = 0.0
            im_p = 0.0
            @simd for k in 1:d
                # Access full matrix with index to avoid slice allocation
                t = full_target_matrix[k, target_idx]
                g = full_state[ancilla_offset + k, j]
                re += real(t)*real(g) + imag(t)*imag(g)
                im_p += real(t)*imag(g) - imag(t)*real(g)
            end

            fidelity = (re*re + im_p*im_p) / branch_norm_sq
            loss -= Γ[i, j] * fidelity
        end
    end

    return loss
end

function train_step!(model::Model, strategy::GradEnzyme, t::Int, autoregressive_reg::ConcreteBatchedArrayReg)
    n_params = 2 * model.n_total * model.n_layers
    params = rand(model.rng, Float64, n_params)
    grad_params = zeros(Float64, n_params)

    opt_state = Optimisers.setup(strategy.optimizer, params)

    n_batch = autoregressive_reg.nbatch
    input_matrix = autoregressive_reg.state

    d = 2^model.n_qubits
    a = 2^model.n_ancilla

    # PRE-OPTIMIZATION: Convert target ensemble once
    full_target_matrix = model.forward_ensembles[begin:end, t-1] |> ensemble_to_matrix

    forward_full_state = zeros(ComplexF64, d * a, n_batch)
    backward_full_state = zeros(ComplexF64, d * a, n_batch)
    d_backward_full_state = zeros(ComplexF64, d * a, n_batch)
    current_gen = zeros(ComplexF64, d, n_batch)

    # Pre-allocate for measurement
    outcomes = zeros(Int, n_batch)
    probs_buf = zeros(Float64, a)

    @progress for k in 1:strategy.iter_schedule[t]
        indices = sample(
            1:model.forward_ensemble_size,
            model.backward_ensemble_size,
            replace = false,
        )
        # Avoid slice allocation for OT by using view if possible,
        # but OT solver might not like views. For now, slice is better than hcat.
        target_matrix_slice = full_target_matrix[:, indices]

        # 1. Forward Pass (Zero-Allocation Logic)
        fill!(forward_full_state, zero(ComplexF64))
        @inbounds for b in 1:n_batch
            for i in 1:d
                forward_full_state[i, b] = input_matrix[i, b]
            end
        end

        apply_backward_circuit_enzyme!(forward_full_state, params, model)

        manual_measure_ancillas!(outcomes, probs_buf, forward_full_state, d, a, n_batch)

        @inbounds @fastmath for b in 1:n_batch
            offset = (outcomes[b] - 1) * d
            norm_sq = 1e-24
            @simd for i in 1:d
                val = forward_full_state[offset + i, b]
                current_gen[i, b] = val
                norm_sq += real(val)^2 + imag(val)^2
            end
            nrm = sqrt(norm_sq)
            @simd for i in 1:d
                current_gen[i, b] /= nrm
            end
        end

        Γ = wasserstein_distance(target_matrix_slice, current_gen; return_map = true)

        # 2. Backward Pass: Enzyme Autodiff (Zero-Allocation pass)
        fill!(grad_params, 0.0)
        fill!(d_backward_full_state, zero(ComplexF64))

        result = Enzyme.autodiff(
            Enzyme.set_runtime_activity(ReverseWithPrimal),
            surrogate_loss_kernel,
            Active,
            Enzyme.Duplicated(params, grad_params),
            Enzyme.Const(model),
            Enzyme.Const(input_matrix),
            Enzyme.Const(full_target_matrix), # PRE-OPTIMIZATION
            Enzyme.Const(indices),            # PRE-OPTIMIZATION
            Enzyme.Const(Γ),
            Enzyme.Const(outcomes),
            Enzyme.Const(n_batch),
            Enzyme.Duplicated(backward_full_state, d_backward_full_state)
        )
        surrogate_loss_val = result[2]

        # 3. Parameter Update
        opt_state, params = Optimisers.update!(opt_state, params, grad_params)

        # Logging
        strategy.loss_history[t][k] = 1.0 + surrogate_loss_val
    end

    model.trained_params[:, t] = params
end
