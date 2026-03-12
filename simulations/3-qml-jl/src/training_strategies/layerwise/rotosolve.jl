export Rotosolve

struct Rotosolve <: StepwiseStrategy
    loss_function::Function
    iter_schedule::Vector{Int}
    loss_history::Vector{Vector{Float64}}

    # Buffers
    target_buffer::Matrix{ComplexF64}

    function Rotosolve(model; loss_function, iter_schedule)
        loss_history = iter_schedule .|> zeros
        # Pre-allocate buffer for target states
        target_buffer = zeros(ComplexF64, (2^model.n_qubits, model.backward_ensemble_size))
        new(loss_function, iter_schedule, loss_history, target_buffer)
    end
end

function train_step!(model::Model, strategy::Rotosolve, t::Int, current_reg::ConcreteBatchedArrayReg)
    params = rand(Float64, (2 * model.n_total * model.n_layers))

    for iter in 1:strategy.iter_schedule[t]
        for pᵢ in eachindex(params)
            # Sample target batch once for this parameter update
            indices = sample(
                1:model.forward_ensemble_size,
                model.backward_ensemble_size,
                replace = false,
            )
            target_ensemble::Ensemble = model.forward_ensembles[indices, 0]

            # OPTIMIZATION: In-place update of target buffer to avoid allocation
            ensemble_to_matrix!(strategy.target_buffer, target_ensemble)

            # Parallel evaluation for 0, pi/2, -pi/2
            tasks = map([0.0, pi/2, -pi/2]) do shift
                Threads.@spawn begin
                    temp_params = copy(params)
                    temp_params[pᵢ] = shift

                    # Denoise using AR input
                    d_matrix = denoise(model, strategy, current_reg, temp_params)

                    # Compute loss against the buffered target matrix
                    strategy.loss_function(strategy.target_buffer, d_matrix)
                end
            end

            losses = fetch.(tasks)
            l₀, l₊, l₋ = losses[1], losses[2], losses[3]

            params[pᵢ] = -pi/2 - atan(2*l₀ - l₊ - l₋, l₊ - l₋)
        end

        # Logging loss
        indices = sample(
            1:model.forward_ensemble_size,
            model.backward_ensemble_size,
            replace = false,
        )
        target_ensemble = model.forward_ensembles[indices, t-1]

        # Use buffer for logging too
        ensemble_to_matrix!(strategy.target_buffer, target_ensemble)

        denoised_matrix = denoise(model, strategy, current_reg, params)
        loss = strategy.loss_function(
            strategy.target_buffer,
            denoised_matrix
        )
        strategy.loss_history[t][iter] = loss
    end

    model.trained_params[:, t] = params
end
