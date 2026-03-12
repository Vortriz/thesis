export SPSA

# SPSA
struct SPSA <: StepwiseStrategy
    loss_function::Function
    iter_schedule::Vector{Int}
    hyper_params::NamedTuple
    loss_history::Vector{Vector{Float64}}

    function SPSA(;loss_function, iter_schedule, hyper_params)
        reverse!(iter_schedule) # We train from T down to 1, so reverse schedule for easier indexing
        loss_history = iter_schedule .|> zeros
        new(loss_function, iter_schedule, hyper_params, loss_history)
    end
end

function calc_grad(model, strategy, k, input_reg, target_ensemble, params)
    # As per https://docs.pennylane.ai/en/stable/code/api/pennylane.SPSAOptimizer.html#pennylane.SPSAOptimizer
    Δₖ = rand([-1.0, 1.0], length(params)) # Rademacher distribution
    c = strategy.hyper_params.c
    γ = strategy.hyper_params.γ
    cₖ = c / k^γ

    # Calculate gradients using the current AR input state
    loss_plus = strategy.loss_function(
        target_ensemble |> ensemble_to_matrix,
        denoise(model, strategy, input_reg, params + cₖ * Δₖ)
    )
    loss_minus = strategy.loss_function(
        target_ensemble |> ensemble_to_matrix,
        denoise(model, strategy, input_reg, params - cₖ * Δₖ)
    )

    return (loss_plus - loss_minus) ./ (2 * cₖ * Δₖ)
end

function train_step!(model::Model, strategy::SPSA, t::Int, current_reg::ConcreteBatchedArrayReg)
    params = rand(Float64, (2 * model.n_total * model.n_layers))

    A = strategy.iter_schedule[t] * 0.1
    α = strategy.hyper_params.α
    a = 0.1 * (A + 1)^α

    @progress for k in 1:strategy.iter_schedule[t]
        indices = sample(
            1:model.forward_ensemble_size,
            model.backward_ensemble_size,
            replace = false,
        )
        target_ensemble::Ensemble = model.forward_ensembles[indices, t-1]

        # Calculate gradient using AR input (current_reg)
        grad = calc_grad(model, strategy, k, current_reg, target_ensemble, params)

        aₖ = a / (A + k)^α
        temp = aₖ * grad
        params .-= temp

        # Log loss
        indices_test = sample(
            1:model.forward_ensemble_size,
            model.backward_ensemble_size,
            replace = false,
        )
        target_ensemble_test::Ensemble = model.forward_ensembles[indices_test, t-1]

        denoised_state = denoise(model, strategy, current_reg, params)
        loss = strategy.loss_function(
            target_ensemble_test |> ensemble_to_matrix,
            denoised_state
        )
        strategy.loss_history[t][k] = loss
    end

    model.trained_params[:, t] = params
end
