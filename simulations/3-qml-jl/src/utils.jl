export ensemble_to_matrix, ensemble_to_matrix!, ensemble_to_batch, matrix_to_ensemble, generate_rand_ensemble, initialize_backward_ensemble, intrange, record_run, save_weights, load_weights, generate_backward_pass

function matrix_to_ensemble(matrix::Matrix{ComplexF64})::Ensemble
    ensemble_size = size(matrix, 2)
    ensemble = Ensemble(undef, ensemble_size)
    for i in 1:ensemble_size
        ensemble[i] = ConcreteArrayReg(matrix[:, i] |> v -> reshape(v, :, 1))
    end
    return ensemble
end

function save_weights(model::Model, path::String; old_weights::Union{Nothing, Matrix{Float64}} = nothing)
    weights_to_save = if isnothing(old_weights)
        model.trained_params
    else
        # Concatenate: new weights (T_new) and then old weights (T_old)
        hcat(model.trained_params, old_weights)
    end

    # Save both weights and the target ensemble (at t=0)
    target_ensemble = model.forward_ensembles[:, 0]

    jldsave(path; trained_params = weights_to_save, target_ensemble = target_ensemble)
    @info "Weights and target ensemble saved to $path (Total T = $(size(weights_to_save, 2)))"
end

function load_weights(path::String)
    file = jldopen(path, "r")
    weights = file["trained_params"]
    target_ensemble = file["target_ensemble"]
    close(file)
    @info "Weights and target loaded from $path (T = $(size(weights, 2)))"
    return weights, target_ensemble
end

function generate_backward_pass(model::Model, strategy::TrainingStrategy, weights::Matrix{Float64})
    T_weights = size(weights, 2)
    @info "Running full backward pass over loaded weights (T=$T_weights to 1)"
    
    current_reg::ConcreteBatchedArrayReg =
        generate_rand_ensemble(model.n_qubits, model.backward_ensemble_size) |> ensemble_to_batch

    @progress for t in T_weights:-1:1
        denoised_matrix = denoise(model, strategy, current_reg, weights[:, t])
        current_reg = ConcreteBatchedArrayReg(denoised_matrix, size(denoised_matrix, 2))
    end
    return current_reg
end

function Base.Matrix(ensemble::Ensemble)::Matrix{ComplexF64}
    n_qubits = size(ensemble[begin].state, 1)
    ensemble_size = length(ensemble)

    ensemble_state = zeros(ComplexF64, (n_qubits, ensemble_size))
    ensemble_to_matrix!(ensemble_state, ensemble)
    return ensemble_state
end

function ensemble_to_matrix(ensemble::Ensemble)::Matrix{ComplexF64}
    return reduce(hcat, ensemble .|> state)
end

function ensemble_to_matrix!(
    dest::Matrix{ComplexF64},
    ensemble::Ensemble
)
	 for (i, reg) in enumerate(ensemble)
        view(dest, :, i) .= reg.state
    end
end

function ensemble_to_batch(ensemble::Ensemble)::ConcreteBatchedArrayReg
    return (
        ensemble |>
        ensemble_to_matrix |>
        matrix -> ConcreteBatchedArrayReg(matrix, size(matrix, 2))
    )
end

function generate_rand_ensemble(n_qubits::Int64, ensemble_size::Int64)::Ensemble
	haar_ensemble = Ensemble()
	for _ in 1:ensemble_size
		push!(haar_ensemble, rand_state(n_qubits))
	end
	return haar_ensemble
end

function intrange(start::Int64, stop::Int64; length::Int64)
    return range(start, stop; length=length) .|> x -> round(Int64, x)
end

get_optimizer_name(strategy::TrainingStrategy) = hasproperty(strategy, :optimizer) ? string(nameof(typeof(strategy.optimizer))) : string(nameof(typeof(strategy)))

function record_run(model, strategy, training_plot, target; save_dir_base = "saves", old_weights::Union{Nothing, Matrix{Float64}} = nothing)
    # 1. Create a unique folder inside save_dir_base/
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS_sss")
    opt_name = get_optimizer_name(strategy)
    save_dir = joinpath(save_dir_base, "$(timestamp)_$(opt_name)")
    mkpath(save_dir)

    # 2. Calculate the approximate final loss (avg of last 50 iterations of final step)
    final_loss = mean(strategy.loss_history[1][max(1, end-50):end])

    # 3. Save Hyperparameters and Final Loss to a summary file
    open(joinpath(save_dir, "summary.txt"), "w") do f
        println(f, "--- Run Result ---")
        println(f, "Final Loss: ", round(final_loss, digits=6))

        println(f, "\n--- Hyperparameters ---")
        if hasproperty(strategy, :optimizer)
            println(f, "Optimizer: ", strategy.optimizer)
        end
        if hasproperty(strategy, :hyper_params)
            println(f, "HyperParams: ", strategy.hyper_params)
        end
        println(f, "n_qubits:  ", model.n_qubits)
        println(f, "n_layers:  ", model.n_layers)
        println(f, "n_ancilla: ", model.n_ancilla)
        println(f, "T:         ", model.T)
        println(f, "Schedule:  ", strategy.iter_schedule)
        println(f, "Target:    ", target)
    end

    # 4. Save the Plot
    save(joinpath(save_dir, "training_plot.png"), training_plot)

    # 5. Save Weights
    save_weights(model, joinpath(save_dir, "weights.jld2"); old_weights = old_weights)

    println("Recorded in: ", save_dir)
    println("Final Loss: ", round(final_loss, digits=4))
end
