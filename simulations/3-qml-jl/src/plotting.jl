export plot_bloch_sphere, plot_forward_fidelity_decay, plot_training_loss_history, plot_eval_loss_history

function plot_bloch_sphere(ensemble::Union{OffsetEnsemble, Ensemble})
    if ensemble[begin].state.size[1] != 2
        return
    end

    b = Bloch()
    points = reduce(
        hcat,
        ensemble .|>
        state .|>
        s ->
            s[1] * basis(2, 0) + s[2] * basis(2, 1) |>
            s -> [expect(sigmax(), s), expect(sigmay(), s), expect(sigmaz(), s)] |> real,
    )
    add_points!(b, points)
	# b.point_size = [10]
    fig, _ = render(b)

    return fig
end

function plot_forward_fidelity_decay(model::Model)
	rand_ensemble = generate_rand_ensemble(model.n_qubits, model.forward_ensemble_size)

    for t in eachindex(model.forward_fidelity_decay)
        model.forward_fidelity_decay[t] = mmd_distance(
			model.forward_ensembles[:, t] |> OffsetArrays.no_offset_view,
			rand_ensemble,
		)
    end

    fig = Figure()
    ax = Axis(
		fig[1, 1],
		# yscale = log10,
		xlabel = "t",
		ylabel = "MMD Distance \n (wrt Random Ensemble)",
		title = "Forward fidelity decay (MMD)"
	)
    ax.xticks = 0:model.T
    ax.yticks = 0:0.2:1
    ylims!(ax, 0, 1)

    scatter!(ax, model.forward_fidelity_decay)

    return fig
end

function plot_training_loss_history(model::Model, strategy::DirectStrategy)
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel = "Iterations",
        ylabel = "Loss",
        title = "Direct training loss history ($(strategy.loss_function |> nameof))",
    )
    ax.yticks = 0:0.2:1

    lines!(
        ax,
        1:strategy.iter_schedule[1],
        strategy.loss_history[1]
    )
    ylims!(ax, 0, 1)

    return fig
end

function plot_training_loss_history(model::Model, strategy::StepwiseStrategy)
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        # yscale = log10
        xlabel = "Iterations",
        ylabel = "Loss",
        title = "Backward process loss history ($(strategy.loss_function |> nameof))",
    )
    ax.yticks = 0:0.2:1

    # Slice to 1:model.T to gracefully handle when the user provides an iter_schedule longer than T
    actual_loss_history = strategy.loss_history[1:model.T]
    actual_iter_schedule = strategy.iter_schedule[1:model.T]

	loss_hist = actual_loss_history |> reverse
	x = actual_iter_schedule |> reverse |> cumsum
	pushfirst!(x, 0)

    for t in 1:model.T
		lines!(
			ax,
			(x[t]+1):x[t+1],
			loss_hist[t]
		)
    end
    ylims!(ax, 0, 1)

    return fig
end

function plot_eval_loss_history(
    model::Model,
    strategy::TrainingStrategy,
    backward_states::OffsetEnsembleCollection
)
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        # yscale = log10
        xlabel = "t",
        ylabel = "Loss",
        title = "Eval Loss History ($(strategy.loss_function |> nameof))",
    )
    ax.xticks = 0:model.T
    ax.yticks = 0:0.2:1

    distances = Vector{Float64}()

    for ensemble in backward_states |> OffsetArrays.no_offset_view |> eachcol
        push!(
            distances,
            strategy.loss_function(
                ensemble |> Ensemble,
                model.forward_ensembles[1:100, 0] |> Ensemble,
            ),
        )
    end

    reverse!(distances)

    scatter!(ax, (0:model.T), distances)
    ylims!(ax, 0, 1)

    @show distances

    return fig
end
