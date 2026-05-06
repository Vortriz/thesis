export plot_bloch_sphere, plot_loss_history, plot_scrambling_decay

function plot_bloch_sphere(ensemble::CTBArrayReg)
	n_samples, dims = ensemble.state.parent.size
	if dims != 2
   	    @info "Plotting on Bloch sphere is only available for 1 qubit system."
		return
	end

    b = Bloch()
	points = zeros(Float64, (3, n_samples))

	for (i, s) in ensemble.state |> eachcol |> enumerate
		s = s[1] * basis(2, 0) + s[2] * basis(2, 1)
		points[:, i] = [expect(sigmax(), s), expect(sigmay(), s), expect(sigmaz(), s)] |> real
	end

    add_points!(b, points)
	b.point_size = [3]
    fig, _ = render(b)

    # To make the plot square and remove axes
    # ax = Axis(fig[1, 1], aspect=1)
	# hidedecorations!(ax)
	# hidespines!(ax)
	# colsize!(fig.layout, 1, Aspect(1, 1.0))
	# resize_to_layout!(fig)

    return fig
end

function plot_loss_history(loss_history::Vector{Vector{Float64}})
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel = "Iterations",
        ylabel = "Loss";
    )
    ax.yticks = 0:0.2:1

    iter_schedule = length.(loss_history)
    x = cumsum(iter_schedule)
    pushfirst!(x, 0)

    for t in 1:length(loss_history)
        lines!(
            ax,
            (x[t]+1):x[t+1],
            loss_history[t]
        )
    end
    ylims!(ax, 0, 1)

    return fig

end

function plot_scrambling_decay(
    arch::ModelArch,
    config::TrainConfig;
    trajectory::Vector{CTBArrayReg},
    metric::Function
)
	rand_ensemble = gen_dist(
		Val(haar);
		n_qubits=arch.n_data,
		n_samples=config.dataset_size,
	)

    distances = Float64[]
    for ensemble in trajectory
        push!(distances, metric(ensemble.state, rand_ensemble.state))
    end

    metric_name = string(nameof(metric))

    fig = Figure()
    ax = Axis(
		fig[1, 1],
		xlabel = "t",
		ylabel = "$metric_name \n (wrt Random Ensemble)",
		title = "Scrambling Decay ($metric_name)"
	)
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.yticks = 0:0.2:1
    ylims!(ax, 0, 1)

    scatter!(ax, 0:config.T, distances)

    return fig
end

# function plot_eval_loss_history(
#     model::Model,
#     strategy::TrainingStrategy,
#     backward_states::OffsetEnsembleCollection
# )
#     fig = Figure()
#     ax = Axis(
#         fig[1, 1],
#         # yscale = log10
#         xlabel = "t",
#         ylabel = "Loss",
#         title = "Eval Loss History ($(strategy.loss_function |> nameof))",
#     )
#     ax.xticks = 0:model.T
#     ax.yticks = 0:0.2:1

#     distances = Vector{Float64}()

#     for ensemble in backward_states |> OffsetArrays.no_offset_view |> eachcol
#         push!(
#             distances,
#             strategy.loss_function(
#                 ensemble |> Ensemble,
#                 model.forward_ensembles[1:100, 0] |> Ensemble,
#             ),
#         )
#     end

#     reverse!(distances)

#     scatter!(ax, (0:model.T), distances)
#     ylims!(ax, 0, 1)

#     @show distances

#     return fig
# end

# function plot_qkr_localization(model::Model, states::Ensemble)
#     n = length(states)
#     dims = 2^model.n_qubits
#     m_vec = [0:dims/2-1; -dims/2:-1]
# 	avg_amplitudes = zeros(dims)
# 	for ϕ in states |> ensemble_to_matrix |> eachcol
# 		amplitudes = abs2.(ϕ)
# 		_, idx = findmax(amplitudes)
# 		circshift!(amplitudes, n-idx+1)
# 		avg_amplitudes += amplitudes
# 	end
# 	avg_amplitudes /= n
# 	fig = Figure()
# 	ax = Axis(fig[1, 1], title="Localization of states", xlabel="m", ylabel="|ψ(p)|²)", yscale=log10)
# 	scatter!(
# 	    ax,
# 		m_vec, avg_amplitudes,
# 		markersize=3,
# 		label=:none,
# 	)
# 	return fig
# end
