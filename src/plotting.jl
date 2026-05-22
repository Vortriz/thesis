export plot_bloch_sphere,
    plot_loss_history, plot_trajectory_convergence, plot_qkr_localization

function plot_bloch_sphere(ensemble::CBArrayReg)
    ensemble = ensemble |> cpu
    dims, n_samples = ensemble.state.size
    if dims != 2
        @info "Plotting on Bloch sphere is only available for 1 qubit system."
        return
    end

    b = Bloch()
    points = zeros(Float64, (3, n_samples))

    for (i, s) in ensemble.state |> eachcol |> enumerate
        s = s[1] * basis(2, 0) + s[2] * basis(2, 1)
        points[:, i] =
            [expect(sigmax(), s), expect(sigmay(), s), expect(sigmaz(), s)] |> real
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

function plot_loss_history(
    loss_history::Vector{Vector{Float64}};
    yscale::Function=identity,
)
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        yscale=yscale,
        xlabel="Iterations",
        ylabel="Loss",
    )

    if yscale == log10
        ylims!(ax, 1e-3, 1)
    else
        ylims!(ax, 0, 1)
    end

    iter_schedule = length.(loss_history)
    x = cumsum(iter_schedule)
    pushfirst!(x, 0)

    for t in eachindex(loss_history)
        lines!(
            ax,
            (x[t]+1):x[t+1],
            loss_history[t],
        )
    end

    final_loss = get_final_training_loss(loss_history)
    hlines!(
        ax,
        final_loss;
        color=:red,
        alpha=0.4,
        linestyle=:dash,
        label="Final Loss = $(round(final_loss, digits=4))",
    )
    axislegend(ax; position=:rt)

    return fig

end

function plot_trajectory_convergence(;
    trajectory::Vector{CBArrayReg},
    target_ensemble::CBArrayReg,
    metric::Function,
    yscale::Function=identity,
    plot_title::String,
)
    distances = Float64[]
    for ensemble in trajectory
        push!(distances, metric(ensemble.state, target_ensemble.state))
    end

    metric_name = string(nameof(metric))

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        yscale=yscale,
        xlabel="t",
        ylabel="$metric_name \n (wrt Target Ensemble)",
        title=plot_title,
    )
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.yticks = 0:0.2:1
    if yscale == log10
        ylims!(ax, 1e-3, 1)
    else
        ylims!(ax, 0, 1)
    end

    scatter!(ax, 0:(length(distances)-1), distances)
    return fig
end

function plot_qkr_localization(ensemble::CBArrayReg)
    ensemble = ensemble |> cpu
    dims, n = size(ensemble.state)
    m_vec = [0:(dims/2-1); (-dims/2):-1]
    avg_amplitudes = zeros(dims)
    for ϕ in ensemble.state |> eachcol
        amplitudes = abs2.(ϕ)
        _, idx = findmax(amplitudes)
        circshift!(amplitudes, dims - idx + 1)
        avg_amplitudes += amplitudes
    end
    avg_amplitudes /= n

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title="Localization of states",
        xlabel="m",
        ylabel="|ψ(p)|²)",
        yscale=log10,
    )
    scatter!(
        ax,
        m_vec, avg_amplitudes;
        markersize=3,
        label=:none,
    )
    return fig
end
