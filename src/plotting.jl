export plot_loss_history

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


export plot_bloch_sphere

function plot_bloch_sphere(ensemble::CBArrayReg; square=false)
    ensemble = ensemble |> cpu
    dims, n_samples = ensemble.state.size
    if dims != 2
        @info "Plotting on Bloch sphere is only available for 1 qubit system."
        return
    end

    b = QT.Bloch()
    points = zeros(Float64, (3, n_samples))

    for (i, s) in ensemble.state |> eachcol |> enumerate
        s = s[1] * basis(2, 0) + s[2] * basis(2, 1)
        points[:, i] =
            [
                expect(sigmax(), s),
                expect(sigmay(), s),
                expect(sigmaz(), s),
            ] |> real
    end

    QT.add_points!(b, points)
    b.point_size = [3]
    fig, _ = QT.render(b)

    # To make the plot square and remove axes
    if square == true
        ax = Axis(fig[1, 1]; aspect=1)
        hidedecorations!(ax)
        hidespines!(ax)
        colsize!(fig.layout, 1, Aspect(1, 1.0))
        resize_to_layout!(fig)
    end

    return fig
end


export plot_trajectory_convergence

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

    metric_name = metric |> nameof |> string

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title=plot_title,
        xlabel=L"t",
        ylabel=rich(
            rich(metric_name; font="mono"),
            "\n (wrt Target Ensemble)",
        ),
        yscale=yscale,
        xgridvisible=false,
        ygridvisible=false,
    )
    if yscale == log10
        ylims!(ax, 1e-3, 1)
        ax.yticks = LogTicks(-3:0)
        ax.yminorticksvisible = true
        ax.yminorticks = IntervalsBetween(5)
    else
        ylims!(ax, 0, 1)
    end

    scatter!(ax, 0:(length(distances)-1), distances)
    hlines!(
        ax,
        distances[end];
        color=:red,
        alpha=0.4,
        linestyle=:dash,
        label="Final Loss = $(round(distances[end], digits=4))",
    )
    axislegend(ax; position=:rt)

    return fig
end


export plot_qkr_localization

function plot_qkr_localization(amplitudes::Vector{Float64})
    dims = length(amplitudes)
    m_vec = [0:(dims/2-1); (-dims/2):-1]

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title="Localization of states",
        xlabel="m",
        ylabel="|ψ(p)|²",
        yscale=log10,
    )
    scatter!(
        ax,
        m_vec, amplitudes;
        markersize=5,
        label=:none,
    )

    return fig
end

function plot_qkr_localization(ensemble::CBArrayReg)
    ensemble = ensemble |> cpu
    dims, n = size(ensemble.state)
    avg_amplitudes = zeros(Float64, dims)
    for ψ in ensemble.state |> eachcol
        amplitudes = get_centered_amplitudes(ψ)
        avg_amplitudes += amplitudes
    end
    avg_amplitudes /= n

    return plot_qkr_localization(avg_amplitudes)
end


export plot_tfim_magnetization_dist

function plot_tfim_magnetization_dist(ensemble::CBArrayReg)
    ensemble = ensemble |> cpu
    magnetization_vals = zeros(Float64, ensemble.nbatch)
    for (i, ψ) in ensemble.state |> eachcol |> enumerate
        magnetization_vals[i] = magnetization(ψ)
    end

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xlabel=L"M",
        ylabel=L"PDF(M)",
        xgridvisible=false,
        ygridvisible=false,
    )
    xlims!(ax, 0, 1)

    density!(
        ax, magnetization_vals;
        alpha=0.75,
    )

    return fig
end
