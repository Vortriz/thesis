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

    final_loss = last(loss_history[end], 10) |> mean
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


export plot_bloch

function _bloch_helper!(
    pos::GridPosition,
    reg::Union{
        BatchedArrayReg{2, ComplexF64, Transpose{ComplexF64, Matrix{ComplexF64}}},
        BatchedArrayReg{2, ComplexF64, Matrix{ComplexF64}},
    },
)
    b = QT.Bloch()
    points = zeros(Float64, (3, length(reg)))

    for (i, s) in reg.state |> eachcol |> enumerate
        ψ = s[1] * QT.basis(2, 0) + s[2] * QT.basis(2, 1)
        points[:, i] =
            [
                QT.expect(QT.sigmax(), ψ),
                QT.expect(QT.sigmay(), ψ),
                QT.expect(QT.sigmaz(), ψ),
            ] |> real
    end

    QT.add_points!(b, points)
    b.point_size = [3]

    QT.render(b; location=pos)
end

function plot_bloch(
    dist::AbstractDist;
    square::Bool=false,
)
    @assert dist.n_qubits == 1 "Plotting on Bloch sphere is only available for 1 qubit system."

    fig = Figure()
    _bloch_helper!(fig[1, 1], dist.register)

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

function plot_bloch(;
    steps::Vector{AbstractDist},
    title::String,
    ref_dist::Union{Nothing, D}=nothing,
    ref_label::Union{Nothing, String}=nothing,
) where {D <: AbstractDist}
    is_ref = !isnothing(ref_dist) && !isnothing(ref_label)

    n_steps = is_ref ? length(steps) + 1 : length(steps)
    n_cols = min(n_steps, 5)
    n_rows = ceil(Int, n_steps / n_cols)
    coords(t) = (t - 1) ÷ n_cols + 1, (t - 1) % n_cols + 1

    fig = Figure(; size=(400 * n_cols, 400 * n_rows + 50))

    for (t, dist) in enumerate(steps)
        row, col = coords(t)
        _bloch_helper!(fig[row, col], dist.register)
        Label(
            fig[row, col, Top()];
            text=L"\textbf{Step $%$(t-1)$}",
            padding=(0, 0, 5, 0),
            fontsize=20,
        )
    end

    if is_ref
        row, col = coords(n_steps)
        _bloch_helper!(fig[row, col], ref_dist.register)
        Label(
            fig[row, col, Top()];
            text=L"\textbf{ %$ref_label }",
            padding=(0, 0, 5, 0),
            fontsize=20,
        )
    end

    Label(fig[0, :], title; fontsize=20, font=:bold, padding=(0, 0, 10, 0))

    return fig
end


function plot(
    ::Type{ClusteredDist};
    steps::Vector{AbstractDist},
    title::String,
    ref_dist::Union{Nothing, D}=nothing,
    ref_label::Union{Nothing, String}=nothing,
) where {D <: AbstractDist}
    distances = Float64[]
    for step in steps
        push!(
            distances,
            mmd_distance(step.register.state, ref_dist.register.state),
        )
    end

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title=title,
        xlabel=L"t",
        ylabel=rich(
            rich("MMD"; font="mono"),
            "\n (wrt $ref_label)",
        ),
        xgridvisible=false,
        ygridvisible=false,
    )
    ylims!(ax, 0, 1)

    scatter!(ax, 0:(length(distances)-1), distances)
    hlines!(
        ax,
        distances[end];
        alpha=0.4, color=:red, linestyle=:dash,
        label="Final Value = $(round(distances[end], digits=4))",
    )
    axislegend(ax; position=:rt)

    return fig
end

function plot(
    ::Type{CircleDist};
    steps::Vector{AbstractDist},
    title::String,
    ref_dist::Union{Nothing, D}=nothing,
    ref_label::Union{Nothing, String}=nothing,
) where {D <: AbstractDist}
    distances = Float64[]
    for step in steps
        push!(
            distances,
            expect(Y, step.register) .|> abs2 |> mean,
        )
    end

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title=title,
        xlabel=L"t",
        ylabel=L"\overline{\left\langle Y \right\rangle^2}",
        xgridvisible=false,
        ygridvisible=false,
    )
    ylims!(ax, 0, 1)

    scatter!(ax, 0:(length(distances)-1), distances)
    hlines!(
        ax,
        distances[end];
        alpha=0.4, color=:red, linestyle=:dash,
        label="Final Value = $(round(distances[end], digits=4))",
    )

    scatter!(
        ax,
        length(distances)-1, expect(Y, ref_dist.register) .|> abs2 |> mean;
        alpha=0.0, strokecolor=:black, strokewidth=2,
        label=ref_label,
    )

    axislegend(ax; position=:rt)

    return fig
end

function _qkr_helper!(
    pos::GridPosition,
    reg::Union{
        BatchedArrayReg{2, ComplexF64, Transpose{ComplexF64, Matrix{ComplexF64}}},
        BatchedArrayReg{2, ComplexF64, Matrix{ComplexF64}},
    },
    title::String,
)
    ax = Axis(
        pos;
        title=title,
        xlabel=L"m",
        ylabel=L"\left| \psi(p) \right|^2",
        yscale=log10,
        yticks=LogTicks(LinearTicks(5)),
        yminorticksvisible=true,
        yminorticks=IntervalsBetween(10),
    )

    dims, n = size(reg.state)
    m_vec = [0:(dims/2-1); (-dims/2):-1]
    avg_amplitudes = zeros(Float64, dims)

    for ψ in reg.state |> eachcol
        amplitudes = get_centered_amplitudes(ψ)
        avg_amplitudes += amplitudes
    end
    avg_amplitudes /= n

    scatter!(
        ax,
        m_vec, avg_amplitudes;
        markersize=5,
        label=:none,
    )
end

function plot(
    dist::QKRLocalizedDist;
    title::String,
)
    fig = Figure()
    _qkr_helper!(fig[1, 1], dist.register, title)

    return fig
end

function plot(
    ::Type{QKRLocalizedDist};
    steps::Vector{AbstractDist},
    title::String,
    ref_dist::Union{Nothing, D}=nothing,
    ref_label::Union{Nothing, String}=nothing,
) where {D <: AbstractDist}
    is_ref = !isnothing(ref_dist) && !isnothing(ref_label)

    n_steps = is_ref ? length(steps) + 1 : length(steps)
    n_cols = min(n_steps, 5)
    n_rows = ceil(Int, n_steps / n_cols)
    coords(t) = (t - 1) ÷ n_cols + 1, (t - 1) % n_cols + 1

    fig = Figure(; size=(400 * n_cols, 400 * n_rows + 50))

    for (t, dist) in enumerate(steps)
        row, col = coords(t)
        _qkr_helper!(fig[row, col], dist.register, L"\textbf{Step $%$(t-1)$}")
    end

    if is_ref
        row, col = coords(n_steps)
        _qkr_helper!(fig[row, col], ref_dist.register, L"\textbf{ %$ref_label }")
    end

    Label(fig[0, :], title; fontsize=20, font=:bold, padding=(0, 0, 10, 0))

    return fig
end

function plot(
    dist::TFIMDist;
    title::String,
)
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title=title,
        xlabel=L"M",
        ylabel=L"PDF(M)",
        xgridvisible=false,
        ygridvisible=false,
    )
    xlims!(ax, 0, 1)

    density!(
        ax,
        magnetization(dist.register);
        alpha=0.5,
    )

    return fig
end

function plot(
    ::Type{TFIMDist};
    steps::Vector{AbstractDist},
    title::String,
    ref_dist::Union{Nothing, D}=nothing,
    ref_label::Union{Nothing, String}=nothing,
) where {D <: AbstractDist}
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title=title,
        xlabel=L"M",
        ylabel=L"PDF(M)",
        xgridvisible=false,
        ygridvisible=false,
    )
    xlims!(ax, 0, 1)

    for (t, step) in enumerate(steps)
        density!(
            ax,
            magnetization(step.register);
            alpha=0.5,
            label=L"t = %$(t-1)",
        )
    end

    if !isnothing(ref_dist) && !isnothing(ref_label)
        density!(
            ax,
            magnetization(ref_dist.register);
            alpha=0.3, strokecolor=:black, strokewidth=2,
            label=ref_label,
        )
    end

    axislegend(ax; position=:lt)

    return fig
end
