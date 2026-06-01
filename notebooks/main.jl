### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 7c3a8a85-b2b5-4dc7-9638-7b7d2d6f3a3e
begin
    import Pkg

    # activate the shared project environment
    Pkg.activate(@__DIR__)

    # This file needs to be executed from notebooks/ dir
    cd(@__DIR__)
    base_path = ".."
    Pkg.develop(Pkg.PackageSpec(; path=base_path))
    Pkg.precompile()

    using CUDA
    using GQML
    import Dates
    import Optimisers
    import Logging
    using TensorBoardLogger
end

# ╔═╡ 7479301c-85c0-4773-bc5d-1195c7cb47ad
begin
    const T = 2
    const TB_LOGGING = true

    ansatz = EHA(;
        n_data=1,
        n_ancilla=2,
        n_layers=3,
        measurement=Normal(),
    )

    initial_dist = HaarDist(;
        n_qubits=ansatz.n_data,
        n_samples=5000,
    ) |> cu
    target_dist = CircleDist(;
        n_samples=5000,
    ) |> cu

    config = TrainConfig(
        # Direct([initial_dist, target_dist]);
        Diffusion(
            scramble(;
                n_qubits=ansatz.n_data,
                distribution=target_dist,
                weight_schedule=range(0.5, 4; length=T) |> collect,
            ),
        );
        batch_size=200,
        epoch_schedule=fill(40, T),
    )

    optimizer = Optimisers.AMSGrad(0.005)
end;

# ╔═╡ d5d5bd83-1e77-4d5f-a24c-a0e576fe1eff
begin
    const save_path =
        joinpath(
            base_path,
            "data",
            Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS"),
        ) |> abspath

    if TB_LOGGING == true
        tbl = TBLogger(
            save_path;
            min_level=Logging.Info,
        )
        @info "Saving at $save_path"
        GQML.log_hyperparams(tbl, ansatz, config)
        GQML.log_optim(tbl, optimizer)
    end
end

# ╔═╡ de779e94-a981-4020-a5d8-ef25ba701015
loss_history, params = train(
    ansatz,
    config;
    optimizer=optimizer,
    callback=(loss) -> begin
        if TB_LOGGING == true
            increment_step!(tbl, 1)
            log_value(tbl, "loss", loss)
        end
    end,
)

# ╔═╡ e7010f58-8607-4bd6-97d2-b0a3a668bbb5
loss_history_fig = plot_loss_history(
    loss_history;
    yscale=log10,
)

# ╔═╡ d95bf7db-e4d9-4964-9019-69ac019dd7fb
if TB_LOGGING == true
    save(
        save_path,
        ansatz, config,
        params, loss_history_fig,
    )
end

# ╔═╡ Cell order:
# ╟─7c3a8a85-b2b5-4dc7-9638-7b7d2d6f3a3e
# ╠═7479301c-85c0-4773-bc5d-1195c7cb47ad
# ╠═d5d5bd83-1e77-4d5f-a24c-a0e576fe1eff
# ╠═de779e94-a981-4020-a5d8-ef25ba701015
# ╠═e7010f58-8607-4bd6-97d2-b0a3a668bbb5
# ╠═d95bf7db-e4d9-4964-9019-69ac019dd7fb
