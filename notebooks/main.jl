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

    using PlutoLinks
end

# ╔═╡ e362cf46-77d8-429e-9369-d58630c3eb32
begin
    using CUDA
    @revise using GQML
    import Dates
    import Optimisers
    import Logging
    import CairoMakie
    using TensorBoardLogger
end

# ╔═╡ 70a09975-f011-4ad3-a39c-b2b999efa363
html"""
<style>
main {
    max-width: 1000px
}
</style>
"""

# ╔═╡ 7479301c-85c0-4773-bc5d-1195c7cb47ad
begin
    const T = 7
    const TB_LOGGING = true

    ansatz = EHA(;
        n_data=9,
        n_ancilla=5,
        n_layers=12,
        measurement=Normal(),
    )

    initial_dist = HaarDist(;
        n_qubits=ansatz.n_data,
        n_samples=5000,
    ) |> cu
    target_dist =
        TFIMDist(;
            n_qubits=ansatz.n_data,
            g=range(0.2, 0.4; length=5000) |> collect,
        ) |> cu

    config = TrainConfig(
        Direct(;
            initial_dist=initial_dist,
            target_dist=target_dist,
        );
        # Diffusion(;
        #     target_dist=target_dist,
        #     weight_schedule=Base.LinRange(0.75, 5.5, T) |> collect,
        # );
        batch_size=400,
        epoch_schedule=vcat(fill(400, 3), fill(600, T - 3)),
    )

    initial_params = RandParams(ansatz, config.T)
    # initial_params = IdentityParams(ansatz, config.T)
    optimizer = Optimisers.AMSGrad(0.03)
    plots = Dict{String, CairoMakie.Figure}()
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
        GQML.log_hyperparams(tbl, ansatz, config, initial_params)
        GQML.log_optim(tbl, optimizer)
    end
end

# ╔═╡ 2df9ad2c-2b71-4f18-a6c9-764638df4307
if ansatz.n_data == 1
    plots["target_bloch"] = GQML.plot_bloch(;
        steps=(config.trajectory.steps |> reverse),
        title="Diffusion Trajectory",
        ref_dist=initial_dist,
        ref_label="Haar Distribution",
    )
end

# ╔═╡ 54eba34c-2f07-49ea-af4f-bd8c64cd4994
if typeof(config.trajectory) == Diffusion
    plots["diffusion_trajectory"] = GQML.plot(
        typeof(target_dist);
        steps=(config.trajectory.steps |> reverse),
        title="Diffusion trajectory",
        ref_dist=initial_dist,
        ref_label="Haar Distribution",
    )
end

# ╔═╡ de779e94-a981-4020-a5d8-ef25ba701015
loss_history, params = train(
    ansatz,
    config;
    params=initial_params.params,
    optimizer=optimizer,
    callback=(loss) -> begin
        if TB_LOGGING == true
            increment_step!(tbl, 1)
            log_value(tbl, "loss", loss)
        end
    end,
)

# ╔═╡ e7010f58-8607-4bd6-97d2-b0a3a668bbb5
plots["loss_history_fig"] = plot_loss_history(
    loss_history;
    yscale=log10,
)

# ╔═╡ d95bf7db-e4d9-4964-9019-69ac019dd7fb
if TB_LOGGING == true
    GQML.save(
        save_path,
        ansatz, config, params,
    )
    GQML.save(
        save_path,
        plots,
    )
end

# ╔═╡ Cell order:
# ╟─7c3a8a85-b2b5-4dc7-9638-7b7d2d6f3a3e
# ╟─70a09975-f011-4ad3-a39c-b2b999efa363
# ╠═e362cf46-77d8-429e-9369-d58630c3eb32
# ╠═7479301c-85c0-4773-bc5d-1195c7cb47ad
# ╠═d5d5bd83-1e77-4d5f-a24c-a0e576fe1eff
# ╠═2df9ad2c-2b71-4f18-a6c9-764638df4307
# ╠═54eba34c-2f07-49ea-af4f-bd8c64cd4994
# ╠═de779e94-a981-4020-a5d8-ef25ba701015
# ╠═e7010f58-8607-4bd6-97d2-b0a3a668bbb5
# ╠═d95bf7db-e4d9-4964-9019-69ac019dd7fb
