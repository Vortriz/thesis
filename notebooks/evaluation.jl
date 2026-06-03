### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 8326465a-1e33-442f-94fe-6b94f42702a4
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

# ╔═╡ 3a7242d3-4662-4f79-81aa-7883e3eb24cd
begin
	using CUDA
    @revise using GQML
    import Optimisers
    using PlutoSerialization
    using CairoMakie
end

# ╔═╡ ac56c5f3-9e63-472b-a2d5-e913036940fc
begin
    path = joinpath(
        base_path,
        "data",
        "2026-06-03_11-17-04",
    )

    model = open(deserialize, joinpath(path, "model.jls"))
    ansatz = model.ansatz
    config = model.config

    target_dist = config.trajectory.steps[end]

    params = open(deserialize, joinpath(path, "params.jls"))

    plots = Dict{String, CairoMakie.Figure}()
end;

# ╔═╡ b05e66c2-c9d3-4d86-beae-efd1b3652861
generated_trajectory = inference(
    ansatz,
    config,
    HaarDist(;
        n_qubits=ansatz.n_data,
        n_samples=config.batch_size,
    ),
    params,
);

# ╔═╡ 13064c91-73e7-4514-adec-76f2f4f0ae10
if ansatz.n_data == 1
    plots["generated_bloch"] = GQML.plot_bloch(;
        steps=generated_trajectory.steps,
        title="Inference Trajectory",
        ref_dist=target_dist,
        ref_label="Target Distribution",
    )
end

# ╔═╡ e2418b03-7082-4d99-bd9b-8ea4f9ac0174
plots["generated_trajectory"] = GQML.plot(
    typeof(target_dist);
    steps=generated_trajectory.steps,
    title="Inference trajectory",
    ref_dist=target_dist,
    ref_label="Target Distribution",
)

# ╔═╡ be71bd87-05b8-4573-8742-ab44f8f8c255
GQML.save(
	path,
	plots,
)

# ╔═╡ Cell order:
# ╟─8326465a-1e33-442f-94fe-6b94f42702a4
# ╠═3a7242d3-4662-4f79-81aa-7883e3eb24cd
# ╠═ac56c5f3-9e63-472b-a2d5-e913036940fc
# ╠═b05e66c2-c9d3-4d86-beae-efd1b3652861
# ╠═13064c91-73e7-4514-adec-76f2f4f0ae10
# ╠═e2418b03-7082-4d99-bd9b-8ea4f9ac0174
# ╠═be71bd87-05b8-4573-8742-ab44f8f8c255
