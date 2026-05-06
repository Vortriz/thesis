### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ abfa3092-3c76-11f1-91c5-1d1b6bcbbbad
begin
    import Pkg

    # activate the shared project environment
    Pkg.activate(Base.current_project())
    Pkg.instantiate()
end

# ╔═╡ 67d89a61-257a-4ef7-861a-4095e90cd9c1
begin|
	include("../src/base.jl")
	using .QML
end

# ╔═╡ 2be29cae-5360-42b6-8678-756bb76530e7
begin
	using Random
	using CairoMakie
end

# ╔═╡ 3029ecf8-0978-4b68-b002-7280f600f8bb
begin
	root = Base.current_project() |> dirname
	base_dir = "$root/assets/images/quddpm"
end

# ╔═╡ 432db477-60e0-45f6-afe6-31166e5b5c4d
begin
	const T = 20
	model = Model(
	    n_qubits = 1,
	    n_ancilla = 1,
	    T = T,
	    dataset_size = 1000,
	    n_layers = 6,
	    batch_size = 100,
	    rng = MersenneTwister(124),
	)

	gen_dist!(model, clustered)
	scramble!(model; weight_schedule=logrange(0.5, 2; length=T))
end

# ╔═╡ ebadf512-835c-4324-85fc-2093a356e97b
ffd = plot_forward_fidelity_decay(model)

# ╔═╡ b6b0a131-0d70-45d0-9fdb-79037acb5061
save("$base_dir/forward_fidelity_decay.png", ffd)

# ╔═╡ d09048bd-bc3d-4b5c-a18f-b57bb29b3f23
for i in [0, 5, 10, 15, 20]
	ca = plot_bloch_sphere(model.forward_ensembles[:, i])
	save("$base_dir/cluster-arbitrary-$i.png", ca)
end

# ╔═╡ Cell order:
# ╟─abfa3092-3c76-11f1-91c5-1d1b6bcbbbad
# ╠═67d89a61-257a-4ef7-861a-4095e90cd9c1
# ╠═2be29cae-5360-42b6-8678-756bb76530e7
# ╠═3029ecf8-0978-4b68-b002-7280f600f8bb
# ╠═432db477-60e0-45f6-afe6-31166e5b5c4d
# ╠═ebadf512-835c-4324-85fc-2093a356e97b
# ╠═b6b0a131-0d70-45d0-9fdb-79037acb5061
# ╠═d09048bd-bc3d-4b5c-a18f-b57bb29b3f23
