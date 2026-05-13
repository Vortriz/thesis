### A Pluto.jl notebook ###
# v0.20.25

using Markdown
using InteractiveUtils

# ╔═╡ 168a33fa-4be8-11f1-937a-99ef8733e91e
begin
    import Pkg

    # activate the shared project environment
    Pkg.activate(Base.current_project())
    Pkg.instantiate()
end

# ╔═╡ 7c3a8a85-b2b5-4dc7-9638-7b7d2d6f3a3e
# ╠═╡ show_logs = false
begin
    include("../src/QML.jl")
    using .QML
end

# ╔═╡ f0dd9925-d3d2-4cad-9b9f-bf11ac792953
begin
    using Yao
    using Random
    using LinearAlgebra
    using CairoMakie
    using StatsBase
    import Zygote
    import Optimisers

    using ProgressLogging
	using MLFlowClient
    import JLSO

	import Dates
	import Logging

    # using BenchmarkTools
    # using JET
    # using ProfilePerfetto
end

# ╔═╡ e7f9dfca-dfd8-4b44-bfeb-91107e281b10
# ╠═╡ show_logs = false
begin
	mlf = MLFlow("http://localhost:5000")
	exp_name = "QML_Training"
	exp = try
        createexperiment(mlf, exp_name)
    catch e
        getexperimentbyname(mlf, exp_name)
    end
    mlf_run = createrun(mlf, exp;
						run_name=Dates.format(Dates.now(), "yyyy-mm-dd @ HH:MM:SS"))
end

# ╔═╡ 7479301c-85c0-4773-bc5d-1195c7cb47ad
begin
    const T = 2
    const rng = MersenneTwister(1234)
    arch = ModelArch(;
        n_data=2,
        n_ancilla=1,
        n_layers=2,
        ansatz_builder=EHA,
        collapse_method=normal,
    )

    initial_ensemble = gen_dist(
        Val(haar),
        rng;
        n_qubits=arch.n_data,
        n_samples=100,
    )
    target_ensemble = gen_dist(
        Val(clustered),
        rng;
        n_qubits=arch.n_data,
        n_samples=1000,
    )

    config = TrainConfig(
        Val(direct);
        initial_ensemble=initial_ensemble,
        target_ensemble=target_ensemble,
        epoch_schedule=fill(300, T),
        optimizer=Optimisers.AMSGrad(0.01),
    )

    logbatch(mlf, mlf_run; params=get_hyperparams(arch, config, rng))

end;

# ╔═╡ 1af82bcf-1787-403e-a4c8-8bc59f1bd995
function train(
    arch::ModelArch,
    config::TrainConfig;
    mlf::Union{MLFlow, Nothing} = nothing,
    mlf_run::Union{Run, Nothing} = nothing
)
    params = rand(rng, Float64, (arch.n_params_ppb, config.T))
    loss_history = [zeros(Float64, n) for n in config.epoch_schedule]

    model_state = ModelState()
    model_state.current_ensemble = config.initial_ensemble |> copy

    global_step = 0

    @progress for t in 1:config.T
        append_qubits!(model_state.current_ensemble, arch.n_ancilla)

        model_state.current_params = params[:, t]
        opt_state = Optimisers.setup(config.optimizer, model_state.current_params)

        target_idx = config.target_schedule[t]
        target_matrix = config.target_trajectory[target_idx].state

        @progress for epoch in 1:config.epoch_schedule[t]
            global_step += 1
            target_indices = sample(
                1:config.dataset_size,
                config.batch_size,
                replace=false,
            )
            model_state.target_matrix = @view target_matrix[:, target_indices]

            loss, grads = loss_and_grads(arch, model_state)

            Optimisers.update!(opt_state, model_state.current_params, grads[1])
            loss_history[t][epoch] = loss

            if !isnothing(mlf) && !isnothing(mlf_run)
                logmetric(mlf, mlf_run, "training_loss", loss; step=global_step)
            end
        end

        model_state.current_ensemble =
            apply_pqc(
                arch,
                model_state.current_ensemble,
                model_state.current_params,
            ) |> batch_and_normalize

        params[:, t] = model_state.current_params
    end

    return loss_history, params
end

# ╔═╡ de779e94-a981-4020-a5d8-ef25ba701015
begin
    loss_history, params = train(arch, config; mlf=mlf, mlf_run=mlf_run)
    final_training_loss = get_final_training_loss(loss_history)

    logmetric(mlf, mlf_run, "final_training_loss", final_training_loss)

	io = IOBuffer()
	JLSO.save(io, :params => params)
	bytes = take!(io)
    uploadartifact(mlf, "params.jld2", bytes)

    @show final_training_loss
end

# ╔═╡ 0b68baca-e370-466a-a3aa-6785d62b9736
function plot_to_bytes(fig)
	io = IOBuffer()
	show(io, MIME"image/svg+xml"(), fig)
	return take!(io)
end

# ╔═╡ 0c83c042-7de8-4b61-a041-59d39f9d61bd
begin
	fig_target_bloch = plot_bloch_sphere(target_ensemble)
	if !isnothing(fig_target_bloch)
    	uploadartifact(mlf, "target_bloch_sphere.svg", plot_to_bytes(fig_target_bloch))
	end
    fig_target_bloch
end

# ╔═╡ e7010f58-8607-4bd6-97d2-b0a3a668bbb5
begin
	fig_loss = plot_loss_history(loss_history; yscale=log10)
    uploadartifact(mlf, "loss_plot.svg", plot_to_bytes(fig_loss))
	fig_loss
end

# ╔═╡ c0c82792-abd4-48b4-8fe2-d913c60d1e92
begin
	generated_trajectory = inference(
	    arch,
	    config,
	    gen_dist(
	        Val(haar),
	        rng;
	        n_qubits=arch.n_data,
	        n_samples=config.batch_size,
	    ),
	    params,
	);

	inf_loss = wasserstein_distance(
		generated_trajectory[end].state,
		target_ensemble.state,
	)
    logmetric(mlf, mlf_run, "inference_loss", inf_loss)
    @show inf_loss
end

# ╔═╡ dcdba0ce-8b94-4d38-8e9e-980070e155e0
begin
	fig_generated_bloch = plot_bloch_sphere(generated_trajectory[end])
	if !isnothing(fig_generated_bloch)
    	uploadartifact(mlf, "generated_bloch_sphere.svg", plot_to_bytes(fig_generated_bloch))
	end

	updaterun(mlf, mlf_run; status=RunStatus.FINISHED)

	fig_generated_bloch
end

# ╔═╡ Cell order:
# ╟─168a33fa-4be8-11f1-937a-99ef8733e91e
# ╠═7c3a8a85-b2b5-4dc7-9638-7b7d2d6f3a3e
# ╠═f0dd9925-d3d2-4cad-9b9f-bf11ac792953
# ╠═e7f9dfca-dfd8-4b44-bfeb-91107e281b10
# ╠═7479301c-85c0-4773-bc5d-1195c7cb47ad
# ╠═0c83c042-7de8-4b61-a041-59d39f9d61bd
# ╠═1af82bcf-1787-403e-a4c8-8bc59f1bd995
# ╠═de779e94-a981-4020-a5d8-ef25ba701015
# ╠═0b68baca-e370-466a-a3aa-6785d62b9736
# ╠═e7010f58-8607-4bd6-97d2-b0a3a668bbb5
# ╠═c0c82792-abd4-48b4-8fe2-d913c60d1e92
# ╠═dcdba0ce-8b94-4d38-8e9e-980070e155e0
