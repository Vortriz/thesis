using Distributed
using Dates

# 1. Setup workers (Managed via julia -p N)
@everywhere begin
    using Pkg
    Pkg.activate(Base.current_project())
end

# Instantiate only on the master process
Pkg.instantiate()

@everywhere include("src/base.jl")
using .QDDPM
@everywhere begin
    using Main.QDDPM
    using Random, Statistics, Optimisers
end

using Hyperopt
using Plots

# For headless environment
ENV["GKSwstype"] = "100"

# 2. Setup job directory for this specific sweep
job_timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
job_dir = joinpath("saves", "sweep_$(job_timestamp)")
mkpath(job_dir)

println("--- Starting Hyperopt Sweep ---")
println("Job Directory: $job_dir")
println("Number of workers: $(nprocs())")

# 3. Define the objective function for the sweep
# We make it available everywhere so workers can call it
@everywhere function run_sweep_trial(strategy_factory::Function, hyper_params::NamedTuple, job_dir::String)
    # Define model parameters (adjust as needed)
    T = 1

    model = Model(
        n_qubits=4,
        n_ancilla=2,
        T=T,
        forward_ensemble_size=1000,
        n_layers=60,
        backward_ensemble_size=100,
        rng=MersenneTwister(124),
    )

    initialize_forward_ensemble!(model; spread=0.05)
    # scramble!(model; weight_schedule=logrange(0.8, 2.4; length=T))
    scramble!(model; weight_schedule=[10])

    # Build strategy using the factory and provided hyperparameters
    strategy = strategy_factory(hyper_params)

    # Train the model
    train!(model, strategy)

    # Record the run
    bplh = plot_training_loss_history(model, strategy)
    record_run(model, strategy, bplh, "Sweep Trial"; save_dir_base=job_dir)

    # Return the final loss (average of last 50 iterations)
    if isempty(strategy.loss_history) || isempty(strategy.loss_history[1])
        return 1.0
    end

    return mean(strategy.loss_history[1][max(1, end - 50):end])
end

@everywhere begin
    # 4. Perform the parallel hyperparameter search
    qnspsa_factory(p) = DirectQNSPSA(
        loss_function=wasserstein_distance,
        n_iters=51,
        hyper_params=(η=p.η, ϵ=p.ϵ, β=p.β, history_length=5)
    )

    grad_factory(p) = DirectGradZygote(
        loss_function=wasserstein_distance,
        optimizer=Optimisers.AMSGrad(p.η),
        n_iters=500
    )
end

# Choose your factory here
current_factory = grad_factory

ho = @phyperopt for i = 2, # Total number of samples
    η = exp10.(range(-3, -0.7, length=20))
    # ϵ = [0.01, 0.05, 0.1],
    # β = [0.01, 0.05, 0.1]

    # Package hyperparameters into a NamedTuple
    params = (η=η,)

    run_sweep_trial(current_factory, params, job_dir)
end

println("\n--- Sweep Completed ---")
println("Best Loss Found: ", ho.minimum)
println("Best Hyperparameters: ", ho.minimizer)

# 5. Save a sweep summary and hyperoptimizer plot
open(joinpath(job_dir, "sweep_result_summary.txt"), "w") do f
    println(f, "--- Hyperopt Sweep Result ---")
    println(f, "Timestamp: ", job_timestamp)
    println(f, "Total Samples: ", 30)
    println(f, "Best Loss: ", ho.minimum)
    println(f, "\n--- Best Parameters ---")
    for (k, v) in zip(keys(ho.params), ho.minimizer)
        println(f, "$k: $v")
    end
end

ho_plot = plot(ho, size=(1200, 800))
savefig(ho_plot, joinpath(job_dir, "hyperopt_sweep_plot.png"))

println("Full results saved in: $job_dir")
