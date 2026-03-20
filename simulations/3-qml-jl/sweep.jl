using Distributed
using Dates

# 1. Setup workers (Adjust the number of workers as needed)
if nprocs() == 1
    addprocs(2) # Change this to match your CPU cores
end

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
    using Random, Statistics
end

using Hyperopt
using Plots
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
@everywhere function run_sweep_trial(η, ϵ, β, job_dir)
    # Define model parameters (adjust as needed)
    T = 5

    model = Model(
        n_qubits = 2,
        n_ancilla = 1,
        T = T,
        forward_ensemble_size = 1000,
        n_layers = 6,
        backward_ensemble_size = 100,
        rng = MersenneTwister(124),
    )

    initialize_forward_ensemble!(model; spread=0.05)
    scramble!(model; weight_schedule=logrange(0.8, 2.4; length=T))

    # Using DirectQNSPSA as example strategy for sweep
    strategy = DirectQNSPSA(
        loss_function = wasserstein_distance,
        n_iters = 51, # Reduced iterations for faster sweep
        hyper_params = (η=η, ϵ=ϵ, β=β, history_length=5),
    )

    # Train the model
    train!(model, strategy)

    # Record the run inside the job directory
    # Generate plot (CairoMakie might be slow or require specific setup on headless workers)
    bplh = plot_training_loss_history(model, strategy)

    record_run(model, strategy, bplh, "Direct (Sweep Trial)"; save_dir_base = job_dir)

    # Return the final loss (average of last 50 iterations)
    if isempty(strategy.loss_history) || isempty(strategy.loss_history[1])
        return 1.0 # High loss for failed runs
    end

    return mean(strategy.loss_history[1][max(1, end-50):end])
end

# 4. Perform the parallel hyperparameter search
# Sampling: η (log space), ϵ and β from discrete choices
ho = @phyperopt for i = 2, # Total number of samples
                  η = exp10.(range(-3, -0.7, length=20)),
                  ϵ = [0.01, 0.05, 0.1],
                  β = [0.01, 0.05, 0.1]

    # Each iteration runs this block (on workers if using @phyperopt)
    run_sweep_trial(η, ϵ, β, job_dir)
end

println("\n--- Sweep Completed ---")
println("Best Loss Found: ", ho.minimum)
println("Best Hyperparameters: ", ho.minimizer)

# 5. Save a sweep summary and hyperoptimizer plot
open(joinpath(job_dir, "sweep_result_summary.txt"), "w") do f
    println(f, "--- Hyperopt Sweep Result ---")
    println(f, "Timestamp: ", job_timestamp)
    println(f, "Total Samples: ", 20)
    println(f, "Best Loss: ", ho.minimum)
    println(f, "\n--- Best Parameters ---")
    for (k, v) in zip(keys(ho.params), ho.minimizer)
        println(f, "$k: $v")
    end
end

ho_plot = plot(ho, size=(1200, 800))
savefig(ho_plot, joinpath(job_dir, "hyperopt_sweep_plot.png"))

println("Full results saved in: $job_dir")
