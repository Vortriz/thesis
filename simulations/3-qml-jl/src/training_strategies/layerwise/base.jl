export train!

function train!(model::Model, strategy::StepwiseStrategy)
    @info "Strategy: $(typeof(strategy)) (Autoregressive)"

    # Initialize with Haar random states (at t=T)
    # Convert Ensemble -> ConcreteBatchedArrayReg
    current_reg::ConcreteBatchedArrayReg =
        generate_rand_ensemble(model.n_qubits, model.backward_ensemble_size) |> ensemble_to_batch

    @progress for t in model.T:-1:1
        # @info "Training timestep t=$t"

        # Train step t using current_reg as input (from t+1)
        # Note: train_step! must accept ConcreteBatchedArrayReg
        train_step!(model, strategy, t, current_reg)

        # Update current_reg for next step (t-1)
        # denoise returns Matrix{ComplexF64}, we wrap it back to Reg
        denoised_matrix = denoise(model, strategy, current_reg, model.trained_params[:, t])
        current_reg = ConcreteBatchedArrayReg(denoised_matrix, size(denoised_matrix, 2))
    end
end

include("grad_enzyme.jl")
include("grad_zygote.jl")
include("qnspsa.jl")
include("rotosolve.jl")
include("spsa.jl")
