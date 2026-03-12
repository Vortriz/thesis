# Gemini Development Log: Porting QDDPM to Julia

This document summarizes the development and debugging process of porting a Quantum Denoising Diffusion Probabilistic Model (QDDPM) from Python to Julia, with the assistance of the Gemini CLI agent.

## 1. Initial Objective

The primary goal was to translate an existing Python implementation (`QDDPM.py`) into a functional Julia script (`notebook_alt.jl`), leveraging Julia's performance and the `Yao.jl` ecosystem for quantum simulation.

## 2. Debugging the Training Process

A significant portion of the effort was dedicated to debugging the training loop, which uses `Zygote.jl` for automatic differentiation. We encountered and resolved several issues:

### 2.1. Zygote & Automatic Differentiation Errors

- **Array Reshaping**: An initial `MethodError` related to `eachrow` was traced back to an incorrect `reshape` operation that didn't account for Julia's column-major memory layout. This was fixed by adjusting the dimension order in `reshape`.

- **Random Sampling Gradient**: Zygote failed with an `iterate(::Nothing)` error when trying to differentiate through the random sampling of measurement outcomes. This was resolved by wrapping the sampling logic in a `Zygote.ignore()` block, correctly treating the discrete outcomes as constants in the gradient calculation.
- **Array Mutation**: A `Mutating arrays is not supported` error was triggered by an in-place update (`setindex!`) inside a loop. The code was refactored to a functional style using a comprehension and `hcat` to build the new array without mutation.
- **Type-related Gradient Errors**:
    - An error occurred when Zygote attempted to find a gradient for the `DitStr` type returned by `Yao.measure`. The fix was to convert the `DitStr` objects to standard `Int`s inside the `Zygote.ignore` block.
    - An obscure `llvmcall requires the compiler` error appeared when using loss functions from `OptimalTransport.jl` and `ExactOptimalTransport.jl`. This pointed to a deep issue in the interaction between the AD engine and the libraries' low-level code.

### 2.2. Yao.jl API Corrections

The debugging process also revealed several incorrect assumptions about the `Yao.jl` API, which were corrected:

- **Batched `probs`**: `probs(register, qubits)` is not implemented for `BatchedArrayReg`. We reverted to a manual calculation by reshaping the state matrix and summing over the appropriate dimensions.
- **Batched `getindex`**: A `BatchedArrayReg` does not support integer indexing (`batch[i]`). The fix was to extract the state vector manually (`batch.state[:, i]`) and construct a new `ArrayReg`.
- **`select` vs. `reshape`**: We explored using the idiomatic `focus!` and `select` functions for measurement collapse. While conceptually cleaner, this approach proved to have subtle interactions with Zygote, causing a hard-to-diagnose dimension mismatch. We ultimately reverted to a more explicit and robust implementation using `reshape` and array slicing, which fixed the issue.

## 3. Loss Function Exploration

- **MMD -> Sinkhorn -> EMD**: We transitioned from the initial MMD loss to the Sinkhorn distance. This led to the `llvmcall` error. We then switched to the exact Earth Mover's Distance (`emd2`) from `ExactOptimalTransport.jl`.
- **EMD Optimizer Configuration**: The `emd2` function required a linear programming solver. We first used `Tulip.jl`, but it was too slow. We then switched to `Clp.jl`, which produced a `AddVariableNotAllowed` error. This was resolved by wrapping the `Clp.Optimizer` in a `MOI.Utilities.CachingOptimizer`, as recommended by the error message.

## 4. Performance Optimization

Once the script was functionally correct, we addressed a major performance bottleneck:

- **Problem**: The training loop was converting states from a `Vector{ArrayReg}` to a `BatchedArrayReg` on every single training step, causing significant memory allocation and overhead.
- **Solution**: The core `Model` data structure was refactored. The `scramble!` and constructor functions were rewritten to work directly with batched states from the beginning. This eliminated the repeated conversion, pre-computing the batches once before training starts and using efficient array slicing within the training loop.

## 5. Type Stability and Performance Tuning with JET.jl

A deep dive with `JET.jl` was conducted to eliminate type instabilities, which are a primary source of poor performance and memory allocations in Julia.

- **Core Principle**: The central theme was to make all data structures and function calls "concrete" from the compiler's perspective, allowing it to generate highly specialized and optimized code.

- **Parametric Structs**:
    - **Problem**: Persistent type instabilities were traced back to fields in the `Model` struct holding abstract types (e.g., `rng::AbstractRNG`, `schedule::StepRangeLen`).
    - **Solution**: The `Model` struct was made fully parametric on these types (e.g., `struct Model{R<:AbstractRNG, S<:StepRangeLen}`). This ensures that any instance of `model` is concretely typed, resolving instabilities in functions that use it.

- **Concrete Constructors**:
    - **Problem**: Generic `Yao.jl` constructors like `ArrayReg(data)` were not inferring type parameters at compile time, resulting in abstractly-typed objects.
    - **Solution**: We replaced generic constructors with their specific, parametric versions (e.g., `ArrayReg{N,T,AT}(...)`). This often required reshaping the input data (`Vector` to `Matrix`) to match the exact backing type expected by the concrete constructor.

- **Containing Library Instability**:
    - **Problem**: Even with concrete inputs, some `Yao.jl` functions (`dispatch`, `apply`) remained type-unstable due to their dynamic internal design (e.g., storing circuit blocks in a `Vector{AbstractBlock}`).
    - **Solution**: The instability was "contained" at the function call boundary using type assertions on the return value (e.g., `circuit = dispatch(...)::ChainBlock{2}`). This prevented the instability from propagating into our code, even though the dispatch within the library call remains.

- **Global Variables & Aliases**:
    - **Problem**: Type aliases defined without `const` were treated as non-constant global variables, causing type instability.
    - **Solution**: All type aliases were declared as `const` (e.g., `const ConcreteArrayReg = ...`), allowing the compiler to treat them as fixed values for optimization.

- **Performance vs. Differentiability (Zygote)**:
    - **Problem**: An optimization for the `measure_and_collapse` function using in-place array mutation (`.=`) drastically reduced allocations but broke automatic differentiation in `Zygote.jl`.
    - **Solution**: A hybrid approach was implemented. The non-differentiable sampling logic was optimized with high-performance, low-allocation code inside a `Zygote.ignore()` block. The rest of the function, which requires gradients, was reverted to a non-mutating "functional" style that Zygote can process, achieving a balance between performance and differentiability.

## 6. Explicit Gradient Calculation with Enzyme.jl

A major shift in the project was moving from a gradient-free approach (`Rotosolve`) to a more standard, gradient-based training paradigm for the diffusion model. This required an efficient method for calculating the gradient of the Wasserstein distance loss.

### 6.1. Gradient Derivation via the Envelope Theorem

The key insight, provided by the user, was to use the **Envelope Theorem** to derive an explicit gradient for the Wasserstein distance, $W(\theta)$, without needing to differentiate through the iterative optimal transport (OT) solver.

The derivation established that the gradient of the Wasserstein distance with respect to the generator parameters $\theta$ is given by:
$$\frac{\partial W}{\partial \theta} = - \sum_{i,j} \Gamma^*_{i,j} \frac{\partial}{\partial \theta} |\langle \phi_i | \psi_j(\theta) \rangle|^2$$
where $\Gamma^*$ is the optimal transport plan (treated as a constant for the gradient calculation) and $|\langle \phi_i | \psi_j(\theta) \rangle|^2$ is the fidelity `F`.

### 6.2. Implementation with Enzyme.jl

This derivation led to an efficient two-stage implementation within the `train_step!` function:

1.  **Forward Pass**: First, for the current parameters `θ`, the IPOT solver (`wasserstein_distance`) is run to find the optimal transport plan `Γ*`. This value is then treated as a fixed constant.

2.  **Backward Pass**: A `surrogate_loss_for_grad` function, defined as $L(\theta) = - \sum \Gamma^*_{i,j} F_{i,j}(\theta)$, is passed to `Enzyme.autodiff`. The gradient of this surrogate function is equivalent to the true gradient of the Wasserstein distance. This avoids the computationally prohibitive task of differentiating through the IPOT solver's loops.

### 6.3. Debugging and Refinement

The implementation of this new training loop involved several debugging and refinement steps:

- **Loss Function Optimization**: The IPOT solver (`ipot_wd`/`wasserstein_distance`) was heavily optimized to reduce memory allocations by using in-place operations (`mul!`, `@.`) and pre-allocated buffers. This reduced allocations per call from >14,000 to fewer than 60.
- **Simplifying the AD Workflow**: We refined the `train_step!` logic to be more efficient. Instead of solving for the loss and the plan separately, the final version solves for the plan `Γ*` and then calculates the true loss from the return value of the `Enzyme.autodiff` call, avoiding redundant computations.
- **Resolving Library Confusion**: A critical `UndefVarError: state not defined in Optim` was traced back to a library mix-up. The code was using an API from `Optimisers.jl` (e.g., `state`, `update!`) while importing `Optim.jl`. The issue was resolved by adding `Optimisers.jl` to the project dependencies and correcting the function calls to `Optimisers.setup` and `Optimisers.update!`.
- **Code Clarity**: User feedback drove several simplifications to the code, such as removing redundant data conversions, using concrete types in function signatures, and making design choices that prioritized simplicity over minor performance gains (e.g., localizing optimizer state).
