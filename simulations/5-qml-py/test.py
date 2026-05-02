import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")

with app.setup:
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import optax
    import pennylane as qml
    from flax import nnx
    from jax import random

    # from src.circuits import (
    #     get_pqc,
    #     hardware_efficient_ansatz,
    # )
    from src.distributions import DistributionType, gen_dist
    from src.losses import mmd_distance, wasserstein_distance
    from src.measurement import measure_stochastic
    from src.plotting import plot_bloch_sphere, plot_loss_training_vs_initial
    from src.types import Model

    progress = mo.status.progress_bar

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)
    rng = random.key(123)
    # qml.capture.enable()


@app.cell
def _():
    T = 5
    rngs = nnx.Rngs(12)
    model = Model(
        n_data=4,
        n_ancilla=2,
        n_layers=12,
        dataset_size=1000,
        batch_size=100,
        target_schedule=jnp.arange(T),
        epoch_schedule=jnp.repeat(500, T),
        rngs=rngs,
    )
    return T, model, rngs


@app.cell
def _(model, rngs):
    target_ensemble = gen_dist(
        DistributionType.CLUSTERED,
        rngs.params(),
        model.n_data,
        model.dataset_size,
        scale=0.08,
    )
    return (target_ensemble,)


@app.cell
def _(target_ensemble):
    mo.mpl.interactive(plot_bloch_sphere(target_ensemble))
    return


@app.cell
def _():
    from jax import Array

    return (Array,)


@app.cell
def _(Array, model):
    dev = qml.device("default.qubit", wires=model.n_qubits)

    # @qml.qjit
    @qml.qnode(dev, interface="jax")
    def pqc_block(n_data, n_layers, params: Array, state: Array):
        # Only prepare the data qubits. 
        # Ancilla qubits (n_data to n_qubits-1) stay in |0> by default.
        qml.StatePrep(state, wires=range(n_data))

        wires = range(model.n_qubits)
        entangle_pairs = [(i, (i + 1)) for i in range(model.n_qubits - 1)]
        for layer in range(n_layers):
            for i in wires:
                qml.RX(params[layer, i, 0], wires=wires[i])
                qml.RY(params[layer, i, 1], wires=wires[i])

            for (i, j) in entangle_pairs:
                qml.CZ(wires=[wires[i], wires[j]])

        return qml.state()

    @jax.jit
    def apply_pqc(key: Array, model: Model, params: Array, state: Array):
        # We pass the small data state directly to the QNode
        output_state = pqc_block(
            model.n_data, model.n_layers, params, state
        )

        return measure_stochastic(key, model, output_state)

    return (apply_pqc,)


@app.cell
def _(apply_pqc):
    def loss_fn(model, t, key, input_batch, target_batch):
        # We target specifically params at timestep t
        params = model.params[t]
        output_batch = apply_pqc(key, model, params, input_batch)

        # Compute the Wasserstein distance between the PQC output and the target batch
        loss = wasserstein_distance(output_batch, target_batch, max_iter=500)
        return loss

    return (loss_fn,)


@app.cell
def _(T, apply_pqc, loss_fn, model, rngs, target_ensemble):
    loss_history = []

    # Initialize current states with Haar noise
    current_ensemble = gen_dist(
        DistributionType.HAAR,
        rngs.params(),
        model.n_data,
        model.batch_size,
    )

    for t in progress(range(T)):
        optimizer = nnx.Optimizer(model, optax.adam(learning_rate=0.01), wrt=nnx.Param)
        epochs = model.epoch_schedule[t].item()
        losses = jnp.zeros(epochs)

        for epoch in progress(range(epochs), remove_on_exit=True):
            current_key = rngs.params()
            _, pqc_key = random.split(current_key)

            # Pass 't' and the FULL target_ensemble to the global loss_fn
            loss, grads = nnx.value_and_grad(loss_fn)(model, t, pqc_key, current_ensemble, target_ensemble)
            losses = losses.at[epoch].set(loss)

            optimizer.update(model, grads)

            # if epoch % 100 == 0:
            #     print(f"Step {t}, Epoch {epoch}: Loss = {loss:.6f}")

        loss_history.append(losses)

        current_ensemble = apply_pqc(
            rngs.params(), model, model.params[t], current_ensemble
        )
    return (loss_history,)


@app.cell
def _(loss_history):
    plot_loss_training_vs_initial(loss_history, title="Training Loss (Sequential Wasserstein)")
    return


@app.cell
def _(apply_pqc, rngs):
    def inference(model, _trigger=None):
        initial_ensemble = gen_dist(
            DistributionType.HAAR,
            rngs.params(),
            model.n_data,
            model.dataset_size,
        )
        for t in range(model.T):
            initial_ensemble = apply_pqc(
                rngs.params(), model, model.params[t], initial_ensemble
            )

        return initial_ensemble

    return (inference,)


@app.cell
def _(inference, loss_history, model):
    inferred = inference(model, _trigger=loss_history)
    return (inferred,)


@app.cell
def _(inferred):
    plot_bloch_sphere(inferred)
    return


@app.cell(disabled=True, hide_code=True)
def _():
    @jax.jit
    def pure_dm_to_state(rho):
        # 1. Get the diagonal to find the largest element (avoids zeros)
        diag = jnp.diag(rho)
        idx = jnp.argmax(jnp.abs(diag))

        # 2. Extract that column
        column = rho[:, idx]

        # 3. Normalize to get the state vector |ψ⟩
        state_vector = column / jnp.sqrt(diag[idx])
        return state_vector


    pure_dm_to_state_vmap = jax.vmap(pure_dm_to_state, in_axes=0)
    return


@app.cell
def _(haar, target_ensemble):
    wasserstein_distance(
        target_ensemble,
        haar
    )
    return


@app.cell
def _(model, rngs):
    haar = gen_dist(
        DistributionType.HAAR,
        rngs.params(),
        model.n_data,
        model.dataset_size,
    )
    return (haar,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
