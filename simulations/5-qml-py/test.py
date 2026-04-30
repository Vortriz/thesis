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
    from src.losses import mmd_distance
    from src.measurement import measure_stochastic
    from src.plotting import plot_bloch_sphere
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
        n_data=1,
        n_ancilla=1,
        n_layers=4,
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
    from jaxtyping import Array

    return (Array,)


@app.cell
def _(Array, model):
    dev = qml.device("lightning.qubit", wires=model.n_qubits)

    @qml.qnode(dev, interface="jax", diff_method="finite-diff")
    def _pqc_block(n_qubits, n_layers, params: Array, state: Array):
        wires = range(n_qubits)
        qml.StatePrep(state, wires=wires)

        entangle_pairs = [(i, (i + 1) % n_qubits) for i in range(n_qubits - (n_qubits == 2))]
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RX(params[layer, i, 0], wires=wires[i])
                qml.RY(params[layer, i, 1], wires=wires[i])

            for (i, j) in entangle_pairs:
                qml.CZ(wires=[wires[i], wires[j]])

        return qml.state()

    # Use jax.vmap to handle batching of the input state
    pqc_block = jax.vmap(_pqc_block, in_axes=(None, None, None, 0))

    @jax.jit
    def apply_pqc(key: Array, model: Model, params: Array, state: Array):
        ancilla = jnp.zeros(2**model.n_ancilla, dtype=jnp.complex128).at[0].set(1)
        state_with_ancilla = jnp.kron(jax.lax.stop_gradient(state), ancilla)

        output_state = pqc_block(
            model.n_qubits, model.n_layers, params, state_with_ancilla
        )

        return measure_stochastic(key, model, output_state)

    return (apply_pqc,)


@app.cell
def _(apply_pqc):
    def loss_fn(model, t, key, input_batch, target_batch):
        # We target specifically params at timestep t
        params = model.params[t]
        output_batch = apply_pqc(key, model, params, input_batch)

        # Compute the MMD distance between the PQC output and the target batch
        loss = mmd_distance(output_batch, target_batch)
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
        optimizer = nnx.Optimizer(model, optax.adam(learning_rate=0.005), wrt=nnx.Param)
        epochs = model.epoch_schedule[t].item()
        losses = jnp.zeros(epochs)

        for epoch in progress(range(epochs), remove_on_exit=True):
            current_key = rngs.params()
            sample_key, pqc_key = random.split(current_key)

            batch_indices = random.choice(
                sample_key, model.dataset_size, (model.batch_size,), replace=False
            )
            target_batch = target_ensemble[batch_indices]

            # Pass 't' to the global loss_fn
            loss, grads = nnx.value_and_grad(loss_fn)(model, t, pqc_key, current_ensemble, target_batch)
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
def _():
    return


@app.cell
def _():
    # import pennylane as qml
    # import jax
    # import jax.numpy as jnp

    _dev_lightning = qml.device("lightning.qubit", wires=1)

    @qml.qnode(_dev_lightning, interface="jax", diff_method="finite-diff")
    def _test_circuit_lightning(params):
        qml.RX(params[0], wires=0)
        return qml.state()

    def _test_loss_lightning(params):
        state = _test_circuit_lightning(params)
        return jnp.real(state[0])

    print("Testing lightning.qubit with finite-diff...")
    try:
        _grad_lightning = jax.grad(_test_loss_lightning)(jnp.array([0.5]))
        print("Success:", _grad_lightning)
    except Exception as e:
        print("lightning.qubit Error:", type(e).__name__)
        print(e)

    print("\n------------------\n")

    _dev_default = qml.device("default.qubit", wires=1)

    @qml.qnode(_dev_default, interface="jax", diff_method="backprop")
    def _test_circuit_default(params):
        qml.RX(params[0], wires=0)
        return qml.state()

    def _test_loss_default(params):
        state = _test_circuit_default(params)
        return jnp.real(state[0])

    print("Testing default.qubit with backprop...")
    try:
        _grad_default = jax.grad(_test_loss_default)(jnp.array([0.5]))
        print("Success:", _grad_default)
    except Exception as e:
        print("default.qubit Error:", type(e).__name__)
        print(e)

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
