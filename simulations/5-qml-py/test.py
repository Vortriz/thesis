import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax import random
    from src.types import Model
    from src.circuits import (
        hardware_efficient_ansatz,
        get_pqc,
    )
    from src.measurement import measure_stochastic
    from src.distributions import gen_dist, DistributionType
    from src.plotting import plot_bloch_sphere
    from src.losses import mmd_distance

    import pennylane as qml
    from catalyst import qjit
    import optax

    import marimo as mo

    progress = mo.status.progress_bar

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)
    rng = random.key(123)
    # qml.capture.enable()


@app.cell
def _():
    T = 5
    model = Model.create(
        n_data=1,
        n_ancilla=1,
        n_layers=4,
        dataset_size=1000,
        batch_size=100,
        target_schedule=jnp.arange(T),
        epoch_schedule=jnp.repeat(100, T),
        key=random.key(12),
    )
    return (model,)


@app.cell
def _(model):
    target_ensemble = gen_dist(
        DistributionType.CLUSTERED,
        model.key,
        model.n_data,
        model.dataset_size,
        scale=0.03,
    )
    return (target_ensemble,)


@app.cell
def _(target_ensemble):
    mo.mpl.interactive(plot_bloch_sphere(target_ensemble))
    return


@app.cell
def _():
    jnp.kron(
        jnp.array([1, 1]) / jnp.sqrt(2),
        jnp.array([1, 0]),
    )
    return


@app.cell
def _(model):
    pqc = get_pqc(model, hardware_efficient_ansatz)
    return (pqc,)


@app.cell
def _(pqc):
    def compute_loss(key, params, batch_data):
        pqc_output = pqc(key, batch_data, params)
        return mmd_distance(pqc_output, batch_data)

    return (compute_loss,)


@app.cell
def _(compute_loss, target_ensemble):
    loss_history = []
    key = model.key
    optimizer = optax.adam(learning_rate=0.005)

    for t in progress(range(model.T)):
        losses = jnp.zeros(model.epoch_schedule[t])
        params = model.params[t]
        opt_state = optimizer.init(params)
    
        for epoch in progress(range(model.epoch_schedule[t]), remove_on_exit=True):
            key, subkey = jax.random.split(key)
        
            # Sample a batch of data
            batch_indices = random.choice(
                subkey, model.dataset_size, (model.batch_size,), replace=False
            )
            batch_data = target_ensemble[batch_indices]

            # Update parameters using an optimizer (e.g., Adam)
            loss, grads = jax.value_and_grad(compute_loss, argnums=1)(subkey, params, batch_data)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            # store the loss
            losses = losses.at[epoch].set(loss)

        model = model.replace(params=model.params.at[t].set(params))
        loss_history.append(losses)
    return (model,)


@app.cell
def _():
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
    return (pure_dm_to_state_vmap,)


@app.function(disabled=True, hide_code=True)
def get_pqc_test(model: Model, ansatz):
    """Returns a jittable Parameterized Quantum Circuit."""
    dev = qml.device("default.qubit", wires=model.n_qubits)

    @qml.qnode(dev, interface="jax")
    def pqc_block(state, params):
        qml.StatePrep(state, wires=range(model.n_qubits))
        ansatz(params, range(model.n_qubits))

        # for i in range(model.n_data, model.n_qubits+1):
        #     qml.measure(i)

        return qml.density_matrix(wires=range(model.n_data))

    return pqc_block


@app.cell(disabled=True)
def _(model, pqc_block):
    d = pqc_block(
        jnp.zeros(2**model.n_qubits, dtype=jnp.complex128).at[0].set(1),
        # jnp.zeros((model.dataset_size, 2**model.n_qubits), dtype=jnp.complex128).at[:, 0].set(1),
        jax.random.normal(model.rng_key, (model.n_layers, model.n_qubits, 2)),
    )
    return (d,)


@app.cell
def _(d):
    d
    return


@app.cell
def _(d):
    jnp.trace(d @ d)
    return


@app.cell(disabled=True)
def _(d, pure_dm_to_state_vmap):
    pure_dm_to_state_vmap(d)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
