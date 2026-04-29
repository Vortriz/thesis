from collections.abc import Callable

import jax
import jax.numpy as jnp
import pennylane as qml

from .measurement import measure_stochastic
from .types import Model


# [TODO] qjit, catalyst, https://docs.pennylane.ai/en/stable/news/program_capture_sharp_bits.html#parameter-broadcasting-and-vmap
def hardware_efficient_ansatz(params, wires):
    """
    Default Hardware Efficient Ansatz.
    params shape: (n_layers, n_qubits, 2)
    """
    n_layers, n_qubits, _ = params.shape
    for layer in range(n_layers):
        # 1. Local rotations
        for i in range(n_qubits):
            qml.RX(params[layer, i, 0], wires=wires[i])
            qml.RY(params[layer, i, 1], wires=wires[i])

        # 2. Entangling layers (ring topology for scalability)
        for i in range(n_qubits):
            qml.CZ(wires=[wires[i], wires[(i + 1) % n_qubits]])


def get_pqc(model: Model, ansatz) -> Callable:
    """Returns a jittable Parameterized Quantum Circuit."""
    dev = qml.device("default.qubit", wires=model.n_qubits)

    @qml.qnode(dev, interface="jax")
    def pqc_block(state, params):
        state_with_ancilla = jnp.kron(state, jnp.zeros(2**model.n_ancilla, dtype=jnp.complex128).at[0].set(1))
        qml.StatePrep(state_with_ancilla, wires=range(model.n_qubits))
        ansatz(params, range(model.n_qubits))
        return qml.state()

    def pqc_with_measurement(key, state, params):
        full_states = pqc_block(state, params)
        return measure_stochastic(key, full_states, model.n_data, model.n_ancilla)

    return jax.jit(pqc_with_measurement)


# [TODO] verify its correctness.
# vmap may need to be removed because all states must be scrambled with different random parameters.
# also need to get rid of the Markovian nature of the scrambling circuit.
# def get_scramble_circuit(model: Model):
#     """Returns a jittable scrambling circuit."""
#     dev = qml.device("default.qubit.jax", wires=model.n_qubits)

#     @qml.qnode(dev, interface="jax")
#     def scramble_node(state, weight, rng_key):
#         """Applies random rotations and entangling gates."""
#         qml.StatePrep(state, wires=range(model.n_qubits))

#         # Local random rotations
#         subkeys = jax.random.split(rng_key, 3)
#         for i in range(model.n_qubits):
#             qml.RX(
#                 weight
#                 * jax.random.uniform(
#                     subkeys[0], shape=(), minval=-jnp.pi, maxval=jnp.pi,
#                 ),
#                 wires=i,
#             )
#             qml.RY(
#                 weight
#                 * jax.random.uniform(
#                     subkeys[1], shape=(), minval=-jnp.pi, maxval=jnp.pi,
#                 ),
#                 wires=i,
#             )
#             qml.RZ(
#                 weight
#                 * jax.random.uniform(
#                     subkeys[2], shape=(), minval=-jnp.pi, maxval=jnp.pi,
#                 ),
#                 wires=i,
#             )

#         # Entangling gates (RZZ as in your Julia code)
#         # We can implement a simplified version or a full combination
#         for i in range(model.n_qubits):
#             for j in range(i + 1, model.n_qubits):
#                 qml.IsingZZ(weight * 0.5, wires=[i, j])

#         return qml.state()

#     return jax.vmap(scramble_node, in_axes=(0, None, 0))
