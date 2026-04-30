from collections.abc import Callable

import jax
import jax.numpy as jnp
import pennylane as qml
from flax import nnx
from jaxtyping import Array

from .measurement import measure_stochastic
from .types import Model


# [TODO] qjit, catalyst, https://docs.pennylane.ai/en/stable/news/program_capture_sharp_bits.html#parameter-broadcasting-and-vmap



dev = qml.device("default.qubit", wires=model.n_qubits)

@qml.qnode(dev, interface="jax")
def pqc(rngs: nnx.Rngs, model: Model, state: Array, params: Array):
    wires = range(model.n_qubits)

    state_with_ancilla = jnp.kron(
        state, jnp.zeros(2**model.n_ancilla, dtype=jnp.complex128).at[0].set(1)
    )
    qml.StatePrep(state_with_ancilla, wires=wires)

    for layer in range(model.n_layers):
        # 1. Local rotations
        for i in wires:
            qml.RX(params[layer, i, 0], wires=wires[i])
            qml.RY(params[layer, i, 1], wires=wires[i])

        # 2. Entangling layers (ring topology for scalability)
        for i in wires:
            qml.CZ(wires=[wires[i], wires[(i + 1) % model.n_qubits]])

    return measure_stochastic(rngs, model, qml.state())


# [TODO] verify its correctness.
# vmap may need to be removed because all states must be scrambled with different random parameters.
# also need to get rid of the Markovian nature of the scrambling circuit.
# def get_scramble_circuit(model: Model):
#     """Returns a jittable scrambling circuit."""
#     dev = qml.device("default.qubit.jax", wires=model.n_qubits)

#     @qml.qnode(dev, interface="jax")
#     def scramble_node(state, weight, rng_key):
#         """Applies random rotations and entangling gates."""
#         qml.StatePrep(state, wires=wires)

#         # Local random rotations
#         subkeys = jax.random.split(rng_key, 3)
#         for i in wires:
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
#         for i in wires:
#             for j in range(i + 1, model.n_qubits):
#                 qml.IsingZZ(weight * 0.5, wires=[i, j])

#         return qml.state()

#     return jax.vmap(scramble_node, in_axes=(0, None, 0))
