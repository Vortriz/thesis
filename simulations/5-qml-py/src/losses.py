import jax
import jax.numpy as jnp

from jaxtyping import Array


def mmd_distance(ensemble1: Array, ensemble2: Array):
    """
    Calculates the MMD distance between two ensembles of quantum states.
    Uses the RBF-like kernel K(psi, phi) = |<psi|phi>|^2.
    """
    # Matrix of inner products: <psi_i | phi_j>
    # ensemble shape: (batch_size, 2**n_qubits)
    inner_prod_11 = jnp.abs(jnp.matmul(ensemble1, ensemble1.conj().T)) ** 2
    inner_prod_22 = jnp.abs(jnp.matmul(ensemble2, ensemble2.conj().T)) ** 2
    inner_prod_12 = jnp.abs(jnp.matmul(ensemble1, ensemble2.conj().T)) ** 2

    # R(ensemble1, ensemble2) = 1 - mean(|<psi|phi>|^2)
    r11 = 1.0 - jnp.mean(inner_prod_11)
    r22 = 1.0 - jnp.mean(inner_prod_22)
    r12 = 1.0 - jnp.mean(inner_prod_12)

    return 2 * r12 - r11 - r22


def wasserstein_distance(ensemble1, ensemble2):
    # [TODO]
    return


# def measure_ancilla(states, n_qubits, n_ancilla, rng_key):
#     """
#     Simulates measurement of ancilla qubits and state collapse.
#     Assumes ancilla are the 'trailing' qubits.
#     """
#     # Reshape to (batch_size, 2**n_ancilla, 2**n_qubits)
#     reshaped = states.reshape(states.shape[0], 2**n_ancilla, 2**n_qubits)

#     # Calculate probabilities of each ancilla measurement outcome
#     probs = jnp.sum(jnp.abs(reshaped) ** 2, axis=2)  # (batch_size, 2**n_ancilla)

#     # Sample outcomes
#     outcomes = jax.random.categorical(rng_key, jnp.log(probs + 1e-15), axis=1)

#     # Extract the collapsed state for each sample
#     def get_collapsed(s, outcome):
#         return reshaped[s, outcome] / jnp.linalg.norm(reshaped[s, outcome])

#     collapsed_states = jax.vmap(get_collapsed)(jnp.arange(states.shape[0]), outcomes)

#     return collapsed_states


# def attach_ancilla(states, n_ancilla):
#     """Prepends/Appends ancilla in |0> state."""
#     batch_size = states.shape[0]
#     dim_main = states.shape[1]
#     dim_total = dim_main * (2**n_ancilla)

#     # Create |psi> \otimes |00...0>
#     # This is equivalent to padding with zeros in the state vector representation
#     padding = jnp.zeros((batch_size, dim_total - dim_main), dtype=jnp.complex128)
#     return jnp.concatenate([states, padding], axis=1)
