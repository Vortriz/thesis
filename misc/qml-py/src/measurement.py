import jax
import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, PRNGKeyArray

from src.types import Model


@jit
def measure_stochastic(key: PRNGKeyArray, model: Model, state: Array) -> Array:
    """
    Simulates a stochastic measurement of the ancilla qubits and returns
    the collapsed, normalized pure state for the data qubits.

    Data qubits are assumed to be the first `n_data` wires.
    Ancilla qubits are assumed to be the last `n_ancilla` wires.

    Args:
        key: JAX PRNG Key
        model: Model object
        state: State vector of shape (batch_size, 2**(n_data + n_ancilla))

    Returns:
        normalized_state: Normalized pure state vector of shape (batch_size, 2**n_data)
    """
    batch_size = state.shape[0]

    # Reshape state to (batch_size, 2**n_data, 2**n_ancilla)
    state_reshaped = state.reshape((batch_size, 2**model.n_data, 2**model.n_ancilla))

    # Calculate probabilities for each ancilla outcome (summing over data dimensions)
    probs = jnp.sum(
        jnp.abs(state_reshaped) ** 2, axis=1
    )  # shape: (batch_size, 2**n_ancilla)

    # Sample outcomes (detach from AD to treat as constants)
    sampled_indices = jax.lax.stop_gradient(
        jax.random.categorical(key, jnp.log(probs + 1e-12))
    )

    # Differentiable slice: select the collapsed state for each item in the batch
    batch_indices = jnp.arange(batch_size)
    gen_sampled = state_reshaped[
        batch_indices, :, sampled_indices
    ]  # shape: (batch_size, 2**n_data)

    # Differentiable normalization
    norms = jnp.linalg.norm(gen_sampled, axis=-1, keepdims=True)

    # Return normalized pure state
    return gen_sampled / (norms + 1e-12)
