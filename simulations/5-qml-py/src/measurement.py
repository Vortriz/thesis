import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


def measure_stochastic(
    rng_key: PRNGKeyArray, state: Array, n_data: int, n_ancilla: int
) -> Array:
    """
    Simulates a stochastic measurement of the ancilla qubits and returns
    the collapsed, normalized pure state for the data qubits.

    Data qubits are assumed to be the first `n_data` wires.
    Ancilla qubits are assumed to be the last `n_ancilla` wires.

    Args:
        state: State vector of shape (batch_size, 2**(n_data + n_ancilla))
        rng_key: JAX PRNG key
        n_data: Number of data qubits
        n_ancilla: Number of ancilla qubits

    Returns:
        normalized_state: Normalized pure state vector of shape (batch_size, 2**n_data)
    """
    batch_size = state.shape[0]

    # Reshape state to (batch_size, 2**n_data, 2**n_ancilla)
    state_reshaped = state.reshape((batch_size, 2**n_data, 2**n_ancilla))

    # Calculate probabilities for each ancilla outcome (summing over data dimensions)
    probs = jnp.sum(
        jnp.abs(state_reshaped) ** 2, axis=1
    )  # shape: (batch_size, 2**n_ancilla)

    # Sample outcomes (detach from AD to treat as constants)
    sampled_indices = jax.lax.stop_gradient(
        jax.random.categorical(rng_key, jnp.log(probs + 1e-12))
    )

    # Differentiable slice: select the collapsed state for each item in the batch
    batch_indices = jnp.arange(batch_size)
    gen_sampled = state_reshaped[
        batch_indices, :, sampled_indices
    ]  # shape: (batch_size, 2**n_data)

    # Differentiable normalization
    norms = jnp.sqrt(jnp.sum(jnp.abs(gen_sampled) ** 2, axis=-1, keepdims=True))

    # Return normalized pure state
    return gen_sampled / (norms + 1e-12)
