from enum import Enum
from typing import Literal

import jax.numpy as jnp
from jax import random
from jaxtyping import Array, PRNGKeyArray
from plum import dispatch


class DistributionType(Enum):
    CLUSTERED = "clustered"
    HAAR = "haar"
    QKR = "qkr"


# [TODO] use nnx.Rngs


@dispatch
def gen_dist(
    _: Literal[DistributionType.CLUSTERED],
    key: PRNGKeyArray,
    n_qubits: int,
    n_samples: int,
    scale: float = 0.05,
) -> Array:
    """Generates enemble of states clustered around a random base state."""
    basis_size = 2**n_qubits
    k1, k2 = random.split(key)
    base_state = random.normal(k1, (1, basis_size), dtype=jnp.complex128)

    states = jnp.repeat(base_state, n_samples, axis=0) + scale * random.normal(
        k2, (n_samples, basis_size), dtype=jnp.complex128
    )

    states /= jnp.linalg.norm(states, axis=1, keepdims=True)

    return states


@dispatch
def gen_dist(  # noqa: F811
    _: Literal[DistributionType.HAAR],
    key: PRNGKeyArray,
    n_qubits: int,
    n_samples: int,
) -> Array:
    """Generates Haar-random ensemble using normalized complex Gaussian vectors."""
    dims = 2**n_qubits
    k1, k2 = random.split(key)

    # A normalized complex Gaussian vector is Haar-random (statistically equivalent to first column of QR)
    z = random.normal(k1, (n_samples, dims)) + 1j * random.normal(k2, (n_samples, dims))

    return z / jnp.linalg.norm(z, axis=1, keepdims=True)


# @dispatch
# def gen_dist(
#     _: Literal[DistributionType.QKR],
#     n_qubits: int,
#     n_samples: int
# ) -> Array:
#     """Generates momentum eigenstates of the Quantum Kicked Rotor (QKR) model."""
#     # [TODO]
#     return
