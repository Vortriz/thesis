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


@dispatch
def gen_dist(
    _: Literal[DistributionType.CLUSTERED],
    key: PRNGKeyArray,
    n_qubits: int,
    n_samples: int,
    scale: float = 0.05,
) -> Array:
    """Generates states clustered around a random base state."""
    basis_size = 2**n_qubits
    k1, k2, k3 = random.split(key, 3)
    base_state = random.uniform(k1, (1, basis_size))

    states = jnp.repeat(base_state, n_samples, axis=0) + scale * (
        random.normal(k2, (n_samples, basis_size))
        + 1j * random.normal(k3, (n_samples, basis_size))
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
    """Generates Haar-random states"""
    dims = 2**n_qubits

    # Generate complex Gaussian matrices (Ginibre ensemble)
    k1, k2 = random.split(key)
    z = (
        random.normal(k1, (n_samples, dims, dims))
         + 1j * random.normal(k2, (n_samples, dims, dims))
    ) / jnp.sqrt(2.0)

    # JAX's qr handles the leading 'n' dimension automatically
    q, r = jnp.linalg.qr(z)

    diag_r = jnp.diagonal(r, axis1=-2, axis2=-1)

    # Compute the phase: e^(i * theta) = r_ii / |r_ii|
    phases = diag_r / jnp.abs(diag_r)

    # Correct Q: multiply each column i by phase i
    # (n, dim, dim) * (n, 1, dim) -> broadcasting takes care of columns
    u = q * phases[:, jnp.newaxis, :]

    return u[:, :, 0]


# @dispatch
# def gen_dist(
#     _: Literal[DistributionType.QKR],
#     n_qubits: int,
#     n_samples: int
# ) -> Array:
#     """Generates momentum eigenstates of the Quantum Kicked Rotor (QKR) model."""
#     # [TODO]
#     return
