import jax
import jax.numpy as jnp
from jaxtyping import Array
from jax import jit
from functools import partial


@jit
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


@partial(jit, static_argnames=("max_iter", "L"))
def ipot(
    C: Array,
    beta: float = 0.01,
    max_iter: int = 100,
    L: int = 1,
) -> Array:
    """
    Inexact Proximal Optimal Transport (IPOT) algorithm.
    Optimized for JAX using `lax.scan` and inner loop unrolling.
    """
    n1, n2 = C.shape
    a1 = jnp.ones(n1) / n1
    a2 = jnp.ones(n2) / n2

    K = jnp.exp(-C / beta)

    def scan_body(carry, _):
        P, u, v = carry
        Q = K * P

        # Unroll the inner L loop since it's static (typically 1)
        # This allows JAX to heavily optimize the sequential updates
        for _ in range(L):
            # We use a very small epsilon (1e-200) to prevent division by zero,
            # but small enough not to swallow values like exp(-100) ~ 1e-44
            u = a1 / (jnp.dot(Q, v) + 1e-200)
            v = a2 / (jnp.dot(Q.T, u) + 1e-200)

        # Broadcasting is highly optimized in XLA
        P = u[:, None] * Q * v[None, :]
        return (P, u, v), None

    P_init = jnp.ones((n1, n2)) / (n1 * n2)
    u_init = jnp.ones(n1)
    v_init = jnp.ones(n2)

    (P_final, _, _), _ = jax.lax.scan(scan_body, (P_init, u_init, v_init), None, length=max_iter)
    
    return P_final


@partial(jit, static_argnames=("max_iter", "L", "return_map"))
def wasserstein_distance(
    ensemble1: Array,
    ensemble2: Array,
    beta: float = 0.01,
    max_iter: int = 1000,
    L: int = 1,
    return_map: bool = False,
):
    """
    Calculates the Wasserstein distance between two ensembles of quantum states using IPOT.
    The cost matrix is the inter-trace distance: C_ij = 1 - |<psi_i | phi_j>|^2.
    """
    # Cost matrix calculation (highly vectorized matrix multiplication)
    C = 1.0 - jnp.abs(jnp.matmul(ensemble1, ensemble2.conj().T)) ** 2

    P = ipot(C, beta=beta, max_iter=max_iter, L=L)

    if return_map:
        return P

    # By the envelope theorem, the gradient of the OT loss with respect to C
    # is exactly the optimal transport plan P. Stopping gradients through P 
    # avoids backpropagating through the `max_iter` steps of the IPOT solver, 
    # dramatically speeding up training without altering the correct gradients.
    P = jax.lax.stop_gradient(P)

    return jnp.sum(P * C)
