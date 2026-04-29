import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from beartype import beartype as typechecker
from flax.struct import dataclass, field
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped
from numpy.random import Generator


@jaxtyped(typechecker=typechecker)
@dataclass
class Model:
    # Metadata fields: Marked as static
    n_data: int = field(pytree_node=False)
    n_ancilla: int = field(pytree_node=False)
    n_qubits: int = field(pytree_node=False)

    T: int = field(pytree_node=False)
    # Number of layers per block in the ansatz
    n_layers: int = field(pytree_node=False)

    # Total number of samples in the training dataset
    dataset_size: int = field(pytree_node=False)
    # Number of samples processed in one epoch
    batch_size: int = field(pytree_node=False)

    # (T,) array: target index for each training step
    target_schedule: Int[Array, "T"] = field(pytree_node=False)
    # (T,) array: epochs per block
    epoch_schedule: Int[Array, "T"] = field(pytree_node=False)

    params: Float[Array, "T n_layers n_qubits 2"]

    key: PRNGKeyArray = field(pytree_node=False)

    def __post_init__(self):
        if self.target_schedule.shape[0] != self.epoch_schedule.shape[0]:
            raise ValueError(
                f"step_indices (len={self.target_schedule.shape[0]}) and "
                f"epoch_schedule (len={self.epoch_schedule.shape[0]}) "
                "must have the same length."
            )

    @classmethod
    def create(
        cls,
        n_data,
        n_ancilla,
        n_layers,
        dataset_size,
        batch_size,
        target_schedule,
        epoch_schedule,
        key,
    ):
        T = target_schedule.shape[0]
        n_qubits = n_data + n_ancilla
        params = random.normal(key, (T, n_layers, n_qubits, 2))

        return cls(
            n_data=n_data,
            n_ancilla=n_ancilla,
            n_qubits=n_qubits,
            T=T,
            n_layers=n_layers,
            dataset_size=dataset_size,
            batch_size=batch_size,
            target_schedule=target_schedule,
            epoch_schedule=epoch_schedule,
            params=params,
            key=key,
        )
