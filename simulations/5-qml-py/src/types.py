from beartype import beartype as typechecker
from flax import nnx
from jaxtyping import Array, Int, jaxtyped


@jaxtyped(typechecker=typechecker)
class Model(nnx.Module):
    """
    Model class for QML simulations using Flax NNX.
    """

    def __init__(
        self,
        n_data: int,
        n_ancilla: int,
        n_layers: int,
        dataset_size: int,
        batch_size: int,
        # [TODO] fix type checking for array (pehaps nnx needs more settings)
        target_schedule: Int[Array, "T"],
        epoch_schedule: Int[Array, "T"],
        rngs: nnx.Rngs,
    ):
        # Metadata fields: Marked as static in NNX
        self.n_data = n_data
        self.n_ancilla = n_ancilla
        self.n_qubits = n_data + n_ancilla
        self.T = target_schedule.shape[0]
        self.n_layers = n_layers
        self.dataset_size = dataset_size
        self.batch_size = batch_size

        # (T,) arrays: target index and epochs per block
        self.target_schedule = target_schedule
        self.epoch_schedule = epoch_schedule

        # Parameters: (T, n_layers, n_qubits, 2)
        # Using rngs.normal for convenience as per user sample
        self.params = nnx.Param(rngs.normal((self.T, n_layers, self.n_qubits, 2)))
