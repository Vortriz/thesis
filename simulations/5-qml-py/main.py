import numpy as np
import jax
import jax.numpy as jnp
from src.types import Model

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


# 1. Configuration
config = Model(
    n_data=2,
    n_ancilla=1,
    n_layers=4,
    batch_size=10,
    learning_rate=0.01,
    epoch_schedule=[100] * 5,
)
rng_key = jax.random.key(42)
stateful_rng = np.random.default_rng(int(jax.random.bits(rng_key)))
