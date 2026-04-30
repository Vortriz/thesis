import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import qutip
from jax import Array


def plot_bloch_sphere(ensemble: Array):
    mo.stop(
        ensemble.shape[1] != 2,
        mo.md("Bloch sphere visualization is only available for single-qubit states."),
    )

    fig = plt.figure()
    ax = fig.add_subplot(azim=-40, elev=30, projection="3d")
    sphere = qutip.Bloch(axes=ax)

    points = []

    for i in range(len(ensemble)):
        state = ensemble[i][0] * qutip.basis(2, 0) + ensemble[i][1] * qutip.basis(2, 1)
        points.append(
            [
                qutip.expect(qutip.sigmax(), state),
                qutip.expect(qutip.sigmay(), state),
                qutip.expect(qutip.sigmaz(), state),
            ],
        )

    points_transposed = np.array(points).T.tolist()

    sphere.add_points(points_transposed)
    sphere.point_size = [3]
    sphere.render()

    return fig


def plot_loss_training_vs_initial(loss_history, title):
    """
    Plots the loss history for each timestep sequentially.
    """
    if isinstance(loss_history, list):
        loss_history = np.array(loss_history)

    T, epochs = loss_history.shape
    losses_flat = np.zeros(shape=T * epochs)

    fig, ax = plt.subplots(figsize=(12, 6))

    for t, losses in enumerate(loss_history):
        losses_flat[t * epochs : (t + 1) * epochs] = losses

    ax.scatter(np.arange(T * epochs), losses_flat, s=1)
    ax.hlines(
        losses_flat[-1],
        0,
        1,
        transform=ax.get_yaxis_transform(),
        linestyle="--",
        label=f"Final Loss = {losses_flat[-1]:.4f}",
    )
    ax.set_ylim(0, 1)

    ax.set_xlabel("Total Epochs (T x epochs)")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title}")
    plt.legend()
    return fig
