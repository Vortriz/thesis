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
