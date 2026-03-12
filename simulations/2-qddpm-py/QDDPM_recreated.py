import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import h5py
    import marimo as mo
    import numpy as np
    import torch
    from QDDPM import (
        BackwardProcess,
        ForwardProcess,
        LossFunction,
        animate_bloch_sphere,
        plot_bloch_sphere,
        plot_forward_fidelity_decay,
        plot_loss_evaluation_vs_initial,
        plot_loss_training_vs_initial,
    )

    rng = np.random.default_rng(1234)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1 qubit
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Forward Process
    """)
    return


@app.cell
def _():
    fp1 = ForwardProcess(
        n_qubits=1,
        n_forward_samples=1000,
        scale=0.08,
        T=20,
        schedule=np.linspace(0.5, 2.75, 20),
    )
    fp1.scramble(rng=rng)
    return (fp1,)


@app.cell
def _(fp1):
    fp1.save(mo.notebook_dir() / "data" / "1qubit_forward_process.h5")
    return


@app.cell(hide_code=True)
def _(fp1):
    mo.md(f"""
    We generate `n_sample={fp1.n_forward_samples}` samples each with `n_qubits={fp1.n_qubits}` such that the amplitude of the all-zeros basis state is dominant and all other basis states have random amplitudes scaled by a factor of `scale={fp1.scale}`.
    """)
    return


@app.cell(hide_code=True)
def _(fp1):
    mo.md(f"""
    Then, we apply a scrambling circuit for `T={fp1.T}` timesteps. The circuit consists of random local rotations and entangling gates. The strength of the scrambling at each timestep is controlled by the `schedule`.
    """)
    return


@app.cell(hide_code=True)
def _(fp1):
    mo.md(f"""
    Finally, we plot the evolution of the average pairwise fidelity of the ensemble of states over time. The fidelity is expected to decay to the fidelity of a Haar-random ensemble, which is `{1 / 2**fp1.n_qubits}`.
    """)
    return


@app.cell
def _(fp1):
    mo.mpl.interactive(plot_forward_fidelity_decay(fp1.forward_states)).style(
        width="650px",
    ).center()
    return


@app.cell(hide_code=True)
def _(fp1):
    slider = mo.ui.slider(
        start=0,
        stop=fp1.T,
        show_value=True,
        debounce=True,
        disabled=fp1.n_qubits != 1,
        label="t =",
    )
    video = animate_bloch_sphere(fp1.forward_states).to_html5_video()
    return slider, video


@app.cell(hide_code=True)
def _(fp1, slider, video):
    mo.hstack(
        [
            mo.vstack(
                [
                    slider,
                    mo.mpl.interactive(
                        plot_bloch_sphere(fp1.forward_states[slider.value]),
                    ).style(
                        width="510px",
                    ),
                ],
                align="center",
            ),
            mo.Html(video).center(),
        ],
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Backward Process
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We only take the 1st column of each unitary because

    $$
    U \ket{0}
    = \begin{bmatrix} u_{11} & u_{12} \\ u_{21} & u_{22} \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix}
    = \underbrace{\begin{bmatrix} u_{11} \\ u_{21} \end{bmatrix}}_{\text{i.e. first column}}
    $$
    """)
    return


@app.cell
def _(fp1):
    bp1 = BackwardProcess().from_forward_process(
        fp1,
        n_ancilla=1,
        n_layers=4,
        n_backward_samples=100,
        epochs=2000,
        loss_function=LossFunction.WASSERSTEIN,
    )
    bp1.run_training(rng)
    return (bp1,)


@app.cell
def _(bp1):
    bp1.save(mo.notebook_dir() / "data" / "1qubit_backward_process.h5")
    return


@app.cell(hide_code=True)
def _(bp1):
    mo.md(f"""
    needs only {torch.numel(bp1.trained_params)} params for {bp1.n_qubits} qubit(s) + {bp1.n_ancilla} ancilla qubit(s).
    """)
    return


@app.cell
def _():
    with h5py.File(mo.notebook_dir() / "data" / "1qubit_forward_process.h5", "r") as f1:
        forward_states1 = f1["forward_states"][:]
    bp1_loaded = BackwardProcess.from_trained_params(
        mo.notebook_dir() / "data" / "1qubit_backward_process.h5",
    )
    return bp1_loaded, forward_states1


@app.cell
def _(bp1_loaded):
    bp1_loaded.denoising_loss_hist[:, -2]
    return


@app.cell
def _(bp1_loaded):
    plot_loss_training_vs_initial(
        bp1_loaded.denoising_loss_hist,
        title="Loss distance of Training Ensemble",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Evaluation
    """)
    return


@app.cell
def _(bp1_loaded, forward_states1):
    plot_loss_evaluation_vs_initial(
        LossFunction.WASSERSTEIN,
        bp1_loaded.run(state=rng),
        forward_states1[0],
        title="Loss distance of Evaluated Ensemble (from Training Set) w.r.t. Initial Ensemble",
    )
    return


@app.cell
def _(bp1_loaded, forward_states1):
    plot_loss_evaluation_vs_initial(
        LossFunction.WASSERSTEIN,
        bp1_loaded.run(state=24),
        forward_states1[0],
        title="Loss distance of Evaluated Ensemble (from Testing Set) w.r.t. Initial Ensemble",
    )
    return


@app.cell
def _(bp1_loaded):
    mo.mpl.interactive(
        plot_bloch_sphere(bp1_loaded.run(state=2)[0]),
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import qutip

    return plt, qutip


@app.cell
def _(qutip):
    def bloch_helper_alt(ax, sphere, samples):
        sphere.clear()
        points = np.zeros(shape=(samples.shape[0], 3))

        for i in range(samples.shape[0]):
            state = samples[i][0] * qutip.basis(2, 0) + samples[i][1] * qutip.basis(
                2, 1,
            )
            points[i] = [
                qutip.expect(qutip.sigmax(), state),
                qutip.expect(qutip.sigmay(), state),
                qutip.expect(qutip.sigmaz(), state),
            ]

        for i in range(len(points) - 1):
            sphere.add_arc(points[i], points[i + 1])
        sphere.add_points(points[1:-1].T)
        sphere.add_points(points[0])
        sphere.add_points(points[-1])
        sphere.point_marker = ["o"]
        sphere.render()

        return ax

    return (bloch_helper_alt,)


@app.cell
def _(bloch_helper_alt, plt, qutip):
    def plot_bloch_sphere_alt(samples, rev=False):
        mo.stop(
            samples.shape[1] != 2,
            mo.md(
                "Bloch sphere visualization is only available for single-qubit states.",
            ),
        )

        fig = plt.figure()
        ax = fig.add_subplot(azim=-40, elev=30, projection="3d")
        sphere = qutip.Bloch(axes=ax)

        if rev:
            samples = torch.flip(samples, dims=(0,))
        ax = bloch_helper_alt(ax, sphere, samples)
        return fig

    return (plot_bloch_sphere_alt,)


@app.cell
def _(forward_states1, plot_bloch_sphere_alt):
    mo.mpl.interactive(
        plot_bloch_sphere_alt(
            torch.from_numpy(forward_states1[:, 0, :]),
            rev=False,
        ),
    )
    return


@app.cell
def _(bp1_loaded, plot_bloch_sphere_alt):
    mo.mpl.interactive(
        plot_bloch_sphere_alt(
            bp1_loaded.run(state=rng)[:, 6, :],
            rev=True,
        ),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2 qubits
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Forward Process
    """)
    return


@app.cell
def _():
    fp2 = ForwardProcess(
        n_qubits=2,
        n_forward_samples=1000,
        scale=0.05,
        T=8,
        schedule=np.linspace(0.3, 4, 10),
    )
    fp2.scramble(rng=rng)
    return (fp2,)


@app.cell
def _(fp2):
    fp2.save(mo.notebook_dir() / "data" / f"{fp2.n_qubits}qubit_forward_process.h5")
    return


@app.cell
def _(fp2):
    mo.mpl.interactive(plot_forward_fidelity_decay(fp2.forward_states)).style(
        width="650px",
    ).center()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Backward Process
    """)
    return


@app.cell
def _(fp2):
    bp2 = BackwardProcess().from_forward_process(
        fp2,
        n_ancilla=1,
        n_layers=6,
        n_backward_samples=100,
        epochs=1000,
        loss_function=LossFunction.WASSERSTEIN,
    )
    bp2.run_training(rng)
    return (bp2,)


@app.cell
def _(bp2):
    bp2.save(mo.notebook_dir() / "data" / f"{bp2.n_qubits}qubit_backward_process.h5")
    return


@app.cell
def _():
    with h5py.File(mo.notebook_dir() / "data" / "2qubit_forward_process.h5", "r") as f2:
        forward_states2 = f2["forward_states"][:]
    bp2_loaded = BackwardProcess.from_trained_params(
        mo.notebook_dir() / "data" / "2qubit_backward_process.h5",
    )
    return bp2_loaded, forward_states2


@app.cell
def _(bp2_loaded):
    plot_loss_training_vs_initial(
        bp2_loaded.denoising_loss_hist,
        title="MMD of Training Ensemble with respect to Initial Ensemble",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Evaluation
    """)
    return


@app.cell
def _(bp2_loaded, forward_states2):
    plot_loss_evaluation_vs_initial(
        LossFunction.WASSERSTEIN,
        bp2_loaded.run(state=rng),
        forward_states2[0],
        title="MMD of Evaluated Ensemble (from Training Set) w.r.t. Initial Ensemble",
    )
    return


@app.cell
def _(bp2_loaded, forward_states2):
    plot_loss_evaluation_vs_initial(
        LossFunction.WASSERSTEIN,
        bp2_loaded.run(state=24),
        forward_states2[0],
        title="MMD of Evaluated Ensemble (from Testing Set) w.r.t. Initial Ensemble",
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
