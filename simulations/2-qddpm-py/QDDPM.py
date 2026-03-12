import itertools
import warnings
from enum import Enum, auto
from pathlib import Path
from typing import no_type_check

import h5py
import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import ot
import qutip
import tensorcircuit as tc
import torch
from matplotlib import animation
from matplotlib.ticker import MultipleLocator
from numpy import pi
from scipy.stats import unitary_group
from torch import nn

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning, module="keras")

progress = mo.status.progress_bar
tc.set_dtype("complex128")
K = tc.set_backend("pytorch")


def bloch_helper(ax, sphere, ensemble):
    sphere.clear()
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
    sphere.point_size = [5]
    sphere.render()

    return ax


def plot_bloch_sphere(ensemble):
    mo.stop(
        ensemble.shape[1] != 2,
        mo.md("Bloch sphere visualization is only available for single-qubit states."),
    )

    fig = plt.figure()
    ax = fig.add_subplot(azim=-40, elev=30, projection="3d")
    sphere = qutip.Bloch(axes=ax)

    ax = bloch_helper(ax, sphere, ensemble)
    return fig


def animate_bloch_sphere(forward_states):
    mo.stop(
        forward_states.shape[2] != 2,
        mo.md("Bloch sphere visualization is only available for single-qubit states."),
    )

    T = forward_states.shape[0] - 1

    fig = plt.figure()
    ax = fig.add_subplot(azim=-40, elev=30, projection="3d")
    sphere = qutip.Bloch(axes=ax)

    def animate(t):
        return bloch_helper(ax, sphere, forward_states[t])

    return animation.FuncAnimation(fig, animate, range(T + 1), blit=False, repeat=False)


def plot_forward_fidelity_decay(forward_states):
    """
    Calculates and plots the fidelity decay over time from the |0...0> state.

    Parameters
    ----------
    forward_states : np.ndarray, shape (T + 1, n_forward_samples, 2**n_qubits)
        The forward time evolution of the quantum states.
    """
    T = forward_states.shape[0] - 1
    n_basis_states = forward_states.shape[2]

    fidelity_evolution = np.zeros(shape=T + 1)
    for t in range(T + 1):
        mean_fidelity = np.mean(np.abs(forward_states[t][:, 0]) ** 2)
        fidelity_evolution[t] = mean_fidelity

    plt.figure()
    plt.plot(range(T + 1), fidelity_evolution, marker="o")
    plt.plot(
        [0, T],
        [1 / n_basis_states] * 2,
        "g--",
        alpha=0.3,
        label="Ideal Haar random fidelity",
    )
    plt.xticks(np.arange(0, T + 1, 2))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylim(0, 1)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Fidelity")
    plt.title("Fidelity Evolution")

    return plt.gcf()


class LossFunction(Enum):
    MMD = auto()
    WASSERSTEIN = auto()


def mmd_distance(ensemble1, ensemble2):
    """
    Calculates the MMD distance between two ensembles of quantum states.
    """
    r11 = 1.0 - torch.mean(
        torch.abs(torch.einsum("mi,ni->mn", ensemble1.conj(), ensemble1)).pow(2.0),
    )
    r22 = 1.0 - torch.mean(
        torch.abs(torch.einsum("mi,ni->mn", ensemble2.conj(), ensemble2)).pow(2.0),
    )
    r12 = 1.0 - torch.mean(
        torch.abs(torch.einsum("mi,ni->mn", ensemble1.conj(), ensemble2)).pow(2.0),
    )
    return 2 * r12 - r11 - r22


def wasserstein_distance(ensemble1, ensemble2):
    """
    Calculates the Wasserstein distance between two ensembles of quantum states.
    The cost matrix is the inter-trace distance between ensembles.
    """
    D = 1.0 - torch.abs(ensemble1.conj() @ ensemble2.T).pow(2.0)
    emt = torch.empty(0)
    return ot.emd2(emt, emt, M=D)


def ema(data, span):
    """
    Computes the Exponential Moving Average (EMA) of data.
    """
    if len(data) == 0:
        return np.array([])
    alpha = 2 / (span + 1.0)
    ema_data = np.zeros_like(data, dtype=float)
    ema_data[0] = data[0]
    for i in range(1, len(data)):
        ema_data[i] = alpha * data[i] + (1 - alpha) * ema_data[i - 1]
    return ema_data


def plot_loss_training_vs_initial(loss_history, title, ema_span=100):
    """
    Plots MMD/Wasserstein loss of training states with respect to initial states (uses EMA smoothing).
    """
    T, epochs = loss_history.shape
    ema_losses = np.zeros(shape=T * epochs)
    fig, ax = plt.subplots(figsize=(12, 6))
    for t, losses in enumerate(np.flip(loss_history, axis=0)):
        ema_losses[t * epochs : (t + 1) * epochs] = ema(losses, span=ema_span)

    ax.scatter(np.arange(T * epochs), ema_losses, s=1)
    ax.hlines(
        ema_losses[-1],
        0,
        1,
        transform=ax.get_yaxis_transform(),
        linestyle="--",
        label=f"Final EMA Loss = {ema_losses[-1]:.4f}",
    )
    ax.set_xlabel("Total Epochs (T x epochs)")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title} (EMA, span={ema_span})")
    plt.legend()
    return fig


def plot_loss_evaluation_vs_initial(
    loss_function: LossFunction,
    evaluated_states_history,
    initial_states,
    title,
):
    """
    Plots MMD/Wasserstein loss of evaluated states (with trained parameters) with respect to the initial states.
    """
    if isinstance(initial_states, np.ndarray):
        initial_states = torch.from_numpy(initial_states)

    loss_callable = (
        mmd_distance if loss_function == LossFunction.MMD else wasserstein_distance
    )
    loss_history = [
        loss_callable(states, initial_states)
        for states in torch.flip(evaluated_states_history, [0])
    ]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()
    ax.plot(range(len(loss_history)), loss_history, marker="o")
    ax.hlines(
        loss_history[-1],
        0,
        1,
        transform=ax.get_yaxis_transform(),
        linestyle="--",
        label=f"Final Loss = {loss_history[-1]:.4f}",
    )
    ax.set_xlabel("Time step")
    ax.set_xticks(np.arange(len(loss_history)))
    ax.set_ylabel("Evaluation Loss")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    plt.title(title)
    plt.legend()
    return fig


class ForwardProcess:
    """Manages the forward diffusion process for a quantum state.

    This class handles the generation of initial quantum states and their
    subsequent evolution under a scrambling unitary process.

    Attributes
    ----------
    n_qubits : int
        The number of qubits in the quantum system.
    n_forward_samples : int
        The number of sample states to generate and evolve.
    scale : float
        A scaling factor for the initial state amplitudes.
    T : int
        The total number of time steps for the diffusion process.
    schedule : np.ndarray, shape (T,)
        The schedule of weights for the scrambling gates.
    initial_forward_states : np.ndarray, shape (n_forward_samples, 2**n_qubits) or None
        The generated initial states.
    forward_states : np.ndarray, shape (T + 1, n_forward_samples, 2**n_qubits) or None
        The forward time evolution of the quantum states towards noise.
    fidelity_evolution : list of float, or None
        The mean fidelity at each time step.
    """

    def __init__(self, n_qubits, n_forward_samples, scale, T, schedule):
        """Initializes the ForwardProcess.

        Parameters
        ----------
        n_qubits : int
            The number of qubits in the quantum system.
        n_forward_samples : int
            The number of sample states to generate and evolve.
        scale : float
            A scaling factor for the initial state amplitudes, biasing
            them towards the |0...0> state.
        T : int
            The total number of time steps for the diffusion process.
        schedule : np.ndarray, shape (T,)
            The schedule of weights for the scrambling gates.
        """
        self.n_qubits = n_qubits
        self.n_forward_samples = n_forward_samples
        self.scale = scale
        self.T = T
        self.schedule = schedule

        self.initial_forward_states = None
        self.forward_states = None
        self.fidelity_evolution = None

    def _gen_initial_forward_states(self, rng):
        """Generates initial quantum states biased towards the |0...0> state.

        Parameters
        ----------
        rng : np.random.Generator
            The random number generator.
        """
        base_state = rng.random(2**self.n_qubits)

        states = np.zeros((self.n_forward_samples, 2**self.n_qubits), dtype=np.complex128)
        for i in range(self.n_forward_samples):
            perturbation = rng.standard_normal(2**self.n_qubits) + 1j * rng.standard_normal(2**self.n_qubits)
            states[i] = base_state + self.scale * perturbation

        normalized_states = states / np.linalg.norm(states, axis=1, keepdims=True)

        self.initial_forward_states = normalized_states

    @no_type_check
    def scramble(self, rng):
        """Applies a Fast Scrambling method to diffuse the states.

        The scrambling consists of random local rotations and entangling gates.
        The results are stored in the `self.forward_states` attribute.

        Parameters
        ----------
        rng : np.random.Generator
            The random number generator.
        """
        self._gen_initial_forward_states(rng)

        self.forward_states = np.zeros(
            shape=(self.T + 1, self.n_forward_samples, 2**self.n_qubits),
            dtype=np.complex128,
        )
        self.forward_states[0] = self.initial_forward_states

        for t in progress(range(self.T), title="Scrambling...", remove_on_exit=True):
            for s in range(self.n_forward_samples):
                c = tc.Circuit(self.n_qubits, inputs=self.forward_states[t, s])

                all_qubits = range(self.n_qubits)

                weight = self.schedule[t]

                # euler angles for local scrambling
                c.rx(
                    all_qubits,
                    theta=weight * (rng.random(self.n_qubits) * pi / 4 - pi / 8),
                )
                c.ry(
                    all_qubits,
                    theta=weight * (rng.random(self.n_qubits) * pi / 4 - pi / 8),
                )
                c.rz(
                    all_qubits,
                    theta=weight * (rng.random(self.n_qubits) * pi / 4 - pi / 8),
                )

                # entangling gates
                if self.n_qubits > 1:
                    combinations = list(itertools.combinations(all_qubits, 2))

                    # angles between [0.4, 0.6] radians
                    angles_r2 = rng.random(len(combinations)) * 0.2 + 0.4
                    for p, (i, j) in enumerate(combinations):
                        (
                            c.rzz(
                                i,
                                j,
                                theta=(weight * angles_r2[p])
                                / (2 * np.sqrt(self.n_qubits)),
                            )
                        )

                self.forward_states[t + 1, s] = c.state()

        return

    def save(self, filename: str | Path):
        """Saves the forward process data to an HDF5 file.

        The file will contain the `forward_states` dataset and simulation
        parameters as attributes.

        Parameters
        ----------
        filename : str or Path
            The path to the HDF5 file where the data will be saved.

        Raises
        ------
        ValueError
            If the forward states have not been generated by running `scramble` first.
        """
        if self.forward_states is None:
            raise ValueError(
                "Forward states have not been generated. Run `scramble()` first.",
            )

        with h5py.File(filename, "w") as f:
            f.create_dataset("forward_states", data=self.forward_states)
            f.attrs["n_qubits"] = self.n_qubits
            f.attrs["n_forward_samples"] = self.n_forward_samples
            f.attrs["scale"] = self.scale
            f.attrs["T"] = self.T
            f.attrs["schedule"] = self.schedule


class BackwardProcess(nn.Module):
    """Manages the backward diffusion (denoising) process for a quantum state.

    This class learns to reverse the forward diffusion process by training a
    series of parameterized quantum circuits, one for each timestep. It can then
    use these trained circuits to generate new quantum states from a random
    distribution.

    Attributes
    ----------
    n_qubits : int
        The number of qubits in the main quantum system.
    n_ancilla : int
        The number of ancilla qubits used in the denoising circuit.
    n_total : int
        The total number of qubits (main + ancilla).
    T : int
        The total number of time steps for the diffusion process.
    n_layers : int
        The number of layers in the variational quantum circuit.
    n_backward_samples : int
        The number of samples to use in each training batch and for generation.
    epochs : int
        The number of training epochs for each timestep's model.
    _back_circuit_vmap : function
        A vmapped version of the `_back_circuit` method for batch processing.
    forward_states : torch.Tensor
        The states from the forward process, used as ground truth for training.
    trained_params : torch.Tensor, shape (T, n_layers, 2 * n_total) or None
        The trained parameters for the denoising circuit at each timestep.
    denoising_loss_hist : np.ndarray, shape (T, epochs) or None
        The MMD/Wasserstein loss history during training.
    """

    def __init__(
        self,
        n_qubits=None,
        n_ancilla=None,
        T=None,
        n_layers=None,
        n_backward_samples=None,
        epochs=None,
        loss_function: LossFunction | None = None,
        forward_states=None,
        trained_params=None,
        denoising_loss_hist=None,
    ):
        """Initializes the BackwardProcess."""
        super().__init__()
        self.n_qubits = n_qubits
        self.n_ancilla = n_ancilla
        self.n_total = (
            n_qubits + n_ancilla
            if n_qubits is not None and n_ancilla is not None
            else None
        )
        self.T = T
        self.n_layers = n_layers
        self.n_backward_samples = n_backward_samples
        self.epochs = epochs
        self.loss_function = loss_function

        if forward_states is not None:
            self.forward_states = torch.from_numpy(forward_states)
        else:
            self.forward_states = None

        if trained_params is not None and isinstance(trained_params, np.ndarray):
            self.trained_params = torch.from_numpy(trained_params)
        else:
            self.trained_params = trained_params

        self.denoising_loss_hist = denoising_loss_hist

        if self.n_total is not None:
            self._back_circuit_vmap = K.vmap(self._back_circuit, vectorized_argnums=0)
        else:
            self._back_circuit_vmap = None

    @classmethod
    def from_forward_process(
        cls,
        forward_process_source: ForwardProcess | str | Path,
        n_ancilla,
        n_layers,
        n_backward_samples,
        epochs,
        loss_function: LossFunction,
    ):
        """Creates a BackwardProcess instance from a forward process.

        Parameters
        ----------
        forward_process_source : ForwardProcess or str or Path
            Either a ForwardProcess instance or the path to an HDF5 file
            containing the forward process data.
        n_ancilla : int
            The number of ancilla qubits to use in the denoising circuit.
        n_layers : int
            The number of layers in the variational quantum circuit.
        n_backward_samples : int
            The number of samples for batching during training and generation.
        epochs : int
            The number of training epochs for each timestep's model.
        loss_function : LossFunction
            The loss function to use for training (MMD or Wasserstein).

        Returns
        -------
        BackwardProcess
            An initialized BackwardProcess instance.
        """
        if isinstance(forward_process_source, (str, Path)):
            with h5py.File(forward_process_source, "r") as f:
                forward_states = f["forward_states"][:]
                n_qubits = f.attrs["n_qubits"]
                T = f.attrs["T"]
        elif isinstance(forward_process_source, ForwardProcess):
            if forward_process_source.forward_states is None:
                raise ValueError(
                    "Forward states have not been generated. Run `scramble()` first.",
                )
            forward_states = forward_process_source.forward_states
            n_qubits = forward_process_source.n_qubits
            T = forward_process_source.T
        else:
            raise TypeError(
                "forward_process_source must be a ForwardProcess instance or a file path.",
            )

        return cls(
            n_qubits=n_qubits,
            T=T,
            forward_states=forward_states,
            n_ancilla=n_ancilla,
            n_layers=n_layers,
            n_backward_samples=n_backward_samples,
            epochs=epochs,
            loss_function=loss_function,
        )

    @classmethod
    def from_trained_params(cls, filepath: str | Path):
        """Loads a BackwardProcess instance from a file with trained parameters for generation.

        Parameters
        ----------
        filepath : str or Path
            The path to the HDF5 file containing the trained parameters.

        Returns
        -------
        BackwardProcess
            An initialized BackwardProcess instance with trained parameters.
        """
        with h5py.File(filepath, "r") as f:
            trained_params = f["trained_params"][:]
            denoising_loss_hist = f["denoising_loss_hist"][:]
            n_qubits = f.attrs["n_qubits"]
            n_ancilla = f.attrs["n_ancilla"]
            n_layers = f.attrs["n_layers"]
            T = f.attrs["T"]
            n_backward_samples = f.attrs["n_backward_samples"]
            epochs = f.attrs["epochs"]
            loss_function_str = f.attrs["loss_function"]

            if loss_function_str == "mmd":
                loss_function = LossFunction.MMD
            elif loss_function_str == "wasserstein":
                loss_function = LossFunction.WASSERSTEIN
            else:
                raise ValueError("Unknown loss function in saved file.")

        return cls(
            n_qubits=n_qubits,
            T=T,
            n_ancilla=n_ancilla,
            n_layers=n_layers,
            n_backward_samples=n_backward_samples,
            epochs=epochs,
            loss_function=loss_function,
            trained_params=trained_params,
            denoising_loss_hist=denoising_loss_hist,
        )

    def _gen_initial_backward_states(self, state):
        """Generates initial Haar-random quantum states for the backward process.

        These states represent the fully noisy state at time T, from which the
        denoising process begins.

        Parameters
        ----------
        state : int or np.random.Generator
            The random state or generator for the unitary group generator.

        Returns
        -------
        torch.Tensor, shape (n_backward_samples, 2**n_qubits)
            Normalized Haar-random initial states.
        """
        return torch.from_numpy(
            unitary_group.rvs(
                dim=2**self.n_qubits,
                size=self.n_backward_samples,
                random_state=state,
            )[:, :, 0],
        )

    @no_type_check
    def _back_circuit(self, input_state, params):
        """Defines the parameterized quantum circuit for a single denoising step.

        This circuit is a hardware-efficient ansatz consisting of layers of
        single-qubit rotations (`rx`, `ry`) and fixed entangling `cz` gates.

        Parameters
        ----------
        input_state : torch.Tensor, shape (2**n_total,)
            A single input state vector for the circuit (including ancilla).
        params : torch.Tensor, shape (n_layers, 2 * n_total)
            The parameters for all layers of the circuit.

        Returns
        -------
        torch.Tensor, shape (2**n_total,)
            The output state vector after passing through the circuit.
        """
        all_qubits = range(self.n_total)
        c = tc.Circuit(self.n_total, inputs=input_state)
        for i in range(self.n_layers):
            layer_params = params[i]
            c.rx(all_qubits, theta=layer_params[: self.n_total])
            c.ry(all_qubits, theta=layer_params[self.n_total : 2 * self.n_total])
            c.cz(list(all_qubits)[:-1:2], list(all_qubits)[1::2])
            if self.n_total > 2:
                c.cz(list(all_qubits)[1::2], list(all_qubits)[2::2])

        return c.state()

    def _measure(self, inputs):
        """Simulates the measurement of the ancilla qubits and state collapse.

        This function performs a probabilistic measurement on the ancilla
        qubits, collapses the state based on the outcome, and returns the
        renormalized state of the main system qubits. This is a key
        non-unitary step that allows for denoising.

        Parameters
        ----------
        inputs : torch.Tensor, shape (n_backward_samples, 2**n_total)
            A batch of state vectors from the circuit, including ancillas.

        Returns
        -------
        torch.Tensor, shape (n_backward_samples, 2**n_qubits)
            The renormalized, post-measurement state vectors of the main system.
        """
        m_probs = (
            torch.abs(
                inputs.reshape(inputs.shape[0], 2**self.n_ancilla, 2**self.n_qubits),
            )
            ** 2
        ).sum(dim=2)
        m_res = torch.multinomial(m_probs, num_samples=1).squeeze()
        indices = 2**self.n_qubits * m_res.view(-1, 1) + torch.arange(2**self.n_qubits)
        post_state = torch.gather(inputs, 1, indices)
        norms = torch.sqrt(torch.sum(torch.abs(post_state) ** 2, dim=1)).unsqueeze(
            dim=1,
        )

        return (1.0 / norms) * post_state

    def _get_denoised_state(self, input_state, params):
        """Performs a full denoising step for a batch of states.

        This helper function attaches ancilla qubits to the input states,
        runs the variational circuit, and performs the measurement simulation.

        Parameters
        ----------
        input_state : torch.Tensor, shape (n_backward_samples, 2**n_qubits)
            A batch of states for the main `n_qubit` system.
        params : torch.Tensor, shape (n_layers, 2 * n_total)
            The parameters for the variational circuit.

        Returns
        -------
        torch.Tensor, shape (n_backward_samples, 2**n_qubits)
            The denoised states.
        """
        input_with_ancilla = torch.concatenate(
            (
                input_state,
                torch.zeros(
                    (
                        self.n_backward_samples,
                        2**self.n_total - 2**self.n_qubits,
                    ),
                    dtype=torch.complex128,
                ),
            ),
            dim=1,
        )
        output_with_ancilla = self._back_circuit_vmap(input_with_ancilla, params)

        return self._measure(output_with_ancilla)

    def _train_step(self, t, input_states, rng):
        """Trains the model for a single timestep `t`.

        This function trains a dedicated set of parameters `params` that
        learns the mapping from the distribution of states provided in `input_states`
        (typically from the previous denoising step at t+1) to the target
        distribution at time `t`.

        Parameters
        ----------
        t : int
            The current timestep to train the model for.
        input_states : torch.Tensor
            The input states for the training step, typically generated by the
            model's previous steps (autoregressive training).
        rng : np.random.Generator
            The random number generator.

        Returns
        -------
        params : torch.Tensor, shape (n_layers, 2 * n_total)
            The trained parameters for this timestep.
        denoising_loss : np.ndarray, shape (epochs,)
            The MMD/Wasserstein loss recorded for each epoch.
        """
        denoising_loss = np.zeros(shape=self.epochs)
        params = torch.tensor(
            rng.normal(size=(self.n_layers, 2 * self.n_total)),
            requires_grad=True,
            dtype=torch.float64,
        )
        optimizer = torch.optim.Adam([params], lr=0.0005)

        loss_callable = (
            mmd_distance
            if self.loss_function == LossFunction.MMD
            else wasserstein_distance
        )

        for epoch in progress(range(self.epochs), title="Epochs", remove_on_exit=True):
            indices = rng.choice(
                self.forward_states.shape[1],
                size=self.n_backward_samples,
                replace=False,
            )
            true_data = self.forward_states[t, indices]

            output_t = self._get_denoised_state(
                input_states,
                params,
            )
            training_loss = loss_callable(output_t, true_data)

            loss = training_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            denoising_loss[epoch] = loss.item()

            if epoch % 100 == 0:
                print(
                    f"DEBUG: t={t}, Step {epoch:04d}, loss: {loss.item():.5f}",
                )

        return params.detach(), denoising_loss

    def run_training(self, rng):
        """Executes the full training pipeline for all timesteps.

        This method iterates backwards from `t=T-1` to `0`, calling
        `_train_step` for each timestep to learn the corresponding circuit
        parameters.

        Parameters
        ----------
        rng : np.random.Generator
            The random number generator.
        """
        self.denoising_loss_hist = np.zeros(shape=(self.T, self.epochs))
        self.trained_params = torch.zeros(
            (self.T, self.n_layers, 2 * self.n_total),
            dtype=torch.float64,
        )

        current_states = self._gen_initial_backward_states(rng)

        with progress(total=self.T, title="Training...", remove_on_exit=True) as bar:
            for t in range(self.T - 1, -1, -1):
                (
                    self.trained_params[t],
                    self.denoising_loss_hist[t],
                ) = self._train_step(t, current_states, rng)

                with torch.no_grad():
                    current_states = self._get_denoised_state(
                        current_states,
                        self.trained_params[t],
                    )

                bar.update()

    def save(self, filename: str | Path):
        """Saves the trained parameters and denoising loss history to an HDF5 file.

        Parameters
        ----------
        filename : str or Path
            The path to the HDF5 file where the data will be saved.

        Raises
        ------
        ValueError
            If `self.trained_params` or `self.denoising_loss_hist` is None.
        """
        if self.trained_params is None or self.denoising_loss_hist is None:
            raise ValueError("Training has not been run or completed.")

        with h5py.File(filename, "w") as f:
            f.create_dataset("denoising_loss_hist", data=self.denoising_loss_hist)
            f.create_dataset("trained_params", data=self.trained_params.numpy())

            f.attrs["n_qubits"] = self.n_qubits
            f.attrs["n_ancilla"] = self.n_ancilla
            f.attrs["n_layers"] = self.n_layers
            f.attrs["T"] = self.T
            f.attrs["n_backward_samples"] = self.n_backward_samples
            f.attrs["epochs"] = self.epochs
            if self.loss_function == LossFunction.MMD:
                f.attrs["loss_function"] = "mmd"
            elif self.loss_function == LossFunction.WASSERSTEIN:
                f.attrs["loss_function"] = "wasserstein"
            else:
                raise ValueError("Unknown loss function; cannot save.")

    def run(self, state):
        """Runs a full generative pass using the trained parameters.

        This method starts from a new set of Haar-random states at time `T` and
        sequentially applies the trained circuit for each timestep to generate
        a final sample distribution at `t=0`.

        Parameters
        ----------
        state : int or np.random.Generator
            The random state or generator for generating the initial noisy states.

        Returns
        -------
        torch.Tensor, shape (T + 1, n_backward_samples, 2**n_qubits)
            Generated states at each timestep of the backward process.

        Raises
        ------
        ValueError
            If training has not been run first and `trained_params` is None.
        """
        if self.trained_params is None:
            raise ValueError(
                "Training states not available! Training required before evaluation.",
            )

        backward_states = torch.zeros(
            (self.T + 1, self.n_backward_samples, 2**self.n_qubits),
            dtype=torch.complex128,
        )
        backward_states[self.T] = self._gen_initial_backward_states(state)

        for t in range(self.T - 1, -1, -1):
            with torch.no_grad():
                backward_states[t] = self._get_denoised_state(
                    backward_states[t + 1],
                    self.trained_params[t],
                )

        return backward_states
