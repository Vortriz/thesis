#import "@preview/kunskap:0.1.0": *
#import "@preview/physica:0.9.8": *

#show: kunskap.with(
  title: [Report: January],
  author: "Rishi Vora",
  date: datetime.today().display(),
  header: "PRJ501",

  headings-font: "Times New Roman",
  body-font-size: 11pt,
)

#set heading(numbering: "1.")
#show heading: it => strong(smallcaps(it), delta: 100)
#set text(spacing: 0.3em)
#set figure(placement: none)


#let mono = it => text(size: 10pt, font: "Maple Mono", it)
#show "Julia": mono
#show "Python": mono
#show "PyTorch": mono


= Short Summary of Previous Work

In July, I worked on learning about Classical DDPM models and implemented a basic version of it with PyTorch. Based on that, we started exploring Quantum DDPM models.

= Base

We focused on the implementation by Zhang et al. @Zhang2024-wm. Here, we take an ensemble of states of our interest, gradually add noise to them via a Quantum Scrambling Circuit (QSC), then train multiple Parameterized Quantum Circuits (PQC) to denoise the ensemble step-by-step.

#figure(
  image("../../assets/zhang2024/overview.png", height: 250pt),
  caption: [Overview of Quantum DDPM \ Credits: @Zhang2024-wm],
)

Let there be ensembles of states $cal(E)_1$ and $cal(E)_2$ of size $n_1$ and $n_2$ respectively. Two loss functions are used here:

- *Maximum Mean Discrepancy (MMD)*
  $ cal(D)_"MMD" (cal(E)_1, cal(E)_2) = dash(F) (cal(E)_1, cal(E)_1) + dash(F) (cal(E)_2, cal(E)_2) - 2 dash(F) (cal(E)_1, cal(E)_2) $

  where

  $ dash(F) (cal(E)_1, cal(E)_2) = bb(E)_(ket(phi) tilde cal(E)_1, ket(psi) tilde cal(E)_2) |braket(phi, psi)|^2 $

  - Worked well for simple distribution of states (e.g. states clustered around the ground state)
  - Computationally efficient

- *Wasserstein Distance*
  $ W_2 (cal(E)_1, cal(E)_2) = min_P & chevron.l P, C chevron.r, \ s.t. quad & P bold(1)_n_1 = 1, \ & P bold(1)_n_2 = 1, \ & P_(i j) >= 0 $

  where

  $ C_(i j) = 1 - |braket(phi_i, psi_j)|^2 $

  and $P$ is a transport plan.
  - Worked better for complex distributions
  - Computationally expensive

= My Work

I implemented the Quantum DDPM model as per the above paper using PyTorch. I used #link("https://github.com/Francis-Hsu/QuantGenMdl")[their own PyTorch implementation] as a reference and reimplemented it in a much cleaner manner. Being Python, it was quite slow to simulate, despite the extensive use of Numpy arrays and PyTorch tensors. So I ported the entire implementation to Julia. The training time was cut down to a fraction of the original time.

#pagebreak()

== Experimentation

=== Architectures

For the distribution of simple states (clustered around the ground state), the original diffusion type model worked well. But Consistency type model @Song2023-jb worked even better, converging faster and yielding better results.

For distribution of arbitrary states, the Consistency model did not work well.

=== Loss functions

Apart from the two loss functions mentioned, I experimented with a few other variants:

- *Sinkhorn Distance*
  - Approximates Wasserstein distance with entropic regularization
  - More computationally efficient than Wasserstein distance

- *Inexact Proximal point method for exact Optimal Transport problem (IPOT)*
  - Another approximation for Wasserstein distance but has theoretical guarantees of convergence to the Wasserstein distance @Xie2018-pc
  - Computationally efficient

=== Optimizers

- Adam optimizer worked well for most cases.
- Rotosolve optimizer showed promise for smaller systems. One key advantage is that it does not require gradient calculations, making it potentially more efficient. More testing is needed.
- @Xie2018-pc hints at a simpler gradient calculation method for the Wasserstein distance, which I am currently testing.

= Planned Work

Next, I have been tasked with generating Quantum Kicked Rotor states, which requires implementing the model for atleast 9-10 qubits. Given the exponential scaling of quantum state simulation on classical hardware, this requires signification compute power. So I am optimising my code further.

I am also looking into implementing potentially better loss functions (e.g. Sinkhorn divergence) and optimizers (e.g. Quantum Natural Gradient Descent).

#pagebreak()

#bibliography("references.bib")