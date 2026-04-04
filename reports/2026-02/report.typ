#import "@preview/kunskap:0.1.0": *
#import "@preview/physica:0.9.8": *

#show: kunskap.with(
  title: [Report: February],
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

In January, I worked on implementing the base of the Quantum Diffusion Probabilistic Model (DDPM) in Julia and performing basic testing for several optimizers and loss functions.

= My Work

Firstly, I refactored my code to be more modular, which made it easier to experiment with different optimizers and loss functions. I also moved from training the model to generate states clustered around the ground state to generating states clustered around arbitrary states, which is more general.

== Analyzing a different training objective

The original paper @Zhang2024-wm used a training objective that minimizes the Wasserstein distance between two distributions of states.

While looking for alternatives, I came across the emerging field of Quantum Optimal Transport (QOT), which is a quantum analogue of classical optimal transport theory. In particular, I found several papers that propose quantum versions of the Wasserstein distance:

- @Beatty2025review reviews several definitions of quantum Wasserstein distance.
- @Kiani2022qemd proposes the quantum Earth Mover's Distance (Wasserstein distance of order 1) and demonstrates its usage in a quantum GAN setting.
- @Beatty2025qwass proposes an intuitive and general definition of quantum Wasserstein distance of order $p$ based on the concept of quantum couplings.
- @Toth2025rel establishes a connection between these quantum Wasserstein distances and, most importantly, mentions that the quantum Wasserstein distance of order 2 as defined in @Beatty2025qwass is equivalent to another formulation which can be computed numerically.

All available work in this area is quite theoretical so far, and there exist no code implementations to the best of my knowledge.

There is also the fact that these distances are defined for density matrices, while my model is built around generating and optimizing over an ensemble of pure states. Recovering a desired ensemble from a density matrix is not straightforward, and I am not sure if it is even possible in general. I will be looking into this aspect as well.

== Optimizers

The original paper @Zhang2024-wm used the Adam optimizer. I intended to find a good gradient-free optimizer, so I experimented with the following:

- *Rotosolve* @Ostaszewski2021rotosolve
  - A gradient-free optimizer that optimizes one parameter at a time by solving a small optimization problem.
  - Worked well for smaller systems (up to 3 qubits) but failed to converge for larger systems.
  - Failed because it is designed for optimizing objective functions that are encoded as expectation values of some operator.

- *Simultaneous Perturbation Stochastic Approximation (SPSA)* @Spall1992spsa
  - A gradient-free optimizer that estimates the gradient by perturbing all parameters simultaneously.
  - Failed to converge for my problem, for unknown reasons.

- *Quantum Natural SPSA (QNSPSA)* @Gacon2021qnspsa
  - A second-order variant of SPSA that takes into account the geometry of the quantum state space by approximating the quantum Fisher information matrix.
  - Showed some promise for smaller systems but did not converge reasonably fast for larger systems.

- *Adam*
  - A previous attempt at using Adam in Julia was not successful due to issues with the automatic differentiation library (Zygote), but I was able to get it to work on another attempt. Compared to Python, it offers a significant speedup (10-15x).
  - I also tried targeting the initial state directly instead of intermediate states, which worked just as well up to 4 qubits. More testing is needed for larger systems.

= Planned Work

Next steps include making the code work on GPUs to see if there are any speedups. Regardless of that, I will be deploying the code on the cluster to train models for larger systems (7-10 qubits). The aim is to be able to train a model for 10 qubits in a reasonable amount of time.

I will also try out more gradient-based optimizers, such as AdamW, RAdam, etc.

A more ambitious and interesting direction is to explore quantum Wasserstein distances as training objectives.

#bibliography("references.bib")