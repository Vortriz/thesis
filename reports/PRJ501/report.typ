// #import "@preview/bananote:0.1.2": *
#import "lib.typ" as lib: *
#import "@preview/physica:0.9.8": *
#import "@preview/subpar:0.2.2"
#import "@preview/theoretic:0.3.1"
#import "@preview/gentle-clues:1.3.1": *

#import "/assets/components/quddpm-circuits.typ": forward-circuit, pqc-block, qsc-block, qudt-circuit, reverse-circuit

#show: note.with(
  title: [Generative Quantum Machine Learning],
  authors: (
    (
      [Rishi Vora],
      [IISER Mohali],
    ),
    (
      [#text(weight: "semibold")[#smallcaps[Supervisor]]: \ M. S. Santhanam],
      [IISER Pune],
    ),
  ),
  version: none,
)

#import theoretic.presets.basic: *
#show ref: theoretic.show-ref

#set math.equation(numbering: "(1)")
#set figure(placement: none)
#show figure.caption: it => [
  #set text(size: 0.85em)

  #text(weight: "bold")[
    #it.supplement #context it.counter.display(it.numbering):
  ]
  #h(3pt)
  #it.body
]
#let multifigure = subpar.grid.with(
  propagate-supplement: false,
  show-sub-caption: (num, it) => [
    #set text(size: 0.85em)
    *#num* #it.body
  ],
  row-gutter: 2em,
)

#let words = ("Julia", "Python")
#show regex(words.join("|")): set text(font: "JetBrainsMono NF", weight: "semibold", size: 0.9em)

#set math.accent(size: 125%)
#set math.vec(delim: "[")
#let argmin = math.limits("argmin")

// Document starts here -----------------------------------

#lib.abstract[
  Generative models have recently emerged as a powerful tool for learning and sampling from complex quantum state distributions. In this work, several architectures are investigated and compared, including Quantum Direct Transport (QuDT) and Quantum Denoising Diffusion Probabilistic Models (QuDDPM). A hybrid variant, Sequential Quantum Direct Transport (S-QuDT), is introduced, demonstrating significant improvement in efficiency when learning quantum states. To ensure efficient training, a comprehensive benchmarking of gradient-free and gradient-based optimizers is performed. Performance optimizations and a simulation framework developed in Julia are detailed to scale towards $10$-qubit systems, with a focus on learning the localised states of the Quantum Kicked Rotor system, a pragmatic model in quantum chaos.
]


= Introduction

The preparation and characterization of complex quantum states are fundamental challenges in quantum information science. While quantum computers promise to simulate many-body systems and solve optimization problems that are classically intractable, the practical utility of these devices often depends on the ability to prepare specific, high-fidelity quantum states, such as ground states of Hamiltonians or thermal states. Generative Quantum Machine Learning models have shown promise in learning and sampling from complex quantum state distributions.

== Problem Statement

Consider a target distribution $cal(E)_0$ of pure quantum states in a Hilbert space $cal(H)$. The task is to generate new states from this distribution, given only a finite number of samples $cal(S)_0 = { ket(psi_i^((0))) }_(i = 1)^m$ from that distribution.

The training process involves finding the optimal parameters $bold(theta)^ast$ that minimize a distance measure $cal(L)$ between the generated and target distributions:

$ bold(theta)^ast = arg min_(bold(theta)) cal(L)(cal(E)_"gen"(bold(theta)), cal(E)_0) $

Achieving this requires not only a sufficiently expressive circuit ansatz but also a distance metric that can faithfully capture the geometric structure of the quantum state space and an efficient optimization strategy to navigate the resulting loss landscape.


= Distance Metrics <ref:distance-metrics>

Given the nature of the problem, fidelity-based distance metrics are used. Two such loss functions and their properties are described @Zhang2024quddpm. Let $cal(E)_1$ and $cal(E)_2$ be two state distributions, and $cal(S)_1$ and $cal(S)_2$ be finite sets of size $n_1$ and $n_2$ sampled from these distributions respectively:

== Maximum Mean Discrepancy (MMD)

$
  cal(D)_"MMD" (cal(S)_1, cal(S)_2) = dash(F) (cal(S)_1, cal(S)_1) + dash(F) (cal(S)_2, cal(S)_2) - 2 dash(F) (cal(S)_1, cal(S)_2)
$

where

$ dash(F) (cal(S)_1, cal(S)_2) = bb(E)_(ket(phi) in cal(S)_1, ket(psi) in cal(S)_2) abs(braket(phi, psi))^2 $

is the mean fidelity. MMD is easy to compute and differentiable, but compares feature averages and hence may not capture the full structure of the distribution.

== 2-Wasserstein Distance ($W_2$)

Assuming that the states are uniformly sampled, the Wasserstein distance of second order amounts to solving the following discrete optimal transport (OT) problem:

$
    W_2 (cal(S)_1, cal(S)_2) = min_Gamma & chevron.l Gamma, C chevron.r, \
    s.t. quad & Gamma bold(1)_n_1 = 1/n_1 bold(1)_n_1, \
    & Gamma bold(1)_n_2 = 1/n_2 bold(1)_n_2, \
    & Gamma_(i j) >= 0
$

where

$ C_(i j) := 1 - abs(braket(phi_i, psi_j))^2 $ <eq:cost>

and $Gamma$ is a transport plan, $C$ is the cost matrix, $chevron.l dot.c, dot.c chevron.r$ is the Frobenius inner product, and $ket(phi_i) in cal(S)_1, ket(psi_j) in cal(S)_2$. Wasserstein distance captures the geometry of the underlying space and is a more faithful measure of distributional similarity. However, finding the optimal transport plan $bold(Gamma)^ast$  is a linear programming problem with $cal(O) (n^3)$ complexity.

While an alternative called Sinkhorn distance @Cuturi2013sinkhorn has reduced complexity by introducing an entropic regularization term, it has several downsides:

- Careful tuning of the regularization parameter is required.
- Specially for generative model learning, it causes shrinkage of the learned distribution towards the mean, and therefore cannot adequately cover the whole support of the target distribution.

== IPOT framework

To compute the Wasserstein distance more efficiently, the Inexact Proximal point method for Optimal Transport (IPOT) @Xie2018ipot is used. It is an iterative algorithm with theoretical convergence guarantees. Theoretically, it has similar complexity as the Sinkhorn method. Empirically, the algorithm seems to be linearly convergent with just one inner iteration, and thus is well suited for the present use case.

For use in a gradient-based optimization approach, the gradient of the Wasserstein distance with respect to the parameters of the generative model is required. This can be obtained in two ways:

- *Automatic differentiation*: The gradient can be computed by backpropagating through the proximal point iterations in IPOT. However, this is computationally expensive and memory intensive, especially for large sample sizes.

- *Envelope theorem*: Backpropagating through the proximal point iterations can be skipped by using the envelope theorem. The theorem statement and the following derivation are adapted for the present use case from Appendix B in @Xie2018ipot:

#theorem(
  label: <thm:envelope>,
  title: [Envelope theorem],
  [
    Let $f (x, theta)$ and l(x) be real-valued continuously differentiable functions, where $x in RR^n$ are choice variables and $theta in RR^m$ are parameters. Denote $x^ast$ to be the optimal solution of $f$ with constraint $l = 0$ and fixed $theta$, i.e.

    $ x^ast = argmin_x f (x, theta) quad s.t. quad l(x) = 0. $

    Then, assume that $V$ is continuously differentiable function defined as $V(theta) eq.triple f (x^ast (theta), theta)$, the derivative of $V$ over parameters is

    $ pdv(V, theta) = pdv(f, theta) $
  ],
)

In the present case:

- $V (theta)$ is the Wasserstein distance $W_2$.
- The choice variable $x$ is the transport plan $Gamma$.
- The objective function is $f (Gamma, theta) = chevron.l Gamma, C chevron.r = sum_(i j) C_(i j) Gamma_(i j)$.

According to the theorem,

$ pdv(W, theta) = sum_(i j) Gamma_(i j)^ast, pdv(C_(i j) (theta), theta) $

For the cost specified as per @eq:cost,

$ pdv(C_(i j), theta) = pdv(, theta) (1 - abs(braket(phi_i, psi_j (theta)))^2) $

Substituting this back into the gradient expression, the following is obtained:

$ pdv(W, theta) = - sum_(i j) Gamma_(i j)^ast pdv(, theta) abs(braket(phi_i, psi_j (theta)))^2. $ <eq:gradient>

So the gradient of the Wasserstein distance can be computed by just backpropagating through the fidelity term, without backpropagating through the proximal point iterations in IPOT, which is much more efficient.

It is also worth noting that calculating the fidelity of two pure states can be performed directly on a quantum computer using the SWAP test as described in @Zhang2024quddpm, making it a promising distance metric for training generative models on quantum hardware in the future.


= Generative Model Architectures

The problem and the distance metrics having been defined, the existing generative models are described, followed by the introduction of the proposed hybrid model.

== Quantum Direct Transport (QuDT)

The most trivial approach can be manifested in the form of the Quantum Direct Transport (QuDT) model @Zhang2024quddpm, which learns a direct mapping from the initial distribution of quantum states to the target distribution. This is achieved by training a single, deep Parameterized Quantum Circuit (PQC) to directly transport the initial states to the target states in one step.

Let a toy problem be specified: learning to generate the distribution of $1$-qubit states clustered around any arbitrary state, starting from a Haar random ensemble.

#{
  set image(height: 90pt)
  figure(
    grid(
      columns: 3,
      align: center + horizon,
      image("/assets/images/quddpm/cluster-arbitrary-20.png"),
      $stretch(->, size: #6em)^text(size: #1.5em, #raw("train"))$,
      image("/assets/images/quddpm/cluster-arbitrary-0.png"),
    ),
    caption: [The overarching goal of the toy problem.],
  )
}

A simple PQC is trained to learn to generate this distribution. It consists of $L$ layers where each layer consists of single qubit rotations using learnable $R_X$ and $R_Y$ gates, followed by entangling control-$Z$ gates on nearest neighbours. Additionally, $n_a$ ancillary qubits are included, initialized in the state $ket(0)^(times.o n_a)$ and measured out at the end of the circuit. The ancillary qubits increase the expressivity of the model, but the reasoning behind their use will be clear in @QuDDPM.

#figure(
  qudt-circuit,
  caption: [The QuDT circuit. $M_Z$ represents measurement in the Z basis.],
)

== Quantum DDPM <QuDDPM>

Using the same core concept as classical Diffusion models (@ref:classical-ddpm), a quantum version of DDPM called Quantum Denoising Diffusion Probabilistic Models (QuDDPM) was proposed @Zhang2024quddpm. An ensemble of states is scrambled over multiple steps in the forward process and then a series of PQCs are trained to reverse this scrambling process step by step in the reverse process. This divides the task of learning a map from full noise to the target distribution into subtasks with low-depth circuits which can avoid barren plateaus.


=== The forward process

This consists of $T$ steps, where at each step $t$ a random unitary $U_i^((t))$ is applied to each state $i$ in the ensemble, which scrambles them. After $T$ steps, the states are completely scrambled and form a Haar random ensemble.

#{
  let t = (0, 5, 10, 15, 20)

  figure(
    grid(
      columns: 5,
      ..t.map(i => text(size: 0.8em)[$t = #i$]),
      ..t.map(i => image(
        "/assets/images/quddpm/cluster-arbitrary-" + str(i) + ".png",
        height: 90pt,
      )),
    ),
    caption: [Scrambling $m=1000$ states sampled from the toy distribution over $T=20$ steps, as shown on the Bloch sphere.],
  )
}

In practice, $U_t$ is implemented using the fast scrambling model @Belyansky2020fastscr --- layers of tunable single-qubit rotation gates followed by homogeneous entangling layers of all-to-all ZZ rotations --- and the hardware efficient ansatz @Kandala2017hwea, which is as follows:

#multifigure(
  figure(
    forward-circuit,
    caption: [Forward process circuit],
  ),
  <fc>,
  figure(
    qsc-block,
    caption: [Quantum Scrambling Circuit (QSC) Block],
  ),
  <qsc>,
  caption: [The forward circuit (@fc) consists of $T$ number of QSC blocks. Each QSC block (@qsc) consists of extrinsic single qubit rotations using Euler angles followed by homogeneous entangling layers of all-to-all ZZ rotations.],
)

=== The reverse process

The reverse process consists of $T$ steps, where at each step $t$ a PQC is trained to learn the denoising map from the scrambled distribution at step $t$ to the less scrambled distribution at step $t-1$. The reverse circuit is as follows:

#multifigure(
  figure(
    reverse-circuit,
    caption: [Reverse process circuit.],
  ),
  <rc>,
  figure(
    pqc-block,
    caption: [PQC block used in the reverse circuit.],
  ),
  <pqc>,
  caption: [The reverse circuit (@rc) consists of $T$ number of PQC blocks. Each PQC block (@pqc) is the same as the one used in the QuDT model, but with its own learnable parameters.],
)

Measurements are necessary, as the denoising map is contractive and maintains the purity of all generated data in $tilde(S)_0$. No specific constraint is placed on the measurement result. The measurement on ancillas is performed and discarded; only the post-measurement state of the data qubits is collected @Zhang2024quddpm.

#idea(title: [Connection with QuDT])[
  The QuDT model can be seen as a special case of QuDDPM with $T=1$ step in the reverse process, where the forward process is just a single random unitary that scrambles the initial distribution. This justifies the choice of ancillary qubits and their measurement in the QuDT model.
]

#pagebreak()

== Sequential Quantum Direct Transport (S-QuDT)

A hybrid model, Sequential Quantum Direct Transport (S-QuDT), is proposed, taking ideas from both QuDT and QuDDPM. The model consists of $T$ steps, where at each step $t$ a PQC is trained to learn a direct transport map from the distribution at step $t$ to the target distribution, instead of learning a denoising map to the less scrambled distribution as in QuDDPM.

This removes the forward scrambling process entirely. Empirically, it is found that this model can achieve similar convergence performance as QuDDPM, while requiring significantly fewer steps and thus much less training time. The reverse circuit used in S-QuDT is the same as the one used in the QuDDPM model (@rc and @pqc). Ancilla qubits are still used and measured out at the end of each step, as in QuDDPM, for the same reasons.


= Optimizers

To find the optimal parameters for the generative model, an optimizer is employed. These are classified into two categories: gradient-free and gradient-based optimizers.

== Gradient-based optimizers

The core idea is to traverse the parameter space in the direction of steepest descent by computing the gradient of the chosen objective function (@ref:distance-metrics) with respect to the parameters. For MMD, automatic differentiation can be employed; whereas for Wasserstein distance, the gradient can be computed using @eq:gradient. This gradient can then be used to perform optimization using standard algorithms like Adam. However, gradient measurements for quantum circuits scale poorly with the system size.

Another problem with standard gradient descent algorithms is that each optimization step is strongly connected to a Euclidean geometry on the parameter space. The natural gradient is a generalization of a typical gradient that accounts for the curvature of the metric function at hand. The Quantum Natural Gradient (QNG) @Stokes2020qng is the quantum version of the natural gradient that uses the Fubini-Study metric tensor to take into account the geometry of the quantum state space. However, constructing the Fubini-Study metric tensor requires $cal(O) (N^2)$ measurements in qubits, which is infeasible for large systems.

=== Benchmarking Gradient-based variants

A preliminary benchmark of several gradient-based optimizers was conducted on a $4$-qubit system to identify the most robust choice for larger simulations. The optimizers included Adam (baseline), RAdam, NAdam, AdamW, AdaBelief, Lion, and AMSGrad.

The results indicated that *AdaBelief* and *AMSGrad* significantly outperformed the standard Adam optimizer in terms of convergence speed and final fidelity. Further testing on a $6$-qubit system confirmed that AMSGrad provided the most stable performance across different initializations. Consequently, AMSGrad was selected as the primary optimizer for the benchmarking of architectures.

These reasons provide motivation for exploring gradient-free optimizers.

== Gradient-free optimizers

These optimizers do not require gradient information and instead rely on evaluating the objective function at different points in the parameter space to guide the search for optimal parameters.

=== Rotosolve

Rotosolve is a coordinate-wise optimization algorithm that iteratively optimizes one parameter at a time while keeping others fixed @Ostaszewski2021rotosolve using the following update rule for the $d$-th parameter:

#let eH(x) = $expval(H)_(theta_d = #x)$
#let halfpi = $frac(pi, 2, style: "horizontal")$

$ theta_d^ast = - pi/2 - "arctan2"(2 eH(0) - eH(halfpi) - eH(-halfpi), eH(halfpi) - eH(-halfpi)) $

This is targeted towards optimization problems where the encoded objective function can be represented as a Hermitian operator. Hence, it is not applicable for the present use case where the objective function is a distance metric between distributions, and not an expectation value of a Hermitian operator.

=== Simultaneous Perturbation Stochastic Approximation (SPSA)

SPSA estimates the gradient by perturbing all parameters simultaneously in random directions @Spall1992spsa with the following update rule:

$
  bold(theta)_(k+1) = bold(theta)_k - a_k (f (bold(theta)_k + c_k bold(Delta)_k) - f (bold(theta)_k - c_k bold(Delta)_k))/(2 c_k ) vec(Delta_(k 1)^(-1), Delta_(k 2)^(-1), dots.v, Delta_(k p)^(-1))
$

Only two evaluations of the objective function per iteration are required, regardless of the number of parameters, making it efficient for high-dimensional optimization problems. However, it can be noisy and may require careful tuning of hyperparameters $a_k$ and $c_k$.

=== Quantum Natural SPSA (QNSPSA)

QNSPSA @Gacon2021qnspsa manages to combine the merits of QNG and SPSA by estimating both the gradient and the metric tensor stochastically. The gradient is estimated in the same fashion as the SPSA algorithm, while the Fubini-Study metric is computed by a second-order process. In practice, it requires 2 (for gradient) + 4 (for metric tensor) + 2 (for the current and the next-step loss) = 8 circuit evaluations per iteration, which is still $cal(O) (1)$. Given a randomly sampled direction $bold(h) tilde cal(U) ({-1, 1}^d)$, the update rule is as follows:

$
  bold(theta)^((k+1)) = bold(theta)^((k)) - eta bold(hat(g, size: #200%))^(-1) (bold(theta)^((k)), bold(h)_1^((k)), bold(h)_2^((k)))_"SPSA" hat(nabla f) (bold(theta)^((k)), bold(h)^((k)))_"SPSA"
$

where

$
  bold(hat(g, size: #200%)) (bold(theta), bold(h)_1, bold(h)_2)_"SPSA" = (delta cal(F))/(8 epsilon.alt^2) (bold(h)_1 bold(h)_2^T + bold(h)_2 bold(h)_1^T)
$

$
  delta cal(F) = cal(F) (bold(theta), bold(theta) + epsilon.alt bold(h)_1 + epsilon.alt bold(h)_2) - cal(F) (bold(theta), bold(theta) + epsilon.alt bold(h)_1) - cal(F) (bold(theta), bold(theta) - epsilon.alt bold(h)_1 + epsilon.alt bold(h)_2) + cal(F) (bold(theta), bold(theta) - epsilon.alt bold(h)_1)
$

$
  cal(F) (bold(theta), bold(theta)') = bb(E)_(ket(phi) in cal(S)_1, ket(psi) in cal(S)_2) abs(braket(phi (bold(theta')), psi (bold(theta))))^2
$

Further details on the algorithm and implementation can be found in @Duan2022qnspsa-demo. The primary difference between the present implementation and that of @Duan2022qnspsa-demo is that an average over all states in the ensemble is taken when computing $cal(F)$. This is a simplification that is not ideal and warrants further investigation.

= Implementation details

The simulation framework for the models discussed in this work was implemented from scratch in the Julia programming language @Bezanson2017julia. Julia was chosen for its high-performance capabilities, particularly for numerical and scientific computing, while maintaining a high-level syntax suitable for rapid prototyping.

== Software Stack

- *Quantum Simulation*: `Yao.jl` @Luo2020yaojl is used, which is a flexible and extensible framework for quantum algorithm research. It provides efficient state vector simulation and allows for the construction of complex, hierarchical quantum circuits.

- *Automatic Differentiation (AD)*: To train the Parameterized Quantum Circuits (PQCs), Zygote.jl @Innes2018zygotejl is employed, which is a source-to-source AD library in Julia used for high-level differentiability of the training loop and the IPOT-based Wasserstein distance computation.

== Computational Optimizations

To address the exponential scaling of quantum state simulation, several optimization strategies were employed:

- *IPOT with Envelope Theorem*: As derived in @ref:distance-metrics, the envelope theorem is used to avoid backpropagating through the iterative optimal transport solver. This significantly reduces the computational overhead when using Wasserstein distance as the loss function.

- *In-place Operations*: Wherever possible, in-place operations were used to minimize memory allocations and reduce garbage collection overhead.

#pagebreak()

= Benchmarking architectures

The performance of the three architectures --- QuDT, QuDDPM, and S-QuDT --- is evaluated on a $5$-qubit system. The models are trained to learn $5$-qubit clustered states. For all models, $n_a = 3$ ancillary qubits are used. AMSGrad optimizer with a learning rate of $eta = 0.01$ and QNSPSA are compared for S-QuDT. To ensure a fair comparison, the total circuit depth is kept constant at $80$ layers for all models. For QuDDPM and S-QuDT, $T=5$ steps are used in the reverse process, which means that each step is allocated $16$ layers. For QuDT, all $80$ layers are used in a single step.

#multifigure(
  figure(
    image("/simulations/3-qml-jl/saves/2026-04-22_191947_724_AMSGrad/training_plot.png"),
    caption: [QuDT ($T=1$, $L=80$) \ \ ],
  ),
  <qudt>,
  figure(
    image("/simulations/3-qml-jl/saves/2026-04-22_205640_400_AMSGrad/training_plot.png"),
    caption: [QuDDPM ($T=5$, $L=16$) \ (Loss (y-axis) is on a logarithmic scale)],
  ),
  <quddpm>,
  figure(
    image("/simulations/3-qml-jl/saves/2026-04-22_210424_141_AMSGrad/training_plot.png"),
    caption: [S-QuDT with AMSGrad ($T=5$, $L=16$) \ (Loss (y-axis) is on a logarithmic scale)],
  ),
  <squdt-amsgrad>,
  figure(
    image("/simulations/3-qml-jl/saves/2026-04-23_063749_608_QNSPSA/training_plot.png"),
    caption: [S-QuDT with QNSPSA ($T=5$, $L=16$) \ (Loss (y-axis) is on a logarithmic scale)],
  ),
  <squdt-qnspsa>,
  columns: (1fr, 1fr),
  caption: [Comparison of training loss history for different architectures on a $5$-qubit system. @qudt, @quddpm, and @squdt-amsgrad use AMSGrad whereas @squdt-qnspsa uses QNSPSA. S-QuDT with AMSGrad demonstrates the fastest and most stable convergence. The spikes in the loss history of QuDDPM and S-QuDT are due to the sequential training process, where each step is trained after the previous step has been fully trained.],
)

The results are summarized in @tbl:benchmarking.

#figure(
  table(
    columns: 4,
    [*Model*], [*Optimizer*], [*Total epochs*], [*Final Loss* ($W_2$)],
    [QuDT], [AMSGrad], [600], [0.8245],
    [QuDDPM], [AMSGrad], [2200], [0.0742],
    [S-QuDT], [AMSGrad], [2200], [0.0045],
    [S-QuDT], [QNSPSA], [5000], [0.8448],
  ),
  caption: [Summary of benchmarking results for $5$-qubit state generation.],
) <tbl:benchmarking>

As observed, the single-step QuDT model fails to converge to a low loss, despite having a deep circuit. This suggests that the landscape for a single deep PQC is extremely difficult to navigate. In contrast, both sequential models (QuDDPM and S-QuDT) achieve much higher fidelity. S-QuDT with QNSPSA fails to converge but since it is a global optimization method, it may perform better when training all the PQC blocks simultaneously. This needs further investigation. S-QuDT with AMSGrad, in particular, achieves an order of magnitude lower loss than QuDDPM for the same training time and circuit depth, suggesting that the direct transport objective with gradient based optimization is more efficient than denoising when applied sequentially.


= Future Work:  Quantum Kicked Rotor

The Hamiltonian of the QKR is given by

$
  i hbar pdv(, t) psi (x, t) = underbrace(- hbar^2/(2 I) pdv(, x, 2), H_"free") psi (x, t) + underbrace(k cos(x) sum_n delta (t - n T), H_"kick") psi (x ,t)
$ <eq:qkr-hamiltonian>

where $x$ is the angular position of the particle, $p$ is the conjugate momentum, $k$ is the kick strength, and $T$ is the period of the kicks. The evolution over one period can be described by the Floquet operator $hat(U)$ as follows:

$
  hat(U) & = hat(U)_"kick" hat(U)_"free" \
         & = exp(- i K/hbar_s cos hat(x)) exp(- i hat(p)^2/(2 hbar_s))
$ <eq:qkr-floquet>

where $hbar_s = (hbar T)/I$, $K = k/hbar$. Detailed derivation of the Floquet operator can be found in @ref:qkr-derivation.

Long time dynamics are generated by successive applications of the Floquet operator. The QKR exhibits a phenomenon called dynamical localization, where the state becomes exponentially localized in momentum space after a certain time, despite the classical counterpart exhibiting unbounded energy growth. This makes it an interesting testbed for generative models to learn and sample from.

A primary objective now is to scale the generative process to $10$-qubit systems (with 5 ancilla qubits), since that is the minimum system size required for the QKR to exhibit its characteristic behavior. In this regime, the Hilbert space dimension grows to $2^(10+5) = 32,768$ and correspondingly, the matrices grow to _billions_ of elements. To handle this would require significant computational resources and optimizations, which are currently being worked on.

#pagebreak()

#bibliography("/references.bib")

#pagebreak()

#show: appendices

= Classical DDPM <ref:classical-ddpm>

The idea behind Denoising Diffusion Probabilistic Models (or Diffusion Models for short) is to destroy the training data by adding noise, and then learn to reverse this process to generate new data.

#figure(
  image("/assets/images/ddpm-demo.png"),
  caption: [From the book Understand Deep Learning @Prince2023dl],
)

#grid(
  columns: (1fr, 35%),
  column-gutter: 1em,
  [Let us consider the case of image generation. There exist many manifolds in the high-dimensional space of images, that contains all valid images. The aim is to sample new images from these manifolds, but the problem is that their algebraic form is not known because it is too complex. Lets consider any such manifold; take a point on it and add noise to it iteratively, pushing it outside the manifold. Then a model is trained to learn to remove the noise step by step, to get back into the manifold. This is the basic idea behind diffusion models. The process of adding noise is called the _forward process_, and the process of removing noise is called the _reverse process_.],
  figure(
    image("/assets/images/ddpm-image-space.png", height: 18%),
    caption: [A high level conceptual overview of DDPM],
  ),
)

= Derivation of the Floquet operator <ref:qkr-derivation>

Starting from the time-dependent Schrodinger equation given in @eq:qkr-hamiltonian, let the Floquet operator $hat(U)$ advance the system from $t = n T^+$ to $t = (n+1) T^+$, where $n$ is an integer. $T^+$ represents the time just after the kick whereas $T^-$ represents the time just before the kick. The evolution can be broken down into two steps:

1. $t = n T^+$ to $t = (n+1) T^-$

During this time interval, the system evolves under the free particle Hamiltonian, which gives the evolution operator

$
  hat(U)_"free" & = exp(- i/hbar integral_(n T^+)^((n+1) T^-) hat(H)_"free" d t) \
                & = exp(- i T/hbar hat(p)^2/(2 I)) \
$

2. $t = (n+1) T^-$ to $t = (n+1) T^+$

During this time interval, the system evolves under the kick Hamiltonian. The kick is instantaneous, so the evolution operator can be computed as:

$
  hat(U)_"kick" & = exp(- i/hbar integral_((n+1) T^-)^((n+1) T^+) hat(H)_"kick" d t) \
                & = exp(- i k/hbar cos hat(x))
$

Combining these two steps, the Floquet operator is given by

$
  hat(U) & = hat(U)_"kick" hat(U)_"free" \
         & = exp(- i k/hbar cos hat(x)) exp(- i T/hbar hat(p)^2/(2 I)) \
$

Substituting $hbar_s = (hbar T)/I$ and $K = k/hbar$, the Floquet operator can be rewritten in the form given in @eq:qkr-floquet.
