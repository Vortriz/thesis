#import "/assets/components/quddpm-circuits.typ": (
    U_ent, U_ent-decomposed, eha-block, forward-circuit, hea-block, qsc-block,
    qudt-circuit, reverse-circuit,
)
#import "@preview/presentate:0.2.6": *
#import "@preview/physica:0.9.8": *
#import "@preview/larrow:1.2.0": *
#import "@preview/subpar:0.2.2"
#import "@preview/gentle-clues:1.3.1": *

#set page(paper: "presentation-16-9")
#set text(size: 18pt)
#set list(spacing: 1.5em)
#show math.equation.where(block: false): set text(0.9em)
#set par(leading: 0.8em)
#show figure.caption: set text(size: 0.8em)
#let main-color = rgb("#25285f")
#let main-color-variant = rgb("#5f65dd")
#let main-gradient = gradient.linear(
    angle: 30deg,
    main-color,
    main-color-variant,
)

#let multifigure = subpar.grid.with(
    propagate-supplement: false,
    show-sub-caption: (num, it) => [
        #set text(size: 0.85em)
        *#num* #it.body
    ],
)

#show cite: super

#let argmin = math.limits("argmin")
#let boxed-heading = it => {
    set text(size: 1.25em, fill: main-color)
    show rect: set block(below: 1em)

    rect(
        width: 100%,
        radius: 0.2em,
        inset: 0.75em,
        stroke: 2pt + main-color-variant.transparentize(75%),
        heading(level: 1, it),
    )
}
#let section = title => {
    set page(fill: main-gradient, margin: (x: 3cm))

    slide[
        #set align(horizon)
        #set par(leading: 0.5em, spacing: 0.5em)

        #show heading.where(level: 1): set text(
            fill: white,
            size: 50pt,
            weight: "black",
        )

        #heading(title)
    ]
}

#page(
    header: none,
    footer: none,
    fill: main-gradient,
    margin: (x: 3cm, y: 0cm),
)[
    #place(
        bottom + left,
        {
            set text(
                size: 13pt,
                fill: main-color-variant.lighten(30%).transparentize(40%),
                font: "Maple Mono",
                weight: "semibold",
            )

            grid(
                columns: (1fr,) * 3,
                align: (left, center, right),
                inset: (bottom: 0.25em, x: 2em),
                [PRJ502], [Rishi Vora], [IISER Mohali],
            )
        },
    )

    #set align(horizon)
    #set par(leading: 0.5em)

    #text(fill: white, size: 50pt, weight: "black")[
        Quantum Generative \
        Machine Learning
    ]
]

#{
    set align(center + horizon)
    set text(
        size: 30pt,
        weight: "black",
        fill: main-gradient,
    )

    slide[
        Quantum Generative Machine Learning
    ]

    slide[
        #hide[Quantum] Generative Machine Learning

        #show: pause

        #place(
            left + horizon,
            [
                #set text(fill: main-color-variant.transparentize(50%))
                #set list(spacing: 2em, marker: none)

                - Auto regressive
                - Diffusion
                - VAE
                - GAN
                - Flow based
            ],
        )
    ]

    slide[
        Quantum #hide[Generative Machine Learning]

        #show: pause

        #place(
            right + horizon,
            [
                #set text(fill: main-color-variant.transparentize(50%))
                #set list(spacing: 2em, marker: none)

                - Financial time series gen @Waller2026Financial
                - Drug design gen @Jain2026DrugDesign
                - Image gen @Huang2021ImageGen
                - #transform(
                        [State gen @Chen2026QGDM @Zhang2024quddpm],
                        text,
                        text.with(fill: main-color.transparentize(20%)),
                        start: none,
                    )
            ],
        )
    ]
}

#slide[
    #boxed-heading[Problem Statement]

    - Consider a target distribution $cal(E)_0$ of pure quantum states in a Hilbert space $cal(H)$.
    - The task is to generate new states from this distribution, given only a finite number of samples $cal(S)_0 = { ket(psi_i^((0))) }_(i = 1)^m$.
    - The model has to find the optimal parameters $theta^*$ that minimize the distance between the generated and target distributions:
    $
        bold(theta)^ast = arg min_(bold(theta)) cal(L)(cal(E)_"gen"(bold(theta)), cal(E)_0)
    $
]

// #slide[
//     #grid(
//         columns: (1fr, 1fr),
//         column-gutter: 2em,
//         align: (left, right),
//         [= Distance Metrics],
//         [
//             #set par(leading: 0.4em)
//             #set text(
//                 size: 0.8em,
//                 fill: main-color-variant.mix(main-color).transparentize(30%)
//             )
//             #show text: smallcaps
//             == Maximum Mean Discrepancy (MMD)
//         ],
//     )
// ]

#slide[
    #set align(center + horizon)

    #place(
        center + horizon,
        text(fill: luma(90%), size: 430pt)[&],
    )

    #set text(size: 50pt, fill: main-gradient)

    #grid(
        columns: (1fr, 1fr),
        [a *Distance*\ _Metric_], [an\ *Optimizer*],
    )
]

#slide[
    #set align(center + horizon)
    #show heading.where(level: 1): set text(
        size: 50pt,
        weight: "regular",
        fill: main-gradient,
    )

    #grid(
        columns: (1fr, 1fr),
        heading[a *Distance*\ _Metric_],
    )
]

#set page(
    margin: 0cm,
    fill: gradient.linear(
        angle: 45deg,
        main-color,
        main-color-variant,
    ),
)

#let windowed-slide(title, body) = slide[
    #show heading.where(level: 2): it => block(
        below: 2em,
        text(fill: main-color, smallcaps(it)),
    )

    #grid(
        columns: (2cm, 1fr),
        rows: 100%,
        align: center + horizon,
        {
            set text(
                size: 1.25em,
                font: "Maple Mono Normal NF",
                fill: white.transparentize(10%),
            )

            rotate(-90deg, reflow: true, title)
        },
        grid.cell(
            inset: 0.5em,
            block(
                width: 100%,
                height: 100%,
                radius: 0.5em,
                fill: white,
                inset: 1cm,
                align(top + left, body),
            ),
        ),
    )
]

#windowed-slide("Distance Metrics")[
    #set align(center + horizon)
    #set text(size: 30pt)
    #columns(2)[
        $ rho_1 #arrow-label(<rho1>) = sum_i a_i #arrow-label(<sum1>) psi_i $

        #colbreak()

        $ #arrow-label(<rho2>) rho_2 = sum_j b_j #arrow-label(<sum2>) phi_j $
    ]

    #let arw = label-arrow.with(
        both-tip: ">",
        stroke: 2pt + gray,
        caption-options: (fill: white, padding: (left: 7pt)),
    )

    #show: pause

    #arw(<rho1>, <rho2>, bend: -105, caption: emoji.crossmark)

    #show: pause

    #arw(<sum1>, <sum2>, bend: 105, caption: [✅], both-offset: (0pt, -70pt))
]

#set par(leading: 0.5em)

#windowed-slide("Distance Metrics")[
    == Maximum Mean Discrepancy (MMD)

    #show: pause

    - Kernel based measure
    - Compares the average features of two distributions

    - $cal(E)_1$ and $cal(E)_2$ #sym.arrow two distributions of pure quantum states \
        $cal(S)_1$ and $cal(S)_2$ #sym.arrow finite sets of sizes $n_1$ and $n_2$ sampled from these distributions \
        For $k := F(ket(phi), ket(psi)) = abs(braket(phi, psi))^2$, the squared empirical MMD is defined as

    $
        cal(D)_"MMD"^2 (cal(S)_1, cal(S)_2) = dash(F) (cal(S)_1, cal(S)_1) + dash(F) (cal(S)_2, cal(S)_2) - 2 dash(F) (cal(S)_1, cal(S)_2)
    $

    where

    $
        dash(F) (cal(S)_1, cal(S)_2) = frac(1, n_1 n_2) sum_(i=1)^(n_1) sum_(j=1)^(n_2) F(ket(phi_i), ket(psi_j)),
    $
]

#windowed-slide("Distance Metrics")[
    #grid(
        columns: 3,
        column-gutter: 1em,
        align: center + horizon,

        figure(
            image("/assets/images/circle-dist.svg"),
            caption: [States lying on X-Z plane of the Bloch sphere.],
        ),
        text(size: 50pt)[
            $approx$
        ],
        figure(
            image("/assets/images/haar-dist.svg"),
            caption: [States sampled from a Haar random distribution.],
        ),
    )
]

#windowed-slide("Distance Metrics")[
    == 1-Wasserstein Distance ($W_1$)

    #show: pause

    - Compares two distributions. How much mass must be transported to transform one distribution into the other.
    - A transport plan specifies how much probability mass from each sample of the first distribution is matched with each sample of the second.
    - Minimize the total transport cost, so unlike MMD, Wasserstein distance uses the geometry of the sample space explicitly.

    $
        W_1 (cal(S)_1, cal(S)_2) = min_Gamma sum_(i=1)^(n_1) sum_(j=1)^(n_2) Gamma_(i j) c(phi_i, psi_j), #h(2em)
        s.t. quad & Gamma bold(1)_(n_2) = 1 / n_1 bold(1)_(n_1), \
                  & Gamma^T bold(1)_(n_1) = 1 / n_2 bold(1)_(n_2), \
                  & Gamma_(i j) >= 0.
    $

    where $c(phi_i, psi_j) = 1 - F(ket(phi_i), ket(psi_j))$
]

#{
    set page(margin: auto, fill: auto)

    slide[
        #set align(center + horizon)
        #show heading.where(level: 1): set text(
            size: 50pt,
            weight: "regular",
            fill: main-gradient,
        )

        #grid(
            columns: (1fr, 1fr),
            [], heading[an\ *Optimizer*],
        )
    ]
}

#windowed-slide("Optimizers")[
    - Need to find the optimal parameters such that the distance between the generated and target distributions is minimized.
    - Optimizers can be broadly classified into two categories: gradient-based and gradient-free.
]

#windowed-slide("Optimizers")[
    == Gradient-based optimizers

    - Traverse the parameter space in the direction of steepest descent by computing the gradient of the chosen objective function with respect to the parameters.
    - MMD #sym.arrow automatic differentiation.
    - Wasserstein distance calculation is done through an *iterative algorithm*, and this makes it difficult for the autodiff engine to compute the gradient numerically because it has to track every single step of that loop to figure out the final gradient.

    #v(4em)

    #align(center)[_But there is an analytical simplification!_]
]

#windowed-slide("Optimizers")[
    === Envelope theorem

    Let $f (x, theta)$ and l(x) be real-valued continuously differentiable functions, where $x in RR^n$ are choice variables and $theta in RR^m$ are parameters. Denote $x^ast$ to be the optimal solution of $f$ with constraint $l = 0$ and fixed $theta$, i.e.

    $ x^ast = argmin_x f (x, theta) quad s.t. quad l(x) = 0. $

    Then, assume that $V$ is continuously differentiable function defined as $V(theta) eq.triple f (x^ast (theta), theta)$, the derivative of $V$ over parameters is

    $ pdv(V, theta) = pdv(f, theta) $
]

#windowed-slide("Optimizers")[
    - $V (theta)$ is the Wasserstein distance $W_1$. \
        The choice variable $x$ is the transport plan $Gamma$. \
        The objective function is $f (Gamma, theta) = chevron.l Gamma, C chevron.r = sum_(i j) C_(i j) Gamma_(i j)$.

    Substituting and solving:

    $
        pdv(W, theta) = - sum_(i j) Gamma_(i j)^ast pdv(, theta) abs(braket(phi_i, psi_j (theta)))^2.
    $

    *So the gradient of the Wasserstein distance can be calculated finding the optimial transport plan and the gradient of the pairwise Fidelity.*

    This gradient can be fed into any gradient-based optimizer.
]

#windowed-slide("Optimizers")[
    == Gradient-free optimizers

    #show: pause

    - Rely on evaluating the objective function at different points in the parameter space to guide the search for optimal parameters.
]

#windowed-slide("Optimizers")[
    === Rotosolve

    Rotosolve is a coordinate-wise optimization algorithm that iteratively optimizes one parameter at a time while keeping others fixed @Ostaszewski2021rotosolve using the following update rule for the $d$-th parameter:

    #let eH(x) = $expval(H)_(theta_d = #x)$
    #let halfpi = $frac(pi, 2, style: "horizontal")$

    $
        theta_d^ast = - pi/2 - "arctan2"(2 eH(0) - eH(halfpi) - eH(-halfpi), eH(halfpi) - eH(-halfpi))
    $

    where $expval(H)_(theta_d)$ denotes the expectation value of the objective function restricted to only depend on the parameter $theta_d$.

    #v(3em)

    #show: pause

    *But it's not useful for us!*

    - Only useful when the objective function can be represented as a Hermitian operator.
    - We have a distance metric between distributions, and not an expectation value of a Hermitian operator.
]

#windowed-slide("Optimizers")[
    === Simultaneous Perturbation Stochastic Approximation (SPSA)

    SPSA estimates the gradient by perturbing all parameters simultaneously in random directions @Spall1992spsa with the following update rule:

    $
        bold(theta)_(k+1) = bold(theta)_k - a_k (f (bold(theta)_k + c_k bold(Delta)_k) - f (bold(theta)_k - c_k bold(Delta)_k))/(2 c_k ) vec(Delta_(k 1)^(-1), Delta_(k 2)^(-1), dots.v, Delta_(k p)^(-1))
    $

    - Only two evaluations of the objective function per iteration are required, regardless of the number of parameters.
    - However, it can be noisy and may require careful tuning of hyperparameters $a_k$ and $c_k$.
]

#{
    set page(fill: main-gradient, margin: (x: 3cm))

    slide[
        #set align(horizon)
        #set par(leading: 0.5em, spacing: 0.5em)

        #show heading.where(level: 1): set text(
            fill: white,
            size: 50pt,
            weight: "black",
        )

        #heading[
            Quantum Generative\
            Model Architectures
        ]

        #text(
            fill: white.transparentize(20%),
            weight: "bold",
        )[for state generation]
    ]
}

#windowed-slide("Architectures")[
    == Quantum Direct Transport (QuDT)

    - Learns a direct mapping from the initial distribution of quantum states to the target distribution.
    - A single, deep Parameterized Quantum Circuit (PQC) transport the initial states to the target states. @Zhang2024quddpm

    #{
        set image(height: 140pt)
        set align(center)

        grid(
            columns: 3,
            align: center + horizon,
            image("/assets/images/quddpm/cluster-arbitrary-20.png"),
            $stretch(->, size: #6em)^text(size: #1.5em, #raw("train"))$,
            image("/assets/images/quddpm/cluster-arbitrary-0.png"),
        )
    }

    - *Toy problem:* Haar random ensemble #sym.arrow States clustered around any arbitrary state
]

#windowed-slide("Architectures")[
    #set align(center + horizon)
    #figure(
        scale(qudt-circuit, 150%, reflow: true),
        caption: [The QuDT circuit. $M_Z$ represents measurement in the Z basis.],
    )
]

#windowed-slide("Architectures")[
    == Quantum DDPM (QuDDPM)

    - An ensemble of states is scrambled over multiple steps in the forward process and then a series of PQCs are trained to reverse this scrambling process step by step in the reverse process.
    - This divides the task of learning a map from full noise to the target distribution into subtasks with low-depth circuits which can avoid barren plateaus.
]

#windowed-slide("Architectures")[
    #{
        let t = (0, 5, 10, 15, 20)

        figure(
            grid(
                columns: 5,
                ..t.map(i => text(size: 0.8em)[$t = #i$]),
                ..t.map(i => image(
                    "/assets/images/quddpm/cluster-arbitrary-"
                        + str(i)
                        + ".png",
                    height: 90pt,
                )),
            ),
            caption: [Scrambling $m=1000$ states sampled from the toy distribution over $T=20$ steps, as shown on the Bloch sphere.],
        )
    }

    #multifigure(
        row-gutter: 1em,
        figure(
            scale(forward-circuit, 120%, reflow: true),
            caption: [Forward process circuit],
        ),
        figure(
            scale(qsc-block, 120%, reflow: true),
            caption: [Quantum Scrambling Circuit (QSC) Block],
        ),
    )
]

#windowed-slide("Architectures")[
    #multifigure(
        figure(
            scale(reverse-circuit, 110%, reflow: true),
            caption: [Reverse process circuit.],
        ),
        figure(
            scale(hea-block, 110%, reflow: true),
            caption: [PQC block (HEA) used in the reverse circuit.],
        ),
    )
]

#windowed-slide("Architectures")[
    - Measurements are necessary, as the denoising map is contractive and maintains the purity of all generated data in $tilde(S)_0$.
    - No specific constraint is placed on the measurement result. The measurement on ancillas is performed and discarded; only the post-measurement state of the data qubits is collected @Zhang2024quddpm.

    #idea(title: [Connection with QuDT])[
        The QuDT model can be seen as a special case of QuDDPM with $T=1$ step in the reverse process, where the forward process is just a single random unitary that scrambles the initial distribution. This justifies the choice of ancillary qubits and their measurement in the QuDT model.
    ]
]

#windowed-slide("Architectures")[
    == Sequential Quantum Direct Transport (S-QuDT)

    This hybrid model is proposed, taking ideas from both QuDT and QuDDPM.

    - The model consists of $T$ steps, where at each step $t$ a PQC is trained to learn a direct transport map from the distribution at step $t$ to the target distribution, instead of learning a denoising map to the less scrambled distribution as in QuDDPM.

    - This removes the forward scrambling process entirely.
]

#section[Results]

#windowed-slide("Results")[
    == Clustered state generation

    #set list(spacing: 1em)

    - 4 qubit clustered states
    - 2 ancilla qubits
    - AMSGrad optimizer with learning rate $0.01$
    - $n = 1000$ samples
    - Haar distribution #sym.arrow Clustered distribution
    - Batch size = $200$
    - $T = 5$ steps
]

#windowed-slide("Results")[
    #set align(center + horizon)

    #grid(
        columns: 2,
        figure(
            image(
                "/data/cluster-data/2026-09-08_01-49-01/loss_history_fig.svg",
            ),
            caption: [QuDDPM],
        ),
        figure(
            image(
                "/data/cluster-data/2026-09-08_11-41-34/loss_history_fig.svg",
            ),
            caption: [S-QuDT],
        ),
    )
]

#windowed-slide("Results")[
    === HEA vs EHA

    #set align(center + horizon)

    #grid(
        columns: 3,
        align: center + horizon,
        figure(
            scale(hea-block, 120%, reflow: true),
            caption: [HEA block],
        ),
        text(size: 50pt)[vs],
        figure(
            stack(
                dir: ttb,
                eha-block,
                U_ent,
                U_ent-decomposed,
            ),
            caption: [EHA block],
        ),
    )
]

#windowed-slide("Results")[
    == Transverse Field Ising Model (TFIM) state generation

    #set list(spacing: 1em)

    - 9 qubit TFIM states
    - 6 ancilla qubits
    - AMSGrad optimizer with learning rate $0.02$
    - $n = 5000$ samples
    - Haar distribution #sym.arrow TFIM ground states distribution
    - Batch size = $400$
    - $T = 8$ steps
    - $14$ EHA layers per step
]

#windowed-slide("Results")[
    #grid(
        columns: 2,
        figure(
            image(
                "/data/cluster-data/2026-08-10_11-58-40/loss_history_fig.svg",
            ),
            caption: [QuDDPM],
        ),
        figure(
            image(
                "/data/cluster-data/2026-08-16_18-45-56/loss_history_fig.svg",
            ),
            caption: [S-QuDT],
        ),
    )
]

#section[Implementation]

#set page(margin: auto, fill: auto)

#slide[
    2k LoC of pure *`Julia`* with *`Yao.jl`*, with GPU acceleration using *`CUDA.jl`*.
    #set align(center + horizon)
    #image("/assets/images/loc.png")
]

#section[Conclusion\ & Future Work]

#slide[
    - S-QuDT is more efficient and matches/outperforms QuDDPM, and completely eliminates the need for a forward scrambling process.
    - EHA is a better ansatz than HEA, as it allows entanglement tuning.
    - Both S-QuDT and QuDDPM fail to converge for exponentially localized states of Quantum Kicked Rotor, so there is scope of improvement.
    - Potential to work on gradient-free optimizers for such a model.
]

#set text(size: 0.8em)

#bibliography("/references.bib", style: "american-physics-society")
