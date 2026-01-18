#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/physica:0.9.7": *
#import "@preview/quill:0.7.2": *

#import "@preview/numbly:0.1.0": numbly

#show: university-theme.with(
    aspect-ratio: "16-9",
    config-info(
        title: [Quantum DDPM],
        author: [Rishi Vora],
        date: datetime.today(),
    ),
    config-page(
        margin: (top: 2.5em, bottom: 1.3em),
    ),
)

#set heading(numbering: numbly("{1}.", default: "1.1"))
#set text(size: 0.7em)
#show figure.caption: it => text(size: 0.7em, it)

#title-slide()

= Overview

==

#columns(2)[
    #align(center)[*Classical DDPM*]

    - Gradually adds Gaussian noise to data until the data is indistinguishable from pure noise.

    - A neural network is trained to reverse this noising process by learning to denoise the data step by step.

    - Once trained, the model can generate new data by starting from random noise and iteratively denoising it.

    #colbreak()

    #align(center)[*Quantum DDPM*]

    - Gradually scrambles ensemble of quantum states until it is indistinguishable from an ensemble that is drawn from a Haar measure.

    - A Parametrized Quantum Circuit (PQC) is trained to reverse this scrambling process by learning to "denoise" the quantum states step by step.

    - Once trained, the model can generate new quantum states by starting from random states and iteratively denoising them.
]

= Forward Process

== Data

We consider the task of the generating new elements from an unknown distribution $cal(E)_0$ of quantum states, provided only a finite number of samples $cal(S)_0 = { ket(psi_i^((0))) , i = 1, dots, m }$ from the distribution.

For example, we choose $n$-qubit states clustered around $ket(0)^(times.o n)$

#figure(
    image("assets/cluster0.png"),
    caption: [$m = 1000$ single qubit states clustered around $ket(0)$],
)


== Circuit

We gradually scramble it over $T$ steps using a weight schedule $beta$.

#{
    set align(center)

    let s(t) = slice(label: [$ket(psi_i^((#t)))$ #v(7pt)], stroke: (
        paint: gray,
        dash: "dashed",
    ))
    let U(t) = [QSC \ $ U_i^((#t)) $]

    quantum-circuit(
        lstick($ket(psi_i^((0)))$),
        setwire(4, wire-distance: 1.3pt),
        U(1),
        1,
        s(1),
        1,
        midstick($dots.c$),
        1,
        s([t-1]),
        1,
        U([t]),
        1,
        s([t]),
        1,
        midstick($dots.c$),
        1,
        s([T-1]),
        1,
        U([T]),
        rstick($ket(psi_i^((T)))$),
        setwire(4, wire-distance: 1.3pt),
        scale: 150%,
    )
}

where each Quantum Scrambling Circuit (QSC) is like:

#{
    set align(center)

    let nq = 3

    quantum-circuit(
        lstick($ket(psi_i^((t-1)))$, n: nq, x: 0, pad: 1em, brace: "["),
        ..range(nq).map(i => gate($R_X$, y: i, x: 2)),
        ..range(nq).map(i => gate($R_Y$, y: i, x: 3)),
        ..range(nq).map(i => gate($R_Z$, y: i, x: 4)),
        mqgate(
            extent: 1em,
            rotate(90deg, reflow: true)[$
                product_(k_1 < k_2) R Z Z_(k_1, k_2)
            $],
            n: nq,
            x: 5,
        ),
        rstick($ket(psi_i^((t)))$, n: nq, x: 7, pad: 1em, brace: "]"),
        scale: 150%,
    )
}

each $R$ gate angle $tilde beta_t thin cal(U) display((- (pi)/ 8, (pi)/ 8))$ #h(1fr) and #h(1fr) each $R Z Z$ gate angle $tilde display((beta_t)/(2 sqrt(n))) thin cal(U) (0.4, 0.6)$

#pagebreak()

We calculate mean fidelity of the states at each step with respect to $ket(0)^(times.o n)$ to quantify the scrambling:

$
    dash(F) (cal(E)_1, cal(E)_2) = bb(E)_(ket(phi) tilde cal(E)_1, ket(psi) tilde cal(E)_2) [ thin |braket(phi, psi)|^2 thin ]
$

After $T$ steps, the states $cal(S)_T = { ket(psi_i^((T))) , i = 1, dots, m }$ are approximately Haar random.

#figure(
    image("assets/fidelity_evolution.png", height: 60%),
    caption: [Gradual decrease in fidelity of single qubit states to $frac(1, 2, style: "skewed")$  over $T = 20$ steps.],
)

= Backward Process

== Model

The aim is to train a series of Parametrized Quantum Circuits (PQC) to reverse the scrambling process, step by step. We start from a Haar random state $ket(psi_i^((T)))$ and apply a PQC at each step to denoise it.

Our loss function is defined as:

$
    cal(D)_"MMD" (cal(E)_1, cal(E)_2) = dash(F) (cal(E)_1, cal(E)_1) +  dash(F) (cal(E)_2, cal(E)_2) - 2 dash(F) (cal(E)_1, cal(E)_2)
$

For practical purposes, we use our finite samples $cal(S)_1$ and $cal(S)_2$ to compute MMD.

We train each PQC over many epochs. In each epoch, we sample a small batch from the scrambled states $cal(S)_(t+1)$ and optimize the MMD loss with respect to the target noising states $cal(S)_t$.


== Circuit

#{
    import tequila as tq
    set align(center)

    let s(t, x) = slice(
        label: [$ket(psi_i^((#t)))$ #v(7pt)],
        stroke: (
            paint: gray,
            dash: "dashed",
        ),
        n: 1,
        x: x,
    )
    let U(t, x) = mqgate([PQC \ $ tilde(U)_i^((#t)) $], n: 2, x: x)

    let mz = box(align(center)[$M_Z$], width: 2.5em, stroke: 0.5pt + black, inset: 0.5em)

    grid(
        columns: 6,
        align: bottom,
        quantum-circuit(
            setwire(4, wire-distance: 1.3pt),
            lstick($ket(tilde(psi)_i^((0)))$),
            lstick(mz, y: 1, x: 0),
            ..tq.build(
                tq.mqgate(0, n:2, [PQC \ $ tilde(U)_i^((0)) $]),
            ),
            s([1], 2),
            [\ ], setwire(4, wire-distance: 1.3pt),
            rstick($ket(0)^(times.o n_a)$, x: 2, y: 1),
            scale: 130%,
        ),
        grid.cell(place(horizon + center, [. . . . .], dy: -10pt)),
        quantum-circuit(
            setwire(4, wire-distance: 1.3pt),
            slice(
                label: [$ket(psi_i^((t)))$ #v(7pt)],
                stroke: (paint: gray, dash: "dashed",),
                n: 1,
                x: 1,
            ),
            lstick(mz, y: 1, x: 0),
            ..tq.build(
                tq.mqgate(0, n:2, [PQC \ $ tilde(U)_i^((t+1)) $]),
            ),
            s([t+1], 2),
            [\ ], setwire(4, wire-distance: 1.3pt),
            rstick($ket(0)^(times.o n_a)$, x: 2, y: 1),
            scale: 130%,
        ),
        grid.cell(place(horizon + center, [. . . . .], dy: -10pt)),
        quantum-circuit(
            setwire(4, wire-distance: 1.3pt),
            slice(
                label: [$ket(psi_i^((T-1)))$ #v(7pt)],
                stroke: (paint: gray, dash: "dashed",),
                n: 1,
                x: 1,
            ),
            lstick(mz, y: 1, x: 0),
            ..tq.build(
                tq.mqgate(0, n:2, [PQC \ $ tilde(U)_i^((T)) $]),
            ),
            rstick($ket(tilde(psi)_i^((T)))$, x: 2, y: 0),
            [\ ], setwire(4, wire-distance: 1.3pt),
            rstick($ket(0)^(times.o n_a)$, x: 2, y: 1),
            scale: 130%,
        ),
    )

}

where each PQC is like:

#{
    import tequila as tq
    set align(center)

    let nq = 3
    let mz = box(align(center)[$M_Z$], width: 2.5em, stroke: 0.5pt + black, inset: 0.5em)

    quantum-circuit(
        slice(
            label: [$ket(psi_i^((t)))$ #v(7pt)],
            stroke: (paint: gray, dash: "dashed",),
            n: 4,
            x: 1,
        ),
        slice(
            label: [$ket(psi_i^((t+1)))$ #v(7pt)],
            stroke: (paint: gray, dash: "dashed",),
            n: 4,
            x: 7,
        ),
        ..range(4, 6).map(i => lstick(mz, y: i, x: 0)),
        ..range(6).map(i => gate($R_Y$, y: i, x: 4)),
        ..range(6).map(i => gate($R_X$, y: i, x: 5)),
        mqgate($Z$, y:0, x:3, target: 1),
        mqgate($Z$, y:1, x:2, target: 1),
        mqgate($Z$, y:2, x:3, target: 1),
        mqgate($Z$, y:3, x:2, target: 1),
        mqgate($Z$, y:4, x:3, target: 1),
        rstick($ket(0)^(times.o n_a)$, n: 2, y: 4, x: 7),
        gategroup(6, 4, x:2, label: [repeat for L layers], stroke: gray),
        scale: 110%,
    )
}

= Note

==

*Notable differences compared to the original paper*:
- Their noising process was not Markovian. They were generating all states from scratch at each step.

#v(3em)

*Scopes and Limitations*:
- Loss function utilizes fidelity estimation which scales poorly with number of qubits. Look for alternative loss functions.
- Learn multiple distributions and generate states according to input requesting a specific distribution.
