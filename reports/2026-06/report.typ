#import "@preview/kunskap:0.1.0": *
#import "@preview/physica:0.9.8": *
#import "../../assets/components/quddpm-circuits.typ": (
    U_ent, U_ent-decomposed, eha-block, hea-block,
)

#show: kunskap.with(
    title: [Report: June],
    author: "Rishi Vora",
    date: datetime.today().display(),
    header: "PRJ502",

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

The proposed S-QuDT model demonstrated an advantage over QuDDPM in terms of convergence speed while maintaining similar, if not better, final loss. This was tested on Clustered state task of upto 5-qubits.

= My Work

== Use of EHA Ansatz

So far the models have been trained using the Hardware Efficient Ansatz (HEA), which looks like:

#align(center)[
    #hea-block
    $stretch(<-, size: #10cm)$
]

But for larger systems or complex distributions, a lot of layers of HEA are required. This makes the circuit too expressive, resulting in barren plateaus @McClean2018barren @Holmes2022expressibility. A major reason is the entanglement layer in HEA is fixed.

So I explored the use of Entanglement-variational Hardware-efficient Ansatz (EHA) @Wang2024EHA which has a tunable entanglement layer:

#align(center)[
    #eha-block
    $stretch(<-, size: #10cm)$
]

where each $U_"ent"$ gate is given by:

#align(center)[
    #U_ent
    #U_ent-decomposed
]

In practice, the EHA ansatz significantly reduces the number of layers required to make the circuit expressive enough to model the target distribution. This is because the entanglement layer is now tunable, and hence can be trained to suit the target distribution.

== Quantum Kicked Rotor (QKR) States

Since the model has been tested on toy distributions and has demonstrated faster convergence than QuDDPM, larger systems are now within reach. So I tried to generate Quantum Kicked Rotor (QKR) states. These states, provided they have a large enough basis size, can demonstrate the phenomenon of dynamical localization in momentum space:

#figure(
    image("/assets/images/qkr-localization.png"),
    caption: [Localization of QKR states of 10 qubits (i.e. 1024 basis states) \ with $K = 12, hbar_s = 0.7$ after 1000 kicks. The y-axis is on a log scale.],
)

To observe this phenomenon reliably, minimum 8 qubits (i.e. 256 basis states) are required. Unfortunately, the model was unable to converge to the target distribution for 8 qubits. The loss plateaued at a value of 0.8 despite trying various optimizers, learning rates, and ansatzes. I am not sure what could be the reason for this, but I am looking into it.

#pagebreak()

== Transverse Field Ising Model (TFIM) States

To check if the reason for the model not converging to QKR states was due to the complexity of the distribution, I tried generating the ground states of Transverse Field Ising Model (TFIM). These states are a good target because:

- We can start as low as 3-4 qubits to see if the model can converge to the target distribution.
- These were also used by @Zhang2024quddpm to demonstrate the performance of QuDDPM, so it would serve as a good benchmark to compare the performance of S-QuDT with QuDDPM.

Given below are the results of training S-QuDT and QuDDPM on 4, 5, and 6 qubit TFIM ground states. Compare the final losses and the overlap of $t=6$ (final time-step) generated states with the target distribution. We can see that S-QuDT converges faster than QuDDPM.


#{
    show figure.caption: it => text(size: 0.75em, it.body)
    set grid(columns: 2, align: center, column-gutter: 1em)
    show grid: set block(breakable: false)
    set grid.vline(stroke: (paint: gray, dash: "dashed"))

    grid(
        grid.cell(colspan: 2, inset: (
            bottom: 1em,
        ))[*4-qubit TFIM ground states*],
        grid.vline(x: 1, start: 1, end: 3),
        image("/assets/images/tfim/4q-diffusion/loss_history_fig.svg"),
        image("/assets/images/tfim/4q-direct/loss_history_fig.svg"),
        image("/assets/images/tfim/4q-diffusion/generated_trajectory.svg"),
        image("/assets/images/tfim/4q-direct/generated_trajectory.svg"),
        text(size: 0.8em, weight: "semibold")[QuDDPM model],
        text(size: 0.8em, weight: "semibold")[S-QuDT model],
    )

    set page(header: none, footer: none, margin: (y: 0pt))
    set align(horizon)

    grid(
        grid.cell(colspan: 2, inset: (
            bottom: 1em,
        ))[*5-qubit TFIM ground states*],
        grid.vline(x: 1, start: 1, end: 3),
        image("/assets/images/tfim/5q-diffusion/loss_history_fig.svg"),
        image("/assets/images/tfim/5q-direct/loss_history_fig.svg"),
        image("/assets/images/tfim/5q-diffusion/generated_trajectory.svg"),
        image("/assets/images/tfim/5q-direct/generated_trajectory.svg"),
        text(size: 0.8em, weight: "semibold")[QuDDPM model],
        text(size: 0.8em, weight: "semibold")[S-QuDT model],
    )

    v(2em)

    grid(
        grid.cell(colspan: 2, inset: (
            bottom: 1em,
        ))[*6-qubit TFIM ground states*],
        grid.vline(x: 1, start: 1, end: 3),
        image("/assets/images/tfim/6q-diffusion/loss_history_fig.svg"),
        image("/assets/images/tfim/6q-direct/loss_history_fig.svg"),
        image("/assets/images/tfim/6q-diffusion/generated_trajectory.svg"),
        image("/assets/images/tfim/6q-direct/generated_trajectory.svg"),
        text(size: 0.8em, weight: "semibold")[QuDDPM model],
        text(size: 0.8em, weight: "semibold")[S-QuDT model],
    )
}

Also, QuDDPM falls significantly behind S-QuDT as we scale up to 6-qubits. But once again, neither of the models could converge for 7-qubits. There seems to be something holding of these models at that system size. This is perhaps the reason QKR states of 8-qubits could not be generated.

Based off this observation, I am currently investigating what could be the reason for this.

= Planned work

There are two possible direction to tackle the problem of convergence for larger systems:
- We can try to use Quantum Autoencoder (QAE) to reduce the system to less than 7-qubits and see if the model can converge to the target distribution.
- I would explore if there is a fundamental reason for this behaviour. I have a lead on some relevant literature that I will be reading up on.

I will explore both these directions in parallel.

#bibliography("../../references.bib")
