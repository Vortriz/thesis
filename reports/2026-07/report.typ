#import "@preview/kunskap:0.1.0": *
#import "@preview/physica:0.9.8": *
#import "../../assets/components/quddpm-circuits.typ": (
    U_ent, U_ent-decomposed, eha-block, hea-block,
)

#show: kunskap.with(
    title: [Report: July],
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

The S-QuDT model demonstrated faster convergence than QuDDPM on Transverse Field Ising Model (TFIM) states upto 6 qubits. It failed to converge past that for unclear reasons. So the model also failed to converge in the case of dynamically localized Quantum Kicked Rotor (QKR) states of 10 qubits.

= My Work

== Quantum Autoencoder (QAE)

A QAE is a type of neural network that can be used to compress and decompress quantum states.

#figure(
    image("/assets/images/qae.png"),
    caption: [Quantum Autoencoder \ Source: #link("https://qiskit-community.github.io/qiskit-machine-learning/tutorials/12_quantum_autoencoder.html")[Qiskit Community blog]],
)

The problem is that all proposed QAEs so far are designed for pure states, not an ensemble of pure states. So it was not clear how to use a QAE to compress the states for our purposes.

== Scaling beyond 6 qubit systems

In the quest to scale beyond 6 qubit systems, I came across a technical note @Grant2019initialization that proposed a technique for initializing the parameters of PQC.

So far, the paramters for the model were initialized from a uniform distribution in the range $[0, 1)$. The technique proposed involves randomly selecting some of the initial parameter values, then choosing the remaining values so that the circuit is a sequence of shallow blocks that each evaluates to the identity. This initialization limits the effective depth of the circuits used to calculate the first parameter update so that they cannot be stuck in a
barren plateau at the start of training.

Before implementing this technique, I did another wide refactoring of the codebase. Then I implemented the initialization technique and ran the training for 7 qubit TFIM states for four variants:

- Variant 1: EHA (Normal) with random parameter initialization
- Variant 2: EHA (Normal) with identity parameter initialization
- Variant 3: EHA (Identity) with random parameter initialization
- Variant 4: EHA (Identity) with identity parameter initialization

#grid(
    columns: 2,
    figure(
        image("/assets/images/tfim/7q-direct/loss_history_normal_rand.svg"),
        caption: "Variant 1",
    ),
    figure(
        image("/assets/images/tfim/7q-direct/loss_history_normal_identity.svg"),
        caption: "Variant 2",
    ),

    figure(
        image("/assets/images/tfim/7q-direct/loss_history_identity_rand.svg"),
        caption: "Variant 3",
    ),
    figure(
        image(
            "/assets/images/tfim/7q-direct/loss_history_identity_identity.svg",
        ),
        caption: "Variant 4",
    ),
)

The results are surprising. The model that earlier refused to converge for 7 qubit TFIM states now converges just fine, and infact, Variant 1 (the exact same model as before) converges better than the other three variants. This is counter-intuitive, as the identity initialization was supposed to help with convergence. This same trend was observed up till 9 qubit TFIM states.

My best guess for the convergence for Variant 1 is that the refactor of the codebase removed a previously unknown bug within the model.

= Planned work

- Testing the model for 10 qubit states.
- Testing the model on generation of Quantum Kicked Rotor (QKR) states of either 9 or 10 qubits.

#bibliography("/references.bib")
