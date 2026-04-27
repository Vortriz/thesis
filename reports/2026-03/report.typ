#import "@preview/kunskap:0.1.0": *
#import "@preview/physica:0.9.8": *

#show: kunskap.with(
  title: [Report: March],
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

In February, I focused on making the Julia QDDPM implementation fast. I fixed major Zygote/Yao autodiff failures, switched to a Wasserstein objective with explicit Zygote gradients using the envelope theorem, tried out several optimizers (gradient-based and gradient-free).

= My Work

== Benchmarking Optimizers

After a successful implementation of gradient-based optimization using the envelope theorem, I benchmarked several optimizers on a 4-qubit system. The optimizers included:

- Adam (baseline)
- RAdam
- NAdam
- AdamW
- AdaBelief
- Lion
- AMSGrad

The results showed that AdaBelief and AMSGrad performed quite better than the baseline Adam optimizer.

Taking these two optimizers forward, I benchmarked them on a 6-qubit system. AMSGrad performed better and was hence chosen as the optimizer for training larger systems.

== Comparision with QuDT

I compared the performance of my implementation with Quantum Direct Transport (QuDT) model as per described in @Zhang2024quddpm. The basic idea to train all the parameters to minimize the distance to the target ensemble. Out of curiosity, I also tweaked the QuDT to include intermediate measurements on ancilla, with the intuition that it would help dump the entropy of the system to make the convergence better.

#align(
  center,
  image(
    "/presentations/2026-03-21/presentation.pdf",
    height: 80%,
  ),
)

It certainly performed better than the vanilla QuDT, but it was still not good enough. Curiously, the gradient-less QNSPSA optimizer performed better than the gradient-based AMSGrad optimizer. But it still no match to AMSGrad with step-wise training. With 6-qubits, the gap widens even more; AMSGrad converged far quicker and better than the QuDT-based approaches.

Hence, the conclusion is that the step-wise training approach with a Wasserstein objective and explicit gradients has a much better chance of working for larger as well as chaotic systems.

== Kicked Rotator

I have started reading up the theory of the kicked rotator and the Quantum kicked rotator.

= Planned Work

To understand the theory of the kicked rotator. Also to do preliminary testing of 10-qubit systems with the AMSGrad optimizer. Then try to generate Quantum kicked rotator states with it.

#bibliography("/references.bib")
