#set page(margin: 1em)

== Parameters

```
n_qubits:  4
n_ancilla: 2
n_layers:  60 (Total)
```

#set align(horizon)
#set rotate(-90deg, reflow: true)

#table(
  columns: (3em, 1fr, 1fr),
  align: center,
  [], [`QNSPSA`], [`AMSGrad`],
  rotate([*Direct* \ (with Intermediate measurements)]),
  image("../../simulations/3-qml-jl/saves/sweep_2026-03-21_025653/2026-03-21_032720_475_DirectQNSPSA/training_plot.png"),
  image("../../simulations/3-qml-jl/saves/sweep_2026-03-21_140144/2026-03-21_140717_424_AMSGrad/training_plot.png"),
  rotate([*Direct* \ (without Intermediate measurements)]),
  image("../../simulations/3-qml-jl/saves/sweep_2026-03-21_031834/2026-03-21_034720_841_DirectQNSPSA/training_plot.png"),
  image("../../simulations/3-qml-jl/saves/sweep_2026-03-21_133313/2026-03-21_134542_100_AMSGrad/training_plot.png"),
  rotate([*Layerwise*]),
  image("../../simulations/3-qml-jl/saves/sweep_2026-03-21_143753/2026-03-21_145725_678_QNSPSA/training_plot.png"),
  image("../../simulations/3-qml-jl/saves/sweep_2026-03-21_151614/2026-03-21_152013_479_AMSGrad/training_plot.png"),
)

