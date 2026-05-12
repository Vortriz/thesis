# Notes to Self

- `append_qubits!` adds zero state qubits to the register but it places them at the higher indices. Eg - Adding 3 qubits to |psi> -> |000> ⊗ |psi>.
- so to `measure` the ancilla qubits, we need to measure the higher indices. Eg - `measure(qreg, 3:5)` to measure the 3 ancilla qubits.
