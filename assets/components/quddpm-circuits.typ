#import "@preview/quill:0.7.2": *
#import "@preview/physica:0.9.8": *

#import tequila as tq

#let forward-circuit = {
  let s(t) = slice(label: [$ket(psi_i^((#t)))$ #v(7pt)], stroke: (
    paint: gray,
    dash: "dashed",
  ))
  let U(t) = [QSC \ $ U_i^((#t)) $]

  grid(
    row-gutter: 0pt,
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
    ),
    text(size: 20pt)[$stretch(->, size: #10cm)$],
  )
}

#let qsc-block = {
  let nq = 3

  quantum-circuit(
    lstick($ket(psi_i^((t-1)))$, n: nq, x: 0, pad: 1em, brace: "["),
    ..range(nq).map(i => gate($R_X$, y: i, x: 2)),
    ..range(nq).map(i => gate($R_Y$, y: i, x: 3)),
    ..range(nq).map(i => gate($R_X$, y: i, x: 4)),
    mqgate(
      extent: 1em,
      rotate(90deg, reflow: true)[$
        product_(k_1 < k_2) R Z Z_(k_1, k_2)
      $],
      n: nq,
      x: 5,
    ),
    rstick($ket(psi_i^((t)))$, n: nq, x: 7, pad: 1em, brace: "]"),
  )
}

#let reverse-circuit = {
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

  let mz = box(
    align(center)[$M_Z$],
    width: 2.5em,
    stroke: 0.5pt + black,
    inset: 0.5em,
  )

  grid(
    columns: 5,
    column-gutter: 0pt,
    row-gutter: 0pt,
    align: bottom,
    quantum-circuit(
      setwire(4, wire-distance: 1.3pt),
      lstick($ket(tilde(psi)_i^((0)))$),
      lstick(mz, y: 1, x: 0),
      ..tq.build(
        tq.mqgate(0, n: 2, [PQC \ $ tilde(U)_i^((0)) $]),
      ),
      s([1], 2),
      [\ ],
      setwire(4, wire-distance: 1.3pt),
      rstick($ket(0)^(times.o n_a)$, x: 2, y: 1),
    ),
    grid.cell(place(horizon + center, [. . . . .], dy: -10pt)),
    quantum-circuit(
      setwire(4, wire-distance: 1.3pt),
      slice(
        label: [$ket(psi_i^((t)))$ #v(7pt)],
        stroke: (paint: gray, dash: "dashed"),
        n: 1,
        x: 1,
      ),
      lstick(mz, y: 1, x: 0),
      ..tq.build(
        tq.mqgate(0, n: 2, [PQC \ $ tilde(U)_i^((t+1)) $]),
      ),
      s([t+1], 2),
      [\ ],
      setwire(4, wire-distance: 1.3pt),
      rstick($ket(0)^(times.o n_a)$, x: 2, y: 1),
    ),
    grid.cell(place(horizon + center, [. . . . .], dy: -10pt)),
    quantum-circuit(
      setwire(4, wire-distance: 1.3pt),
      slice(
        label: [$ket(psi_i^((T-1)))$ #v(7pt)],
        stroke: (paint: gray, dash: "dashed"),
        n: 1,
        x: 1,
      ),
      lstick(mz, y: 1, x: 0),
      ..tq.build(
        tq.mqgate(0, n: 2, [PQC \ $ tilde(U)_i^((T)) $]),
      ),
      rstick($ket(tilde(psi)_i^((T)))$, x: 2, y: 0),
      [\ ],
      setwire(4, wire-distance: 1.3pt),
      rstick($ket(0)^(times.o n_a)$, x: 2, y: 1),
    ),
    grid.cell(colspan: 5, text(size: 20pt)[$stretch(<-, size: #10cm)$]),
  )
}

#let pqc-block = {
  let mz = box(
    align(center)[$M_Z$],
    width: 2.5em,
    stroke: 0.5pt + black,
    inset: 0.5em,
  )

  quantum-circuit(
    slice(
      label: [$ket(psi_i^((t)))$ #v(7pt)],
      stroke: (paint: gray, dash: "dashed"),
      n: 4,
      x: 1,
    ),
    ..range(4, 6).map(i => lstick(mz, y: i, x: 0)),
    mqgate($Z$, y: 0, x: 4, target: 1),
    mqgate($Z$, y: 1, x: 3, target: 1),
    mqgate($Z$, y: 2, x: 4, target: 1),
    mqgate($Z$, y: 3, x: 3, target: 1),
    mqgate($Z$, y: 4, x: 4, target: 1),
    ..range(6).map(i => gate($R_Y$, y: i, x: 5)),
    ..range(6).map(i => gate($R_X$, y: i, x: 6)),
    slice(
      label: [$ket(psi_i^((t+1)))$ #v(7pt)],
      stroke: (paint: gray, dash: "dashed"),
      n: 4,
      x: 9,
    ),
    rstick($ket(0)^(times.o n_a)$, n: 2, y: 4, x: 9),
    gategroup(6, 4, x: 3, label: [repeat for L layers], stroke: (paint: gray, dash: "dashed"), padding: (top: 10pt)),
  )
}

#let qudt-circuit = {
  quantum-circuit(
    ..range(6).map(i => gate($R_X$, y: i, x: 1)),
    ..range(6).map(i => gate($R_Y$, y: i, x: 2)),
    mqgate($Z$, y: 0, x: 3, target: 1),
    mqgate($Z$, y: 1, x: 4, target: 1),
    mqgate($Z$, y: 2, x: 3, target: 1),
    mqgate($Z$, y: 3, x: 4, target: 1),
    mqgate($Z$, y: 4, x: 3, target: 1),
    ..range(4, 6).map(i => gate($M_Z$, y: i, x: 6)),
    gategroup(6, 4, x: 1, label: [repeat for L layers], stroke: (paint: gray, dash: "dashed"), padding: (top: 10pt)),
    lstick($ket(psi_i^(("initial")))$, n: 4, y: 0, x: 0),
    lstick($ket(0)^(times.o n_a)$, n: 2, y: 4, x: 0),
    rstick($ket(psi_i^(("target")))$, n: 4, y: 0, x: 6),
  )
}
