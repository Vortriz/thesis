#import "@preview/quill:0.7.3": *
#import "@preview/physica:0.9.8": *

#import tequila as tq

#let mz = box(
    align(center)[$M_Z$],
    width: 2.5em,
    stroke: 0.5pt + black,
    inset: 0.5em,
)

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

#let hea-block = {
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
        gategroup(
            6,
            4,
            x: 3,
            label: [repeat for L layers],
            stroke: (paint: gray, dash: "dashed"),
            padding: (top: 10pt),
        ),
    )
}

#let eha-block = {
    quantum-circuit(
        slice(
            label: [$ket(psi_i^((t)))$ #v(7pt)],
            stroke: (paint: gray, dash: "dashed"),
            n: 4,
            x: 1,
        ),
        ..range(4, 6).map(i => lstick(mz, y: i, x: 0)),
        mqgate($U_("ent")$, y: 0, x: 3, n: 2),
        mqgate($U_("ent")$, y: 1, x: 2, n: 2),
        mqgate($U_("ent")$, y: 2, x: 3, n: 2),
        mqgate($U_("ent")$, y: 3, x: 2, n: 2),
        mqgate($U_("ent")$, y: 4, x: 3, n: 2),
        ..range(6).map(i => gate($R_Z$, y: i, x: 4)),
        ..range(6).map(i => gate($R_Y$, y: i, x: 5)),
        ..range(6).map(i => gate($R_Z$, y: i, x: 6)),
        slice(
            label: [$ket(psi_i^((t+1)))$ #v(7pt)],
            stroke: (paint: gray, dash: "dashed"),
            n: 4,
            x: 9,
        ),
        rstick($ket(0)^(times.o n_a)$, n: 2, y: 4, x: 9),
        gategroup(
            6,
            5,
            x: 2,
            label: [repeat for L layers],
            stroke: (paint: gray, dash: "dashed"),
            padding: (top: 10pt),
        ),
    )
}

#let U_ent = {
    grid(
        columns: 3,
        align: horizon,
        quantum-circuit(
            wires: 2,
            1,
            mqgate($U_("ent")$, n: 2),
            1,
        ),
        [=],
        quantum-circuit(
            wires: 2,
            1,
            mqgate($X X$, n: 2),
            mqgate($Y Y$, n: 2),
            mqgate($Z Z$, n: 2),
            1,
        )
    )
}

#let U_ent-decomposed = {
    set math.frac(style: "horizontal")
    set grid(align: horizon)

    let XX = grid(
        columns: 3,
        quantum-circuit(
            wires: 2,
            1,
            mqgate($X X (phi)$, n: 2),
            1,
        ),
        [=],
        quantum-circuit(
            ctrl(1), gate($R_X (phi)$), ctrl(1), [\ ],
            targ(), 1, targ(),
        )
    )

    let ZZ = grid(
        columns: 3,
        quantum-circuit(
            wires: 2,
            1,
            mqgate($Z Z (phi)$, n: 2),
            1,
        ),
        [=],
        quantum-circuit(
            ctrl(1), 1, ctrl(1), [\ ],
            targ(), gate($R_Z (phi)$), targ(),
        )
    )

    let YY = grid(
        columns: 3,
        quantum-circuit(
            wires: 2,
            1,
            mqgate($Y Y (phi)$, n: 2),
            1,
        ),
        [=],
        quantum-circuit(
            1, gate($R_X (pi/2)$), ctrl(1), 1, ctrl(1), gate($R_X (-pi/2)$), 1, [\ ],
            1, gate($R_X (pi/2)$), targ(), gate($R_Z (phi)$), targ(), gate($R_X (-pi/2)$), 1,
        )
    )

    grid(
        columns: 2,
        stroke: gray,
        gutter: 0.5em,
        inset: 0.5em,
        XX,
        ZZ,
        grid.cell(
            colspan: 2,
            YY,
        ),
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
        gategroup(
            6,
            4,
            x: 1,
            label: [repeat for L layers],
            stroke: (paint: gray, dash: "dashed"),
            padding: (top: 10pt),
        ),
        lstick($ket(psi_i^(("initial")))$, n: 4, y: 0, x: 0),
        lstick($ket(0)^(times.o n_a)$, n: 2, y: 4, x: 0),
        rstick($ket(psi_i^(("target")))$, n: 4, y: 0, x: 6),
    )
}
