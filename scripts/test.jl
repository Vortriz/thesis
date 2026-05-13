using QML
using Random
using Optimisers


const T = 2
const rng = MersenneTwister(1234)

arch = ModelArch(;
    n_data=2,
    n_ancilla=1,
    n_layers=2,
    ansatz_builder=EHA,
    collapse_method=normal,
)

initial_ensemble = gen_dist(
    Val(haar),
    rng;
    n_qubits=arch.n_data,
    n_samples=100,
)
target_ensemble = gen_dist(
    Val(clustered),
    rng;
    n_qubits=arch.n_data,
    n_samples=1000,
)

config = TrainConfig(
    Val(direct);
    initial_ensemble=initial_ensemble,
    target_ensemble=target_ensemble,
    epoch_schedule=fill(300, T),
    optimizer=Optimisers.AMSGrad(0.01),
)

get_hyperparams(arch, config, rng) |> println
