module GQML

using Random
using LinearAlgebra
using Statistics

using Yao
import Zygote
import Optimisers

using Combinatorics: combinations
using Bessels: besselj
using StatsBase

using LaTeXStrings
using CairoMakie
using QuantumToolbox:
    QuantumToolbox as QT,
    AbstractQuantumObject, Operator

using ProgressLogging: @progress   # [TODO] get rid of this
using PlutoSerialization


# Module's own RNG
const RNG = Xoshiro(6868)

# Order is important
include("circuits.jl")
include("types.jl")
include("distributions.jl")
include("trajectory.jl")
include("measurement.jl")
include("arch.jl")
include("losses.jl")
include("utils.jl")
include("plotting.jl")
include("model.jl")
include("save.jl")

end
