module QDDPM

using Random
using Yao
using Combinatorics
using OffsetArrays
using Statistics
using LinearAlgebra
using WGLMakie
using QuantumToolbox: Bloch, basis, expect, sigmax, sigmay, sigmaz, add_points!, render, rand_unitary
using Zygote, Enzyme
import Optimisers
using StatsBase
using OptimalTransport
using ProgressLogging
using Dates, Statistics

# Order is important
include("types.jl")
include("utils.jl")
include("losses.jl")
include("model.jl")
include("forward.jl")
include("training_strategies/direct/default.jl")
include("training_strategies/layerwise/training.jl")
include("evaluation.jl")
include("plotting.jl")

end
