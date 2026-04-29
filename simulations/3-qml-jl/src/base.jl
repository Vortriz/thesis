module QDDPM

using Random
using Yao
using Combinatorics
using OffsetArrays
using Statistics
using LinearAlgebra
using CairoMakie
using QuantumToolbox: Bloch, basis, expect, sigmax, sigmay, sigmaz, add_points!, render, rand_unitary
using Zygote, Enzyme
import Optimisers
using StatsBase
using OptimalTransport
using ProgressLogging
using Dates, Statistics
using Bessels
using JLD2
using LaTeXStrings

# Order is important
include("types.jl")
include("model.jl") # Model must be defined before utils.jl which uses it for type hints
include("utils.jl")
include("losses.jl") # Depends on utils.jl (Matrix(ensemble))
include("forward.jl")
include("training_strategies/direct/base.jl")
include("training_strategies/layerwise/base.jl")
include("evaluation.jl")
include("plotting.jl")

end
