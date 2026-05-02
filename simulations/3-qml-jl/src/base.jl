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
include("utils.jl")
include("circuits.jl")
include("losses.jl")
include("distributions.jl")
# include("training_strategies/direct/base.jl")
# include("training_strategies/layerwise/base.jl")
include("inference.jl")
include("plotting.jl")

end
