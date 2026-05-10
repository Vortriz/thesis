module QML

using CUDA
using Random
using LinearAlgebra
using Statistics

using Yao
using Zygote
import Optimisers

using Combinatorics
using Bessels
using StatsBase
using OptimalTransport
using JLD2

using LaTeXStrings
using CairoMakie
using QuantumToolbox: Bloch, basis, expect, sigmax, sigmay, sigmaz, add_points!, render

using ProgressLogging

export Device, StorageType
# I hope this is not cursed
const Device = CUDA.functional() ? CUDA : Base
const StorageType = CUDA.functional() ? CuArray : Array

# Order is important
include("types.jl")
include("circuits.jl")
include("utils.jl")
include("losses.jl")
include("distributions.jl")
include("inference.jl")
include("plotting.jl")

end
