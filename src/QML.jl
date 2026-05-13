module QML

using CUDA

using Random
using LinearAlgebra
using Statistics

using Yao
import Zygote
import Optimisers

using Combinatorics: combinations
using Bessels: besselj
using StatsBase
using JLD2

using LaTeXStrings
using CairoMakie
using QuantumToolbox: Bloch, basis, expect, sigmax, sigmay, sigmaz, add_points!, render

using ProgressLogging: @progress


export Device, StorageType

# I hope this is not cursed
const Device = CUDA.functional() ? CUDA : Base
const StorageType = CUDA.functional() ? CuArray : Array

# Force CPU irrespective of GPU availability (for testing purposes)
# const Device = Base
# const StorageType = Array

# Order is important
include("types.jl")
include("mlflow.jl")
include("circuits.jl")
include("utils.jl")
include("helpers.jl")
include("losses.jl")
include("distributions.jl")
include("inference.jl")
include("plotting.jl")

end
