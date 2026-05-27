module QML

using CUDA

using Random
using LinearAlgebra
using Statistics
using Serialization

using Yao
import Zygote
import Optimisers

using Combinatorics: combinations
using Bessels: besselj
using StatsBase

using LaTeXStrings
using CairoMakie
using QuantumToolbox

using ProgressLogging: @progress
using TensorBoardLogger, Logging


export Device, StorageType

# I hope this is not cursed
const has_cuda = CUDA.functional()
const Device = has_cuda ? CUDA : Base
const StorageType = has_cuda ? CuArray : Array

# Force CPU irrespective of GPU availability (for testing purposes)
# const Device = Base
# const StorageType = Array

# Order is important
include("types.jl")
include("circuits.jl")
include("utils.jl")
include("helpers.jl")
include("losses.jl")
include("distributions.jl")
include("inference.jl")
include("plotting.jl")
include("logging.jl")

end
