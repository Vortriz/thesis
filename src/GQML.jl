module GQML

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

using LaTeXStrings
using CairoMakie
using QuantumToolbox:
    QuantumToolbox as QT,
    eye, sigmax, sigmay, sigmaz, Qobj, basis,
    kron, expect, eigenstates,
    AbstractQuantumObject, Operator

using ProgressLogging: @progress
using TensorBoardLogger: TBLogger, log_text
using PlutoSerialization


export Device, StorageType

# [TODO] move CUDA to ext
const has_cuda = CUDA.functional()
const Device = has_cuda ? CUDA : Base
const StorageType = has_cuda ? CuArray : Array

# Force CPU irrespective of GPU availability (for testing purposes)
# const Device = Base
# const StorageType = Array

# Module's own RNG
const RNG = Xoshiro(6868)

# Order is important
include("circuits.jl")
include("types.jl")
include("losses.jl")
include("utils.jl")
include("distributions.jl")
include("plotting.jl")
include("model.jl")
include("logging.jl")

end
