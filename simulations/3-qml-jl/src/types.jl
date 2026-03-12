export ConcreteArrayReg, ConcreteBatchedArrayReg, Ensemble, OffsetEnsemble, OffsetEnsembleCollection, TrainingStrategy, StepwiseStrategy, DirectStrategy, denoise

abstract type TrainingStrategy end
abstract type StepwiseStrategy <: TrainingStrategy end
abstract type DirectStrategy <: TrainingStrategy end

const ConcreteArrayReg = ArrayReg{2, ComplexF64, Matrix{ComplexF64}}
const ConcreteBatchedArrayReg = BatchedArrayReg{2, ComplexF64, Matrix{ComplexF64}}

const Ensemble = Vector{ConcreteArrayReg}
const OffsetEnsemble = OffsetVector{ConcreteArrayReg, Ensemble}
const OffsetEnsembleCollection = OffsetMatrix{ConcreteArrayReg, Matrix{ConcreteArrayReg}}
