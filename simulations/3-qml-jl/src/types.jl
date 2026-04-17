export TrainingStrategy, StepwiseStrategy, DirectStrategy

abstract type TrainingStrategy end
abstract type StepwiseStrategy <: TrainingStrategy end
abstract type DirectStrategy <: TrainingStrategy end


export ConcreteArrayReg, ConcreteBatchedArrayReg, Ensemble, OffsetEnsemble, OffsetEnsembleCollection

const ConcreteArrayReg = ArrayReg{2, ComplexF64, Matrix{ComplexF64}}
const ConcreteBatchedArrayReg = BatchedArrayReg{2, ComplexF64, Matrix{ComplexF64}}
const Ensemble = Vector{ConcreteArrayReg}
const OffsetEnsemble = OffsetVector{ConcreteArrayReg, Ensemble}
const OffsetEnsembleCollection = OffsetMatrix{ConcreteArrayReg, Matrix{ConcreteArrayReg}}


export Distribution, clustered, qkrlocalized

abstract type Distribution end
struct Clustered <: Distribution end
struct QKRLocalized <: Distribution end
const clustered = Clustered()
const qkrlocalized = QKRLocalized()
