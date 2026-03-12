export ConcreteArrayReg, ConcreteBatchedArrayReg, Ensemble, OffsetEnsemble, OffsetEnsembleCollection

const ConcreteArrayReg = ArrayReg{2, ComplexF64, Matrix{ComplexF64}}
const ConcreteBatchedArrayReg = BatchedArrayReg{2, ComplexF64, Matrix{ComplexF64}}

const Ensemble = Vector{ConcreteArrayReg}
const OffsetEnsemble = OffsetVector{ConcreteArrayReg, Ensemble}
const OffsetEnsembleCollection = OffsetMatrix{ConcreteArrayReg, Matrix{ConcreteArrayReg}}
