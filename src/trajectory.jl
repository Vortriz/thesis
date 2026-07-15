export ArbitraryTrajectory

struct ArbitraryTrajectory <: AbstractTrajectory
    steps::Vector{AbstractDist}
end


export Direct

struct Direct <: AbstractTrajectory
    steps::Vector{AbstractDist}

    function Direct(;
        initial_dist::AbstractDist,
        target_dist::AbstractDist,
    )
        return new(AbstractDist[initial_dist, target_dist])
    end
end


export Diffusion, scramble_circuit

RZZ(n::Int64, i::Int64, j::Int64)::ChainBlock{2} =
    chain(n, control(i, j => X), put(j => Rz(0)), control(i, j => X))

function scramble_circuit(n_qubits::Int64)::ChainBlock{2}
    register = 1:n_qubits
    circuit = chain(n_qubits)

    for i in register
        push!(circuit, put(i => Rz(0)))
        push!(circuit, put(i => Ry(0)))
        push!(circuit, put(i => Rz(0)))
    end

    RZZ_combinations = combinations(register, 2)
    for (i, j) in RZZ_combinations
        push!(circuit, RZZ(n_qubits, i, j))
    end

    return Optimise.canonicalize(circuit)
end

struct Diffusion <: AbstractTrajectory
    steps::Vector{AbstractDist}

    function Diffusion(;
        target_dist::AbstractDist,
        weight_schedule::Vector{Float64},
    )
        T = length(weight_schedule)
        n_qubits = target_dist.n_qubits
        circuit = scramble_circuit(n_qubits)

        steps = Vector{AbstractDist}(undef, T + 1)
        steps[begin] = deepcopy(target_dist)

        for t in 1:T
            reg = deepcopy(target_dist.register)

            for r in 1:reg.nbatch
                reg_view = viewbatch(reg, r)
                # Run through all steps up to the current timestep t
                for prev_t in 1:t
                    # Generate random parameters scaled by the weight schedule for this step
                    params = vcat(
                        weight_schedule[prev_t] .*
                        (rand(RNG, Float64, n_qubits * 3) .* (π / 4) .- (π / 8)),
                        weight_schedule[prev_t] .*
                        (rand(RNG, Float64, binomial(n_qubits, 2)) .* 0.2 .+ 0.4) ./
                        (2.0 * sqrt(n_qubits)),
                    )

                    dispatch!(circuit, params)
                    apply!(reg_view, circuit)
                end
            end

            steps[t+1] = ArbitraryDist(reg)
        end

        steps[end] = HaarDist(steps[end].register)
        reverse!(steps)

        return new(steps)
    end
end
