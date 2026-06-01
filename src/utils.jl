export get_centered_amplitudes

function get_centered_amplitudes(ψ::State)
    dims = length(ψ)
    amplitudes = abs2.(ψ)
    _, idx = findmax(amplitudes)
    circshift!(amplitudes, dims - idx + 1)

    return amplitudes
end


export magnetization

function magnetization(ψ::State)
    n = ψ |> length |> log2 |> Int64
    M = 0
    for (i, ψᵢ) in enumerate(ψ)
        ψᵢ_M = 0
        for spin in digits(i - 1; base=2, pad=n) |> reverse |> BitVector
            ψᵢ_M += abs2(ψᵢ) * (spin ? 1 : -1)
        end
        ψᵢ_M /= n
        @assert abs(ψᵢ_M) <= 1
        M += abs(ψᵢ_M)
    end

    return M
end
