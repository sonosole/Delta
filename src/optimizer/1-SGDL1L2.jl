"""
    SGD(::Vector{XVariable}; lr=1e-4, L1decay=0.001, L2decay=0.01)

Implements stochastic gradient descent
"""
mutable struct SGD <: Optimizer
    xparams::Vector{XVariable}
    lr::AbstractFloat
    L1decay::AbstractFloat
    L2decay::AbstractFloat
    name::String
    function SGD(xparams::Vector{XVariable}; lr=1e-4, L1decay=0.001, L2decay=0.01)
        new(xparams, lr, L1decay, L2decay, "SGD")
    end
end

# pretty printing
function Base.show(io::IO, O::SGD)
    print("SGD(lr=$(O.lr), L1decay=$(O.L1decay), L2decay=$(O.L2decay))")
end


function update!(O::SGD; clipfn::Function=LPInfNormClip, clipvalue=10.0)
    μ = - O.lr
    λ₁ = O.L1decay
    λ₂ = O.L2decay

    for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        ∇ = clipfn(setNanInfZero(δ(θ)), clipvalue)
        𝒗 = ᵛ(θ)
        if c == 'w'
            if λ₁==0 && λ₂==0
                @. 𝒗 += μ * ∇
            elseif λ₁==0 && λ₂!=0
                @. 𝒗 += μ * (∇ + λ₂ * 𝒗)
            elseif λ₁!=0 && λ₂==0
                @. 𝒗 += μ * (∇ + λ₁ * sign(𝒗))
            else  # λ₁!=0 && λ₂!=0
                @. 𝒗 += μ * (∇ + λ₁ * sign(𝒗) + λ₂ * 𝒗)
            end
        else
            @. 𝒗 += μ * ∇
        end
    end
end
