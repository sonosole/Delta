"""
    AdaGrad(::Vector{XVariable}; lr=1e-2, eps=1e-10, L1decay=0.001, L2decay=0.01)

Implements Adagrad algorithm. Refer to `Adaptive Subgradient Methods for Online Learning and Stochastic`
"""
mutable struct AdaGrad <: Optimizer
    xparams::Vector{XVariable}
    w::Vector
    lr::AbstractFloat
    ϵ::AbstractFloat
    L1decay::AbstractFloat
    L2decay::AbstractFloat
    name::String
    function AdaGrad(xparams::Vector{XVariable}; lr=1e-2, eps=1e-10, L1decay=0.001, L2decay=0.01)
        n = length(xparams)
        w = Vector(undef,n)
        for i = 1:n
            c , θ = xparams[i]
            w[i] = Zeros(typeof(ᵛ(θ)), θ.shape)
        end
        new(xparams, w, lr, eps, L1decay, L2decay, "AdaGrad")
    end
end


function Base.show(io::IO, O::AdaGrad)
    print("AdaGrad(lr=$(O.lr), ϵ=$(O.ϵ), L1decay=$(O.L1decay), L2decay=$(O.L2decay))")
end


function update!(O::AdaGrad; clipfn::Function=LPInfNormClip, clipvalue=10.0)
    w = O.w
    μ = - O.lr
    ϵ = O.ϵ
    λ₁ = O.L1decay
    λ₂ = O.L2decay

    for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        ∇ = clipfn(setNanInfZero(δ(θ)), clipvalue)
        𝒗 = ᵛ(θ)
        @. w[i] += ∇ * ∇
        if c == 'w'
            if λ₁==0 && λ₂==0
                @. 𝒗 += μ / (sqrt(w[i]) + ϵ) * ∇
            else if λ₁==0 && λ₂!=0
                @. 𝒗 += μ / (sqrt(w[i]) + ϵ) * (∇ + λ₂ * 𝒗)
            else if λ₁!=0 && λ₂==0
                @. 𝒗 += μ / (sqrt(w[i]) + ϵ) * (∇ + λ₁ * sign(𝒗))
            else  # λ₁!=0 && λ₂!=0
                @. 𝒗 += μ / (sqrt(w[i]) + ϵ) * (∇ + λ₁ * sign(𝒗) + λ₂ * 𝒗)
            end
        else
            @. 𝒗 += μ / (sqrt(w[i]) + ϵ) * ∇
        end
    end
end
