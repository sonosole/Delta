"""
    Adam(::Vector{XVariable}; lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, L1decay=0.001, L2decay=0.01)

Implements Adam algorithm. Refer to `Adam: A Method for Stochastic Optimization`.
"""
mutable struct Adam <: Optimizer
    xparams::Vector{XVariable}
    w1::Vector
    w2::Vector
    lr::AbstractFloat
    b1::AbstractFloat
    b2::AbstractFloat
    ϵ::AbstractFloat
    t::UInt
    b1t::AbstractFloat
    b2t::AbstractFloat
    L1decay::AbstractFloat
    L2decay::AbstractFloat
    name::String
    function Adam(xparams::Vector{XVariable}; lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, L1decay=0.001, L2decay=0.01)
        num = length(xparams)
        w1  = Vector(undef,num)
        w2  = Vector(undef,num)
        for i = 1:num
            c , θ = xparams[i]
            w1[i] = Zeros(typeof(ᵛ(θ)), θ.shape)
            w2[i] = Zeros(typeof(ᵛ(θ)), θ.shape)
        end
        new(xparams,w1,w2,lr, b1, b2, eps, 0, 1.0, 1.0, L1decay, L2decay, "Adam")
    end
end


function Base.show(io::IO, O::Adam)
    print("Adam(lr=$(O.lr), β₁=$(O.b1), β₂=$(O.b2), ϵ=$(O.ϵ), L1decay=$(O.L1decay), L2decay=$(O.L2decay))");
end


function update!(O::Adam; clipfn::Function=LPInfNormClip, clipvalue=10.0)
    w₁ = O.w1
    w₂ = O.w2
    lr = O.lr
    b₁ = O.b1
    b₂ = O.b2
    ϵ  = O.ϵ
    λ₁ = O.L1decay
    λ₂ = O.L2decay
    O.t   += 1
    O.b1t *= b₁
    O.b2t *= b₂
    b₁ᵗ = O.b1t
    b₂ᵗ = O.b2t

    for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        μ = - sqrt(1-b₂ᵗ) / (1-b₁ᵗ) * lr
        ∇ = clipfn(setNanInfZero(δ(θ)), clipvalue)
        𝒗 = ᵛ(θ)
        @. w₁[i] = b₁ * w₁[i] + (1-b₁) * ∇
        @. w₂[i] = b₂ * w₂[i] + (1-b₂) * ∇ * ∇

        Δ = @. w₁[i] / sqrt(w₂[i] + ϵ)
        if c == 'w'
            if λ₁==0 && λ₂==0
                @. 𝒗 += μ * Δ
            elseif λ₁==0 && λ₂!=0
                @. 𝒗 += μ * (Δ + λ₂ * 𝒗)
            elseif λ₁!=0 && λ₂==0
                @. 𝒗 += μ * (Δ + λ₁ * sign(𝒗))
            else  # λ₁!=0 && λ₂!=0
                @. 𝒗 += μ * (Δ + λ₁ * sign(𝒗) + λ₂ * 𝒗)
            end
        else
            @. 𝒗 += μ * Δ
        end
    end
end
