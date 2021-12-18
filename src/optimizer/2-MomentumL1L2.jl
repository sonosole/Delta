"""
    Momentum(::Vector{XVariable}; lr=1e-4, inertia=0.9, L1decay=0.001, L2decay=0.01)

Implements stochastic gradient descent with momentum
"""
mutable struct Momentum <: Optimizer
    xparams::Vector{XVariable}
    v::Vector
    lr::AbstractFloat
    inertia::AbstractFloat
    L1decay::AbstractFloat
    L2decay::AbstractFloat
    name::String
    function Momentum(xparams::Vector{XVariable}; lr=1e-4, inertia=0.9, L1decay=0.001, L2decay=0.01)
        num = length(xparams)
        vel = Vector(undef,num)
        for i = 1:num
            c , θ = xparams[i]
            vel[i] = Zeros(typeof(ᵛ(θ)), θ.shape)
        end
        new(xparams, vel, lr, inertia, L1decay, L2decay, "Momentum")
    end
end


function Base.show(io::IO, O::Momentum)
    print("Momentum(lr=$(O.lr), inertia=$(O.inertia), L1decay=$(O.L1decay), L2decay=$(O.L2decay))")
end


function update!(O::Momentum; clipfn::Function=LPInfNormClip, clipvalue=10.0)
    vel = O.v
    μ = - O.lr
    ρ = O.inertia
    λ₁ = O.L1decay
    λ₂ = O.L2decay

    for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        ∇ = clipfn(setNanInfZero(δ(θ)), clipvalue)
        𝒗 = ᵛ(θ)
        @. vel[i] = ρ * vel[i] + ∇
        if c == 'w'
            if λ₁==0 && λ₂==0
                @. 𝒗 += μ * vel[i]
            else if λ₁==0 && λ₂!=0
                @. 𝒗 += μ * (vel[i] + λ₂ * 𝒗)
            else if λ₁!=0 && λ₂==0
                @. 𝒗 += μ * (vel[i] + λ₁ * sign(𝒗))
            else  # λ₁!=0 && λ₂!=0
                @. 𝒗 += μ * (vel[i] + λ₁ * sign(𝒗) + λ₂ * 𝒗)
            end
        else
            @. 𝒗 += μ * vel[i]
        end
    end
end
