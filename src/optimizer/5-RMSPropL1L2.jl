mutable struct RMSPropL1L2 <: Optimizer
    xparams::Vector{XVariable}
    w::Vector
    lr::AbstractFloat
    ϵ::AbstractFloat
    inertia::AbstractFloat
    L1decay::AbstractFloat
    L2decay::AbstractFloat
    name::String
    function RMSPropL1L2(xparams::Vector{XVariable}; lr=1e-2, inertia=0.99, epsilon=1e-8, L1decay=0.001, L2decay=0.01)
        n = length(xparams)
        w = Vector(undef,n)
        for i = 1:n
            c , θ = xparams[i]
            w[i] = Zeros(typeof(θ.value), θ.shape)
        end
        new(xparams, w, lr, epsilon, inertia, L1decay, L2decay, "RMSPropL1L2")
    end
end


function Base.show(io::IO, O::RMSPropL1L2)
    print("RMSPropL1L2(lr=$(O.lr), ϵ=$(O.ϵ), inertia=$(O.inertia), L1decay=$(O.L1decay), L2decay=$(O.L2decay))")
end


function update!(O::RMSPropL1L2; clipfn::Function=LPInfNormClip, clipvalue=10.0)
    w = O.w
    ϵ = O.ϵ
    μ = - O.lr
    ρ = O.inertia
    λ₁ = O.L1decay
    λ₂ = O.L2decay

    for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        ∇ = clipfn(setNanInfZero(θ.delta), clipvalue)
        @. w[i] = ρ * w[i] + (1-ρ) * ∇ * ∇
        if c == 'w'
            @. θ.value += μ / (sqrt(w[i])+ϵ) * (∇ + λ₁ * sign(θ.value) + λ₂ * θ.value)
        else
            @. θ.value += μ / (sqrt(w[i])+ϵ) * ∇
        end
    end
end
