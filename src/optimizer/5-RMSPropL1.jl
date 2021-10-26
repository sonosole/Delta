mutable struct RMSPropL1 <: Optimizer
    xparams::Vector{XVariable}
    w::Vector
    lr::AbstractFloat
    ϵ::AbstractFloat
    inertia::AbstractFloat
    L1decay::AbstractFloat
    name::String
    function RMSPropL1(xparams::Vector{XVariable}; lr=1e-2, inertia=0.99, epsilon=1e-8, L1decay=0.001)
        n = length(xparams)
        w = Vector(undef,n)
        for i = 1:n
            c , θ = xparams[i]
            w[i] = Zeros(typeof(θ.value), θ.shape)
        end
        new(xparams, w, lr, epsilon, inertia, L1decay, "RMSPropL1")
    end
end


function Base.show(io::IO, O::RMSPropL1)
    print("RMSPropL1(lr=$(O.lr), ϵ=$(O.ϵ), inertia=$(O.inertia), L1decay=$(O.L1decay))")
end


function update!(O::RMSPropL1; clipfn::Function=LPInfNormClip, clipvalue=10.0)
    w = O.w
    ϵ = O.ϵ
    μ = - O.lr
    ρ = O.inertia
    λ = O.L1decay

    for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        ∇ = clipfn(setNanInfZero(θ.delta), clipvalue)
        @. w[i] = ρ * w[i] + (1-ρ) * ∇ * ∇
        if c == 'w'
            @. θ.value += μ / (sqrt(w[i])+ϵ) * (∇ + λ * sign(θ.value))
        else
            @. θ.value += μ / (sqrt(w[i])+ϵ) * ∇
        end
    end
end
