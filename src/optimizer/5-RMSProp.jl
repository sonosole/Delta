mutable struct RMSProp <: Optimizer
    xparams::Vector{XVariable}
    w::Vector
    lr::AbstractFloat
    ϵ::AbstractFloat
    inertia::AbstractFloat
    name::String
    function RMSProp(xparams::Vector{XVariable}; lr=1e-2, inertia=0.99, epsilon=1e-8)
        n = length(xparams)
        w = Vector(undef,n)
        for i = 1:n
            c , θ = xparams[i]
            w[i] = Zeros(typeof(θ.value), θ.shape)
        end
        new(xparams, w, lr, epsilon, inertia, "RMSProp")
    end
end


function Base.show(io::IO, O::RMSProp)
    print("RMSProp(lr=$(O.lr), ϵ=$(O.ϵ), inertia=$(O.inertia))")
end


function update!(O::RMSProp; clipfn::Function=LPInfNormClip, clipvalue=10.0)
    w = O.w
    ϵ = O.ϵ
    μ = - O.lr
    ρ = O.inertia

    for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        ∇ = clipfn(setNanInfZero(θ.delta), clipvalue)
        @. w[i] += ρ * w[i] + (1-ρ) * ∇ * ∇
        @. θ.value += μ / (sqrt(w[i])+ϵ) * ∇
    end
end
