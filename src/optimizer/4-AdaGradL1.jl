mutable struct AdaGradL1 <: Optimizer
    xparams::Vector{XVariable}
    w::Vector
    lr::AbstractFloat
    ϵ::AbstractFloat
    L1decay::AbstractFloat
    name::String
    function AdaGradL1(xparams::Vector{XVariable}; lr=1e-2, epsilon=1e-10, L1decay=0.001)
        n = length(xparams)
        w = Vector(undef,n)
        for i = 1:n
            c , θ = xparams[i]
            w[i] = Zeros(typeof(θ.value), θ.shape)
        end
        new(xparams, w, lr, epsilon, L1decay, "AdaGradL1")
    end
end


function Base.show(io::IO, O::AdaGradL1)
    print("AdaGradL1(lr=$(O.lr), ϵ=$(O.ϵ), L1decay=$(O.L1decay))")
end


function update!(O::AdaGradL1; clipfn::Function=LPInfNormClip, clipvalue=10.0)
    w = O.w
    μ = - O.lr
    ϵ = O.ϵ
    λ = O.L1decay
    for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        ∇ = clipfn(setNanInfZero(θ.delta), clipvalue)
        @. w[i] += ∇ * ∇
        if c == 'w'
            @. θ.value += μ / (sqrt(w[i]) + ϵ) * (∇ + λ * sign(θ.value))
        else
            @. θ.value += μ / (sqrt(w[i]) + ϵ) * ∇
        end
    end
end
