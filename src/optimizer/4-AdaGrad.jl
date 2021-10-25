mutable struct AdaGrad <: Optimizer
    xparams::Vector{XVariable}
    w::Vector
    lr::AbstractFloat
    ϵ::AbstractFloat
    name::String
    function AdaGrad(xparams::Vector{XVariable}; lr=1e-2, epsilon=1e-10, L1decay=0.001)
        n = length(xparams)
        w = Vector(undef,n)
        for i = 1:n
            c , θ = xparams[i]
            w[i] = Zeros(typeof(θ.value), θ.shape)
        end
        new(xparams, w, lr, epsilon, "AdaGrad")
    end
end


function Base.show(io::IO, O::AdaGrad)
    print("AdaGrad(lr=$(O.lr), ϵ=$(O.ϵ))")
end


function update!(O::AdaGrad; clipfn::Function=LPInfNormClip, clipvalue=10.0)
    w = O.w
    μ = - O.lr
    ϵ = O.ϵ
    for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        ∇ = clipfn(setNanInfZero(θ.delta), clipvalue)
        @. w[i] += ∇ * ∇
        @. θ.value += μ / (sqrt(w[i]) + ϵ) * ∇
    end
end
