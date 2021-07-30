mutable struct AdaGrad <: Optimizer
    w::Vector
    lr::AbstractFloat
    ϵ::AbstractFloat
    name::String
    function AdaGrad(params::Vector{Variable}; lr=1e-2, epsilon=1e-10)
        n = length(params)
        w = Vector(undef,n)
        for i = 1:n
           w[i] = Zeros(typeof(params[i].value), params[i].shape)
        end
        new(w, lr, epsilon, "AdaGrad")
    end
end


function Base.show(io::IO, A::AdaGrad)
    print("AdaGrad(lr=$(A.lr), ϵ=$(A.ϵ))")
end


function update!(m::AdaGrad, params::Vector{Variable}; clipfn::Function=LPInfNormClip, clipvalue=1e1)
    w  = m.w
    lr = m.lr
    ϵ  = m.ϵ
    for i = 1:length(params)
        ∇ = clipfn(params[i].delta, clipvalue)
        @. w[i] += ∇ * ∇
        @. params[i].value += (-lr) / (sqrt(w[i]) + ϵ) * ∇
    end
end
