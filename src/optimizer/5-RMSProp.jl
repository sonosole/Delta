mutable struct RMSProp <: Optimizer
    w::Vector
    lr::AbstractFloat
    ϵ::AbstractFloat
    inertia::AbstractFloat
    name::String
    function RMSProp(params::Vector{Variable}; lr=1e-2, inertia=0.99, epsilon=1e-8)
        n = length(params)
        w = Vector(undef,n)
        for i = 1:n
           w[i] = Zeros(typeof(params[i].value), params[i].shape)
        end
        new(w, lr, epsilon, inertia, "RMSProp")
    end
end


function Base.show(io::IO, R::RMSProp)
    print("RMSProp(lr=$(R.lr), ϵ=$(R.ϵ), inertia=$(R.inertia))")
end


function update!(m::RMSProp, params::Vector{Variable}; clipfn::Function=LPInfNormClip, clipvalue=10.0)
    w  = m.w
    ϵ  = m.ϵ
    lr = m.lr
    ρ  = m.inertia
    for i = 1:length(params)
        ∇ = clipfn(setNanInfZero(params[i].delta), clipvalue)
        @. w[i] += ρ * w[i] + (1-ρ) * ∇ * ∇
        @. params[i].value += (-lr) / (sqrt(w[i])+ϵ) * ∇
    end
end
