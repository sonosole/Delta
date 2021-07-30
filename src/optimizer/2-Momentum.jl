mutable struct Momentum <: Optimizer
    v::Vector
    lr::AbstractFloat
    inertia::AbstractFloat
    decay::AbstractFloat
    name::String
    function Momentum(params::Vector{Variable}; lr=1e-4, inertia=0.9, decay=1.0)
        num = length(params)
        vel = Vector(undef,num)
        for i = 1:num
           vel[i] = Zeros(typeof(params[i].value), params[i].shape)
        end
        new(vel, lr, inertia, decay, "Momentum")
    end
end


function Base.show(io::IO, M::Momentum)
    print("Momentum(lr=$(M.lr), inertia=$(M.p), decay=$(M.decay))")
end


function update!(m::Momentum, params::Vector{Variable}; clipfn::Function=LPInfNormClip, clipvalue=1e1)
    vel = m.v
    lr  = m.lr
    ρ   = m.inertia
    m.lr *= m.decay
    for i = 1:length(params)
        ∇ = clipfn(params[i].delta, clipvalue)
        @. vel[i] = ρ * vel[i] + ∇
        @. params[i].value += (-lr) * vel[i]
    end
end
