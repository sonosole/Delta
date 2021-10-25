mutable struct Momentum <: Optimizer
    xparams::Vector{XVariable}
    v::Vector
    lr::AbstractFloat
    inertia::AbstractFloat
    name::String
    function Momentum(xparams::Vector{XVariable}; lr=1e-4, inertia=0.9)
        num = length(xparams)
        vel = Vector(undef,num)
        for i = 1:num
            c , θ = xparams[i]
            vel[i] = Zeros(typeof(θ.value), θ.shape)
        end
        new(xparams, vel, lr, inertia, "Momentum")
    end
end


function Base.show(io::IO, O::Momentum)
    print("Momentum(lr=$(O.lr), inertia=$(O.inertia))")
end


function update!(O::Momentum; clipfn::Function=LPInfNormClip, clipvalue=10.0)
    vel = O.v
    μ = - O.lr
    ρ = O.inertia

    for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        ∇ = clipfn(setNanInfZero(θ.delta), clipvalue)
        @. vel[i] = ρ * vel[i] + ∇
        @. θ.value += μ * vel[i]
    end
end
