mutable struct MomentumL2 <: Optimizer
    xparams::Vector{XVariable}
    v::Vector
    lr::AbstractFloat
    inertia::AbstractFloat
    L2decay::AbstractFloat
    name::String
    function MomentumL2(xparams::Vector{XVariable}; lr=1e-4, inertia=0.9, L2decay=0.01)
        num = length(xparams)
        vel = Vector(undef,num)
        for i = 1:num
            c , θ = xparams[i]
            vel[i] = Zeros(typeof(θ.value), θ.shape)
        end
        new(xparams, vel, lr, inertia, L2decay, "MomentumL2")
    end
end


function Base.show(io::IO, O::MomentumL2)
    print("MomentumL2(lr=$(O.lr), inertia=$(O.inertia), L2decay=$(O.L2decay))")
end


function update!(O::MomentumL2; clipfn::Function=LPInfNormClip, clipvalue=10.0)
    vel = O.v
    μ = - O.lr
    ρ = O.inertia
    λ = O.L2decay

    for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        ∇ = clipfn(setNanInfZero(θ.delta), clipvalue)
        @. vel[i] = ρ * vel[i] + ∇
        if c == 'w'
            @. θ.value += μ * (vel[i] + λ * θ.value)
        else
            @. θ.value += μ * vel[i]
        end
    end
end
