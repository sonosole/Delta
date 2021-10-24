mutable struct MomentumL1 <: Optimizer
    xparams::Vector{XVariable}
    v::Vector
    lr::AbstractFloat
    inertia::AbstractFloat
    L1decay::AbstractFloat
    name::String
    function MomentumL1(xparams::Vector{XVariable}; lr=1e-4, inertia=0.9, L1decay=0.001)
        num = length(xparams)
        vel = Vector(undef,num)
        for i = 1:num
            c , θ = xparams[i]
            vel[i] = Zeros(typeof(θ.value), θ.shape)
        end
        new(xparams, vel, lr, inertia, L1decay, "MomentumL1")
    end
end


function Base.show(io::IO, O::MomentumL1)
    print("MomentumL1(lr=$(O.lr), inertia=$(O.p), L1decay=$(O.L1decay))")
end


function update!(O::MomentumL1; clipfn::Function=LPInfNormClip, clipvalue=10.0)
    vel = O.v
    μ = - O.lr
    ρ = O.inertia
    λ = O.L1decay

    for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        ∇ = clipfn(setNanInfZero(θ.delta), clipvalue)
        @. vel[i] = ρ * vel[i] + ∇
        if c == 'w'
            @. θ.value += μ * (vel[i] + λ * sign(θ.value))
        else
            @. θ.value += μ * vel[i]
        end
    end
end
