mutable struct SGDL1 <: Optimizer
    xparams::Vector{XVariable}
    lr::AbstractFloat
    L1decay::AbstractFloat
    name::String
    function SGDL1(xparams::Vector{XVariable}; lr=1e-4, L1decay=0.001)
        new(xparams, lr, L1decay, "SGDL1")
    end
end

# pretty printing
function Base.show(io::IO, O::SGDL1)
    print("SGDL1(lr=$(O.lr), L1decay=$(O.L1decay))")
end


function update!(O::SGDL1; clipfn::Function=LPInfNormClip, clipvalue=10.0)
    μ = - O.lr
    λ = O.L1decay

    for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        ∇ = clipfn(setNanInfZero(θ.delta), clipvalue)
        if c == 'w'
            @. θ.value += μ * (∇ + λ * sign(θ.value))
        else
            @. θ.value += μ * ∇
        end
    end
end
