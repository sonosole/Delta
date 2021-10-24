mutable struct SGDL2 <: Optimizer
    xparams::Vector{XVariable}
    lr::AbstractFloat
    L2decay::AbstractFloat
    name::String
    function SGDL2(xparams::Vector{XVariable}; lr=1e-4, L2decay=0.01)
        new(xparams, lr, L2decay, "SGDL2")
    end
end

# pretty printing
function Base.show(io::IO, O::SGDL2)
    print("SGDL2(lr=$(O.lr), L2decay=$(O.L2decay))")
end


function update!(O::SGDL2; clipfn::Function=LPInfNormClip, clipvalue=10.0)
    μ = - O.lr
    λ = O.L2decay

    for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        ∇ = clipfn(setNanInfZero(θ.delta), clipvalue)
        if c == 'w'
            @. θ.value += μ * (∇ + λ * θ.value)
        else
            @. θ.value += μ * ∇
        end
    end
end
