mutable struct SGDL1L2 <: Optimizer
    xparams::Vector{XVariable}
    lr::AbstractFloat
    L1decay::AbstractFloat
    L2decay::AbstractFloat
    name::String
    function SGDL1L2(xparams::Vector{XVariable}; lr=1e-4, L1decay=0.001, L2decay=0.01)
        new(xparams, lr, L1decay, L2decay, "SGDL1L2")
    end
end

# pretty printing
function Base.show(io::IO, O::SGDL1L2)
    print("SGDL1L2(lr=$(O.lr), L1decay=$(O.L1decay), L2decay=$(O.L2decay))")
end


function update!(O::SGDL1L2; clipfn::Function=LPInfNormClip, clipvalue=10.0)
    μ = - O.lr
    λ₁ = O.L1decay
    λ₂ = O.L2decay

    for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        ∇ = clipfn(setNanInfZero(θ.delta), clipvalue)
        if c == 'w'
            @. θ.value += μ * (∇ + λ₁ * sign(θ.value) + λ₂ * θ.value)
        else
            @. θ.value += μ * ∇
        end
    end
end
