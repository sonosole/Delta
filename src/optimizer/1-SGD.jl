mutable struct SGD <: Optimizer
    xparams::Vector{XVariable}
    lr::AbstractFloat
    L1decay::AbstractFloat
    name::String
    function SGD(xparams::Vector{XVariable}; lr=1e-4)
        new(xparams, lr, "SGD")
    end
end

# pretty printing
function Base.show(io::IO, O::SGD)
    print("SGD(lr=$(O.lr))")
end


function update!(O::SGD; clipfn::Function=LPInfNormClip, clipvalue=10.0)
    μ = - O.lr
    for i = 1:length(O.xparams)
        c , θ = O.xparams[i]
        ∇ = clipfn(setNanInfZero(θ.delta), clipvalue)
        @. θ.value += μ * ∇
    end
end
