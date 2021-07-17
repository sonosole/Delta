export regularize
export L1Regularize
export L2Regularize


"""
    regularize(Ws, lambda::AbstractFloat; method::String="L2")

apply L1 or L2 regularization
"""
function regularize(Ws::Vector; lambda::AbstractFloat, method::String="L2")
    method = uppercase(method)
    method=="L2" && return L2Regularize(Ws, lambda)
    method=="L1" && return L1Regularize(Ws, lambda)
end


"""
    L1Regularize(w::Vector{AbstractArray}, lambda::AbstractFloat)

apply L1 regularization
"""
function L1Regularize(Ws::Vector, λ::AbstractFloat)
    for w in Ws
        @. w += -λ * sign(w)
    end
end

"""
    L2Regularize(w::Vector{AbstractArray}, lambda::AbstractFloat)

apply L2 regularization
"""
function L2Regularize(Ws::Vector, λ::AbstractFloat)
    for w in Ws
        @. w += -2λ * w
    end
end
