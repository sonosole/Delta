export regularize
export L1Regularize
export L2Regularize
export sparsityof
export sparsitypair

"""
    regularize(Ws::Vector; lambda::AbstractFloat, method::String="L2")

apply L1 or L2 regularization to unbiased weights, recurrent weights not included
"""
function regularize(Ws::Vector; lambda::AbstractFloat, method::String="L2")
    method = uppercase(method)
    method=="L2" && return L2Regularize(Ws, lambda)
    method=="L1" && return L1Regularize(Ws, lambda)
end


"""
    L1Regularize(Ws::Vector, λ::AbstractFloat)

apply L1 regularization to unbiased weights, recurrent weights not included
"""
function L1Regularize(Ws::Vector, λ::AbstractFloat)
    Threads.@threads for w in Ws
        @. w += -λ * sign(w)
    end
end

"""
    L2Regularize(Ws::Vector, λ::AbstractFloat)

apply L2 regularizationto unbiased weights, recurrent weights not included
"""
function L2Regularize(Ws::Vector, λ::AbstractFloat)
    Threads.@threads for w in Ws
        @. w += -λ * w
    end
end


"""
    sparsityof(x, a::AbstractFloat) -> sparserRatio

calculate the sparsity of Array `x` at given threshold `a > 0`
"""
function sparsityof(x, a::AbstractFloat)
    @assert a>0.0 "the threshold should > 0, but got $a"
    return sum(abs.(x) .< a) / length(x)
end


"""
    sparsitypair(x, a::AbstractFloat) -> (sparsity::Int, length(x)::Int)

calculate the #sparsity of Array `x` at given threshold `a > 0` and return (sparsity, length(x))
"""
function sparsitypair(x, a::AbstractFloat)
    @assert a>0.0 "the threshold should > 0, but got $a"
    return sum(abs.(x) .< a), length(x)
end


"""
    sparsityof(xs::Vector, a::AbstractFloat) -> sparserRatio

calculate the sparsity of a Vector of Arrays `xs` at given threshold `a > 0`
"""
function sparsityof(xs::Vector, a::AbstractFloat)
    @assert a>0.0 "the threshold should > 0, but got $a"
    S = 0
    T = 0
    for x in xs
        s, t = sparsitypair(x, a)
        S += s
        T += t
    end
    return S/T
end
