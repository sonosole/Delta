export dropout
export dropout!


"""
    dropout(x::Variable{T}; p=0.1) -> y::Variable{T}

Randomly zeroes some elements of the input tensor `x` with probability `p` using samples
from a Bernoulli distribution. This is an effective way to regularization and preventing
the co-adaptation of neurons. The output elements of `y` are scaled by a factor of `1/(1-p)`
during training. During evaluation, `dropout` should be removed. `dropout` is also viewed as
a mean of data augmentation.
"""
function dropout(x::Variable{T}; p=0.1) where T
    @assert 0.0<=p<=1.0 "p is in [0,1), but got p=$p"
    τ = eltype(T)
    l = τ(1)
    p = τ(p)
    m = T(rand(τ, x.shape) .< (l - p)) .* (l/(l - p)) # weighted mask
    y = Variable{T}(ᵛ(x) .* m, x.backprop)
    if x.backprop
        function dropoutBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* m
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, dropoutBackward)
    end
    return y
end



"""
    dropout!(x::Variable{T}; p=0.1) -> y::Variable{T}

Randomly zeroes some elements of the input tensor `x` with probability `p` using samples
from a Bernoulli distribution. This is an effective way to regularization and preventing
the co-adaptation of neurons. The output elements of `y` are scaled by a factor of `1/(1-p)`
during training. During evaluation, `dropout!` should be removed. `dropout` is also viewed as
a mean of data augmentation.
"""
function dropout!(x::Variable{T}; p=0.1) where T
    @assert 0.0<=p<=1.0 "p is in [0,1), but got p=$p"
    τ = eltype(T)
    l = τ(1)
    p = τ(p)
    m = T(rand(τ, x.shape) .< (l - p)) .* (l/(l - p)) # weighted mask
    y = Variable{T}(dotmul!(ᵛ(x), m), x.backprop)
    if x.backprop
        function dropoutBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* m
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, dropoutBackward)
    end
    return y
end
