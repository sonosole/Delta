export loss
export cost


# Internal function
function _sum(x::Variable{T}) where T
    y = Variable{T}([sum(ᵛ(x))], x.backprop)
    if x.backprop
        function _sumBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        push!(graph.backward, _sumBackward)
    end
    return y
end


# Internal function
function _mean(x::Variable{T}) where T
    n = eltype(x)(1) / prod(size(x))
    μ = Variable{T}([sum(ᵛ(x)) * n], x.backprop)
    if x.backprop
        function _meanBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(μ) .* n
            end
            ifNotKeepδThenFreeδ!(μ);
        end
        push!(graph.backward, _meanBackward)
    end
    return μ
end


"""
    loss(x::Variable{T}; reduction::String="sum") -> y::Variable{T}

Sums or takes mean over all elements in `value` of `x` as the loss `Variable`, i.e. ⤦\n
+ `y = Variable{T}([sum(ᵛ(x))], x.backprop)`, if reduction="sum"
+ `y = Variable{T}([sum(ᵛ(x))/length(x)], x.backprop)`, if reduction="mean"
This is very convenient for mutiple loss training. e.g. ⤦\n
    totalLoss = β*loss(mse(x₁, ̂x₁)) + (1 - β)*loss(crossEntropy(y₁, ̂y₁))
or in a lazy way:\n
    totalLoss = β*mseLoss(x₁, ̂x₁) + (1 - β)*crossEntropyLoss(y₁, ̂y₁)
where `β` is weight of mseLoss function.
"""
function loss(x::Variable{T}; reduction::String="sum") where T
    by = lowercase(reduction)
    by=="sum" && return _sum(x)
    by=="mean" && return _mean(x)
    @error "wrong reduction method" reduction=="sum" || reduction=="mean"
end


"""
    cost(x::Variable) -> y::eltype(x)

sums over all elements in `value` of `x` as the final loss value which is a scalar, i.e. ⤦\n
    y = sum(x.value), so δ(x) = ∂y/∂x = 𝟙
"""
function cost(x::Variable{T}) where T
    if x.backprop
        𝟙 = eltype(x)(1.0)
        function costBackward()
            if need2computeδ!(x)
                δ(x) .+= 𝟙
            end
        end
        push!(graph.backward, costBackward)
    end
    return sum(ᵛ(x))
end


include("./1-criterion-regression.jl")
include("./1-criterion-probabilistic.jl")
