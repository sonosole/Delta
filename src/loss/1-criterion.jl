export loss
export cost
export mse
export mseLoss
export mseCost
export crossEntropy
export crossEntropyLoss
export crossEntropyCost
export binaryCrossEntropy
export binaryCrossEntropyLoss
export binaryCrossEntropyCost



"""
    crossEntropy(var::Variable{T}, label::Variable{T}) -> Variable{T}
    cross entropy loss = - l * log(x) where l is label and x is the output of the network.
"""
function crossEntropy(var::Variable{T}, label::Variable{T}) where T
    @assert (var.shape == label.shape)
    backprop = (var.backprop || label.backprop)
    EPS = eltype(var)(1e-38)
    out = Variable{T}(- label.value .* log.(var.value .+ EPS), backprop)
    if backprop
        function crossEntropyBackward()
            if need2computeδ!(var)
                var.delta += - label.value ./ (var.value .+ EPS) .* out.delta;
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, crossEntropyBackward)
    end
    return out
end


"""
    binaryCrossEntropy(x::Variable{T}, l::Variable{T}) -> Variable{T}
    binary cross entropy
    loss = - l * log(x) - (1-l) * log(1 - x)
"""
function binaryCrossEntropy(var::Variable{T}, label::Variable{T}) where T
    @assert (var.shape == label.shape)
    backprop = (var.backprop || label.backprop)
    TOO  = eltype(var)
    EPS  = TOO(1e-38)
    ONE  = TOO(1.000)
    tmp1 = - label.value .* log.(var.value .+ EPS)
    tmp2 = - (ONE .- label.value) .* log.(ONE .- var.value .+ EPS)
    out  = Variable{T}(tmp1 + tmp2, backprop)
    if backprop
        function binaryCrossEntropyBackward()
            if need2computeδ!(var)
                temp1 = (ONE .- label.value) ./ (ONE .- var.value .+ EPS)
                temp2 = label.value ./ (var.value .+ EPS)
                var.delta += out.delta .* (temp1 - temp2)
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, binaryCrossEntropyBackward)
    end
    return out
end


function mse(var::Variable{T}, label::Variable{T}) where T
    @assert (var.shape == label.shape)
    backprop = (var.backprop || label.backprop)
    TWO = eltype(var)(2.0)
    out = Variable{T}((var.value - label.value).^TWO, backprop)
    if backprop
        function mseBackward()
            if need2computeδ!(var)
                var.delta += TWO .* (var.value - label.value) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out)
        end
        push!(graph.backward, mseBackward)
    end
    return out
end


function loss(var::Variable{T}) where T
    out = Variable{T}([sum(var.value)], var.backprop)
    if var.backprop
        function lossBackward()
            if need2computeδ!(var) var.delta .+= out.delta end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, lossBackward)
    end
    return out
end


function cost(var::Variable{T}) where T
    if var.backprop
        ONE = eltype(var)(1.0)
        function costBackward()
            if need2computeδ!(var)
                var.delta .+= ONE
            end
        end
        push!(graph.backward, costBackward)
    end
    return sum(var.value)
end


# -- Loss & Cost --
mseLoss(var::Variable{T}, label::Variable{T}) where T = loss( mse(var, label) )
mseCost(var::Variable{T}, label::Variable{T}) where T = cost( mse(var, label) )
crossEntropyLoss(var::Variable{T}, label::Variable{T}) where T = loss( crossEntropy(var, label) )
crossEntropyCost(var::Variable{T}, label::Variable{T}) where T = cost( crossEntropy(var, label) )
binaryCrossEntropyLoss(var::Variable{T}, label::Variable{T}) where T = loss( binaryCrossEntropy(var, label) )
binaryCrossEntropyCost(var::Variable{T}, label::Variable{T}) where T = cost( binaryCrossEntropy(var, label) )
