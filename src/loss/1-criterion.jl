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
    crossEntropy(x::Variable{T}, label::Variable{T}) -> Variable{T}
cross entropy = - y * log(̂y) where y is target and ̂y is the output of the network.
"""
function crossEntropy(x::Variable{T}, label::Variable{T}) where T
    @assert (x.shape == label.shape)
    backprop = (x.backprop || label.backprop)
    ϵ = eltype(x)(1e-38)
    y = Variable{T}(- ᵛ(label) .* log.(ᵛ(x) .+ ϵ), backprop)
    if backprop
        function crossEntropyBackward()
            if need2computeδ!(x)
                δ(x) .-= δ(y) .* ᵛ(label) ./ (ᵛ(x) .+ ϵ)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, crossEntropyBackward)
    end
    return y
end


"""
    binaryCrossEntropy(x::Variable{T}, l::Variable{T}) -> Variable{T}
binary cross entropy = - y * log(̂y) - (1 - y) * log(1-̂y)
"""
function binaryCrossEntropy(x::Variable{T}, label::Variable{T}) where T
    @assert (x.shape == label.shape)
    backprop = (x.backprop || label.backprop)
    TOO  = eltype(x)
    ϵ  = TOO(1e-38)
    𝟙  = TOO(1.0f0)
    tmp1 = - ᵛ(label) .* log.(ᵛ(x) .+ ϵ)
    tmp2 = - (𝟙 .- ᵛ(label)) .* log.(𝟙 .- ᵛ(x) .+ ϵ)
    y  = Variable{T}(tmp1 + tmp2, backprop)
    if backprop
        function binaryCrossEntropyBackward()
            if need2computeδ!(x)
                temp1 = (𝟙 .- ᵛ(label)) ./ (𝟙 .- ᵛ(x) .+ ϵ)
                temp2 = ᵛ(label) ./ (ᵛ(x) .+ ϵ)
                δ(x) .+= δ(y) .* (temp1 - temp2)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, binaryCrossEntropyBackward)
    end
    return y
end


function mse(x::Variable{T}, label::Variable{T}) where T
    @assert (x.shape == label.shape)
    backprop = (x.backprop || label.backprop)
    𝟚 = eltype(x)(2.0f0)
    y = Variable{T}((ᵛ(x) - ᵛ(label)).^𝟚, backprop)
    if backprop
        function mseBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* 𝟚 .* (ᵛ(x) - ᵛ(label))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        push!(graph.backward, mseBackward)
    end
    return y
end


function loss(x::Variable{T}) where T
    y = Variable{T}([sum(ᵛ(x))], x.backprop)
    if x.backprop
        function lossBackward()
            if need2computeδ!(x) δ(x) .+= δ(y) end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, lossBackward)
    end
    return y
end


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


# -- Loss & Cost --
mseLoss(x::Variable{T}, label::Variable{T}) where T = loss( mse(x, label) )
mseCost(x::Variable{T}, label::Variable{T}) where T = cost( mse(x, label) )
crossEntropyLoss(x::Variable{T}, label::Variable{T}) where T = loss( crossEntropy(x, label) )
crossEntropyCost(x::Variable{T}, label::Variable{T}) where T = cost( crossEntropy(x, label) )
binaryCrossEntropyLoss(x::Variable{T}, label::Variable{T}) where T = loss( binaryCrossEntropy(x, label) )
binaryCrossEntropyCost(x::Variable{T}, label::Variable{T}) where T = cost( binaryCrossEntropy(x, label) )
