## regression loss

export mae
export maeLoss, maeCost
export L1Loss, L1Cost

export mse
export mseLoss, mseCost
export L2Loss, L2Cost

export LpLoss, LpCost


"""
    mae(x::Variable{T}, label::Variable{T}) -> y::Variable{T}

mean absolute error (mae) between each element in the input `x` and target `label`. Also called L1Loss. i.e. ⤦\n
    y = |xᵢ - lᵢ|
"""
function mae(x::Variable{T}, label::Variable{T}) where T
    @assert (x.shape == label.shape)
    backprop = (x.backprop || label.backprop)
    y = Variable{T}(abs.(ᵛ(x) - ᵛ(label)), backprop)
    if backprop
        function maeBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* sign.(ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        push!(graph.backward, maeBackward)
    end
    return y
end

maeLoss(x::Variable{T}, label::Variable{T}) where T = loss( mae(x, label) )
maeCost(x::Variable{T}, label::Variable{T}) where T = cost( mae(x, label) )
L1Loss(x::Variable{T},  label::Variable{T}) where T = loss( mae(x, label) )
L1Cost(x::Variable{T},  label::Variable{T}) where T = cost( mae(x, label) )


"""
    mse(x::Variable{T}, label::Variable{T}) -> y::Variable{T}

mean sqrt error (mse) between each element in the input `x` and target `label`. Also called L2Loss. i.e. ⤦\n
    y = (xᵢ - lᵢ)²
"""
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

mseLoss(x::Variable{T}, label::Variable{T}) where T = loss( mse(x, label) )
mseCost(x::Variable{T}, label::Variable{T}) where T = cost( mse(x, label) )
L2Loss(x::Variable{T},  label::Variable{T}) where T = loss( mse(x, label) )
L2Cost(x::Variable{T},  label::Variable{T}) where T = cost( mse(x, label) )


"""
    Lp(x::Variable{T}, label::Variable{T}; p=3) -> y::Variable{T}

absolute error's p-th power between each element in the input `x` and target `label`. Also called LpLoss. i.e. ⤦\n
    y = |xᵢ - lᵢ|ᵖ
"""
function Lp(x::Variable{T}, label::Variable{T}; p=3) where T
    @assert (x.shape == label.shape)
    backprop = (x.backprop || label.backprop)
    Δ = ᵛ(x) - ᵛ(label)
    y = Variable{T}(Δ .^ p, backprop)
    if backprop
        function LpBackward()
            if need2computeδ!(x)
                i = (Δ .!= eltype(T)(0.0))
                x.delta[i] .+= y.delta[i] .* y.value[i] ./ Δ[i] .* p
                # δ(x) .+= δ(y) .* ᵛ(y) ./ Δ .* p
            end
            ifNotKeepδThenFreeδ!(y)
        end
        push!(graph.backward, LpBackward)
    end
    return y
end

LpLoss(x::Variable{T}, label::Variable{T}; p=3) where T = loss( Lpnorm(x, label; p=p) )
LpCost(x::Variable{T}, label::Variable{T}; p=3) where T = cost( Lpnorm(x, label; p=p) )
