export loss
export cost


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


include("./1-criterion-regression.jl")
include("./1-criterion-probabilistic.jl")
