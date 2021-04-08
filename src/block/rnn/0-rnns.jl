include("./1-indrnn.jl")

export resethidden
export makeRNNBatch
export unionRNNSteps

global RNNLIST = [indrnn];
# global RNNLIST = [rnn, rin, lstm, indrnn, indlstm, RNN, RIN, LSTM, INDLSTM];



"""
    makeRNNBatch(inputs::Vector)

```julia
julia> makeRNNBatch([ones(2,1), 2ones(2,2), 3ones(2,3)])
2×3×3 Array{Float64,3}:
[:, :, 1] =
 1.0  0.0  0.0
 1.0  0.0  0.0

[:, :, 2] =
 2.0  2.0  0.0
 2.0  2.0  0.0

[:, :, 3] =
 3.0  3.0  3.0
 3.0  3.0  3.0
 ```

"""
function makeRNNBatch(inputs::Vector)
    # all Array of inputs shall have the same size in dim-1
    batchSize = length(inputs)
    lengths   = [size(inputs[i], 2) for i in 1:batchSize]
    featDims  = size(inputs[1], 1)
    maxSteps  = maximum(lengths)
    InPackage = zeros(eltype(inputs[1]), featDims, maxSteps, batchSize)
    for i = 1:batchSize
        Tᵢ = lengths[i]
        InPackage[:,1:Tᵢ,i] = inputs[i]
    end
    return InPackage
end


"""
    unionRNNSteps(inputs::Vector{Variable{T}}) where T
`x1 = Variable( ones(2,2),keepsgrad=true)`\n
`x2 = Variable(2ones(2,2),keepsgrad=true)`\n
`x3 = Variable(3ones(2,2),keepsgrad=true)`\n
`unionRNNSteps([x1, x2, x3])`
"""
function unionRNNSteps(inputs::Vector{Variable{T}}) where T
    timeSteps = length(inputs)
    featDims  = size(inputs[1], 1)
    batchSize = size(inputs[1], 2)
    InPackage = zeros(eltype(inputs[1]), featDims, timeSteps, batchSize)
    for t = 1:timeSteps
        InPackage[:,t,:] = inputs[t].value
    end
    out = Variable{T}(InPackage, inputs[1].backprop)

    if out.backprop
        function unionRNNStepsBackward()
            for t = 1:timeSteps
                if need2computeδ!(inputs[t])
                    inputs[t].delta += out.delta
                end
            end
            ifNotKeepδThenFreeδ!(out)
        end
        push!(graph.backward, unionRNNStepsBackward)
    end
    return out
end
