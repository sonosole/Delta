include("./1-indrnn.jl")

export resethidden
export makeRNNBatch
export unionRNNSteps

global RNNLIST = [indrnn];
# global RNNLIST = [rnn, rin, lstm, indrnn, indlstm, RNN, RIN, LSTM, INDLSTM];



"""
    makeRNNBatch(inputs::Vector)
pad zeros to align raw input features probably with different length
# Examples
```jldoctest
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
    rnnBatch  = zeros(eltype(inputs[1]), featDims, maxSteps, batchSize)
    for i = 1:batchSize
        Tᵢ = lengths[i]
        rnnBatch[:,1:Tᵢ,i] = inputs[i]
    end
    return rnnBatch
end


"""
    unionRNNSteps(inputs::Vector{Variable})
union output of RNN of different time steps.
# Examples
`x1 = Variable( ones(2,2),keepsgrad=true)`\n
`x2 = Variable(2ones(2,2),keepsgrad=true)`\n
`x3 = Variable(3ones(2,2),keepsgrad=true)`\n
`unionRNNSteps([x1, x2, x3])`
"""
function unionRNNSteps(inputs::Vector{Variable})
    timeSteps = length(inputs)
    featDims  = size(inputs[1], 1)
    batchSize = size(inputs[1], 2)
    rnnBatch  = zeros(eltype(inputs[1]), featDims, timeSteps, batchSize)
    for t = 1:timeSteps
        rnnBatch[:,t,:] = inputs[t].value
    end
    out = typeof(inputs[1])(rnnBatch, inputs[1].backprop)

    if out.backprop
        function unionRNNStepsBackward()
            for t = 1:timeSteps
                if need2computeδ!(inputs[t])
                    inputs[t].delta += out.delta[:,t,:]
                end
            end
            ifNotKeepδThenFreeδ!(out)
        end
        push!(graph.backward, unionRNNStepsBackward)
    end
    return out
end


function unionRNNSteps(inputs::Vector{T}) where {T <: AbstractArray}
    timeSteps = length(inputs)
    featDims  = size(inputs[1], 1)
    batchSize = size(inputs[1], 2)
    rnnBatch  = zeros(eltype(inputs[1]), featDims, timeSteps, batchSize)
    for t = 1:timeSteps
        rnnBatch[:,t,:] = inputs[t]
    end
    return rnnBatch
end
