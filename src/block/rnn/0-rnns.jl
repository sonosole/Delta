include("./1-indrnn.jl")

export resethidden
export PadSeqPackBatch
export PackSeqSlices
export PackedSeqPredict
export PackedSeqForward

global RNNLIST = [indrnn];
# global RNNLIST = [rnn, rin, lstm, indrnn, indlstm, RNN, RIN, LSTM, INDLSTM];



"""
    PadSeqPackBatch(inputs::Vector{T}; epsilon::Real=0.0) where {T<: AbstractArray} -> AbstractArray{Real,3}
pad epsilon to align raw input features probably with different length
# Examples
    julia> PadSeqPackBatch([ones(2,1), 2ones(2,2), 3ones(2,3)])
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
"""
function PadSeqPackBatch(inputs::Vector; epsilon::Real=0.0)
    # all Array of inputs shall have the same size in dim-1
    batchSize = length(inputs)
    lengths   = [size(inputs[i], 2) for i in 1:batchSize]
    featDims  = size(inputs[1], 1)
    maxSteps  = maximum(lengths)
    rnnBatch  = zeros(eltype(inputs[1]), featDims, maxSteps, batchSize)
    fill!(rnnBatch, epsilon)

    for i = 1:batchSize
        Tᵢ = lengths[i]
        rnnBatch[:,1:Tᵢ,i] .= inputs[i]
    end
    return rnnBatch
end


"""
    PackSeqSlices(inputs::Vector{Variable})
union output of RNN of different time steps.
# Examples
    x1 = Variable( ones(2,2), keepsgrad=true)
    x2 = Variable(2ones(2,2), keepsgrad=true)
    x3 = Variable(3ones(2,2), keepsgrad=true)
    PackSeqSlices([x1, x2, x3])
"""
function PackSeqSlices(inputs::Vector{Variable})
    timeSteps = length(inputs)
    featDims  = size(inputs[1], 1)
    batchSize = size(inputs[1], 2)
    rnnBatch  = zeros(eltype(inputs[1]), featDims, timeSteps, batchSize)
    for t = 1:timeSteps
        rnnBatch[:,t,:] .= inputs[t].value
    end
    out = typeof(inputs[1])(rnnBatch, inputs[1].backprop)

    if out.backprop
        function PackSeqSlicesBackward()
            for t = 1:timeSteps
                if need2computeδ!(inputs[t])
                    inputs[t].delta .+= out.delta[:,t,:]
                end
            end
            ifNotKeepδThenFreeδ!(out)
        end
        push!(graph.backward, PackSeqSlicesBackward)
    end
    return out
end


function PackSeqSlices(inputs::Vector{T}) where {T <: AbstractArray}
    timeSteps = length(inputs)
    featDims  = size(inputs[1], 1)
    batchSize = size(inputs[1], 2)
    rnnBatch  = zeros(eltype(inputs[1]), featDims, timeSteps, batchSize)
    for t = 1:timeSteps
        rnnBatch[:,t,:] .= inputs[t]
    end
    return rnnBatch
end


function PackedSeqForward(chain::Chain, x::Variable)
    T = size(x,2)
    y = Vector{Variable}(undef,T)
    resethidden(chain)
    for t = 1:T
        y[t] = forward(chain, x[:,t,:])
    end
    return PackSeqSlices(y)
end


function PackedSeqPredict(chain::Chain, x::AbstractArray{S,N}) where {S,N}
    T = size(x,2)
    y = Vector{AbstractArray{S,N-1}}(undef,T)
    resethidden(chain)
    for t = 1:T
        y[t] = predict(chain, x[:,t,:])
    end
    return PackSeqSlices(y)
end
