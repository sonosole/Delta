include("./1-indrnn.jl")

export resethidden
export PadSeqPackBatch
export PackSeqSlices
export PackedSeqPredict
export PackedSeqForward

global RNNLIST = [indrnn];
# global RNNLIST = [rnn, rin, lstm, indrnn, indlstm, RNN, RIN, LSTM, INDLSTM];



"""
    PadSeqPackBatch(inputs::Vector; epsilon::Real=0.0) -> output
+ `inputs` <: AbstractArray{Real,2}
+ `output` <: AbstractArray{Real,3}
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
    RNNBatch  = zeros(eltype(inputs[1]), featDims, maxSteps, batchSize)
    fill!(RNNBatch, epsilon)

    for i = 1:batchSize
        Tᵢ = lengths[i]
        RNNBatch[:,1:Tᵢ,i] .= inputs[i]
    end
    return RNNBatch
end


"""
    PackSeqSlices(inputs::Vector{Variable{T}}) where T
union output of RNN of different time steps.
# Examples
    x1 = Variable( ones(2,2), keepsgrad=true)
    x2 = Variable(2ones(2,2), keepsgrad=true)
    x3 = Variable(3ones(2,2), keepsgrad=true)
    PackSeqSlices([x1, x2, x3])
"""
function PackSeqSlices(inputs::Vector{Variable{T}}) where T
    timeSteps = length(inputs)
    featDims  = size(inputs[1], 1)
    batchSize = size(inputs[1], 2)
    RNNBatch  = Variable{T}(T(undef, featDims, timeSteps, batchSize), inputs[1].backprop)

    for t = 1:timeSteps
        RNNBatch.value[:,t,:] .= inputs[t].value
    end

    if RNNBatch.backprop
        function PackSeqSlicesBackward()
            for t = 1:timeSteps
                if need2computeδ!(inputs[t])
                    inputs[t].delta .+= RNNBatch.delta[:,t,:]
                end
            end
            ifNotKeepδThenFreeδ!(RNNBatch)
        end
        push!(graph.backward, PackSeqSlicesBackward)
    end
    return RNNBatch
end


function PackSeqSlices(inputs::Vector{T}) where {T <: AbstractArray}
    timeSteps = length(inputs)
    featDims  = size(inputs[1], 1)
    batchSize = size(inputs[1], 2)
    RNNBatch  = zeros(eltype(inputs[1]), featDims, timeSteps, batchSize)
    for t = 1:timeSteps
        RNNBatch[:,t,:] .= inputs[t]
    end
    return RNNBatch
end


function PackedSeqForward(chain::Chain, x::Variable{S}) where S
    T = size(x,2)
    y = Vector{Variable{S}}(undef, T)
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
