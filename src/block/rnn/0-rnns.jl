include("./1-indrnn.jl")
include("./2-indlstm.jl")
include("./3-rnn.jl")

export RNN
export RNNs
export indrnn
export INDRNN
export indlstm
export INDLSTM

export resethidden
export PadSeqPackBatch
export PackedSeqPredict
export PackedSeqForward

global RNNLIST = [indrnn,INDRNN,
                  indlstm,INDLSTM,
                  RNN,RNNs];
# global RNNLIST = [RNN, rin, lstm, indrnn, indlstm, RNNs, RIN, LSTM, INDLSTM];



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


function PackedSeqForward(chain::Chain, x::Variable{S}) where S
    T = size(x, 2)
    y = Vector{Variable{S}}(undef, T)
    resethidden(chain)
    for t = 1:T
        y[t] = forward(chain, x[:,t,:])
    end

    timeSteps = T
    featsDims = size(y[1], 1)
    batchSize = size(y[1], 2)
    RNNBatch  = Variable{S}(S(undef, featsDims, timeSteps, batchSize), x.backprop)

    for t = 1:timeSteps
        RNNBatch.value[:,t,:] .= y[t].value
    end

    if RNNBatch.backprop
        function PackSeqSlicesBackward()
            for t = 1:timeSteps
                if need2computeδ!(y[t])
                    y[t].delta .+= RNNBatch.delta[:,t,:]
                end
            end
            ifNotKeepδThenFreeδ!(RNNBatch)
        end
        push!(graph.backward, PackSeqSlicesBackward)
    end
    return RNNBatch
end


function PackedSeqPredict(chain::Chain, x::AbstractArray{S}) where S
    T = size(x,2)
    y = Vector{AbstractArray{S}}(undef,T)
    resethidden(chain)
    for t = 1:T
        y[t] = predict(chain, x[:,t,:])
    end

    timeSteps = T
    featsDims = size(y[1], 1)
    batchSize = size(y[1], 2)
    RNNBatch  = typeof(x)(undef, featsDims, timeSteps, batchSize)
    for t = 1:timeSteps
        RNNBatch[:,t,:] .= y[t]
    end
    return RNNBatch
end
