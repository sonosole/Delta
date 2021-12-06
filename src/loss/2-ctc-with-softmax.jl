export DNN_CTC_With_Softmax
export DNN_Batch_CTC_With_Softmax
export RNN_Batch_CTC_With_Softmax
export CRNN_Batch_CTC_With_Softmax


"""
    DNN_CTC_With_Softmax(x::Variable{Array{T}}, seq; blank=1, weight=1.0)

case batchsize==1 for test case. `x` is the output of a whole complete input sequence

# Inputs
`x`      : 2-D Variable, input sequence\n
`seq`    : 1-D Array, input sequence's label\n
`weight` : weight for CTC loss

# Structure
    ┌───┐
    │ │ │
    │ W ├──►─┐
    │ │ │    │
    └───┘    │
    ┌───┐    │    ┌───┐          ┌───┐
    │ │ │  ┌─┴─┐  │ │ │ softmax  │ │ │   ┌───────┐
    │ Z ├─►│ × ├─►│ X ├─────────►│ P ├──►│CTCLOSS│◄── (seqLabel)
    │ │ │  └───┘  │ │ │          │ │ │   └───┬───┘
    └───┘         └─┬─┘          └─┬─┘       │
                    │              │+        ▼
                  ┌─┴─┐            ▼       ┌─┴─┐
                  │ │ │          ┌─┴─┐ -   │ │ │
                  │ δ │◄─────────┤ - │──◄──┤ r │
                  │ │ │          └───┘     │ │ │
                  └───┘                    └───┘
"""
function DNN_CTC_With_Softmax(x::Variable{Array{T}}, seq; blank=1, weight=1.0) where T
    p = softmax(ᵛ(x); dims=1)
    L = length(seq) * 2 + 1
    r, loglikely = CTC(p, seq, blank=blank)
    if x.backprop
        function DNN_CTC_With_Softmax_Backward()
            if need2computeδ!(x)
                if weight==1.0
                    δ(x) .+=  p - r
                else
                    δ(x) .+= (p - r) .* weight
                end
            end
        end
        push!(graph.backward, DNN_CTC_With_Softmax_Backward)
    end
    return loglikely / L
end


"""
    DNN_Batch_CTC_With_Softmax(x::Variable{Array{T}}, seqlabels::Vector, inputlens; blank=1, weight=1.0) where T

a batch of concatenated input sequence is processed by neural networks into `x`

# Inputs
`x`         : 2-D Variable, inputs of softmax\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`inputlens` : records each input sequence's length, like [20,17,...]\n
`weight`    : weight for CTC loss

# Structure
    ┌───┐
    │ │ │
    │ W ├──►─┐
    │ │ │    │
    └───┘    │
    ┌───┐    │    ┌───┐          ┌───┐
    │ │ │  ┌─┴─┐  │ │ │ softmax  │ │ │   ┌───────┐
    │ Z ├─►│ × ├─►│ X ├─────────►│ P ├──►│CTCLOSS│◄── (seqLabel)
    │ │ │  └───┘  │ │ │          │ │ │   └───┬───┘
    └───┘         └─┬─┘          └─┬─┘       │
                    │              │+        ▼
                  ┌─┴─┐            ▼       ┌─┴─┐
                  │ │ │          ┌─┴─┐ -   │ │ │
                  │ δ │◄─────────┤ - │──◄──┤ r │
                  │ │ │          └───┘     │ │ │
                  └───┘                    └───┘
"""
function DNN_Batch_CTC_With_Softmax(x::Variable{Array{T}}, seqlabels::Vector, inputlens; blank=1, weight=1.0) where T
    batchsize = length(inputLengths)
    loglikely = zeros(T, batchsize)
    I, F = indexbounds(inputlens)
    p = softmax(ᵛ(x); dims=1)
    r = zero(p)

    Threads.@threads for b = 1:batchsize
        span = I[b]:F[b]
        r[:,span], loglikely[b] = CTC(p[:,span], seqlabels[b], blank=blank)
        loglikely[b] /= length(seqlabels[b]) * 2 + 1
    end

    if x.backprop
        function DNN_Batch_CTC_With_Softmax_Backward()
            if need2computeδ!(x)
                if weight==1.0
                    δ(x) .+=  p - r
                else
                    δ(x) .+= (p - r) .* weight
                end
            end
        end
        push!(graph.backward, DNN_Batch_CTC_With_Softmax_Backward)
    end
    return sum(loglikely)/batchsize
end


"""
    RNN_Batch_CTC_With_Softmax(x::Variable{Array{T}}, seqlabels::Vector, inputlens; blank=1, weight=1.0) where T

a batch of padded input sequence is processed by neural networks into `x`

# Inputs
`x`         : 3-D Variable with shape (featdims,timesteps,batchsize), inputs of softmax\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`inputlens` : each input's length, like [19,97,...]\n
`weight`    : weight for CTC loss

# Structure
    ┌───┐
    │ │ │
    │ W ├──►─┐
    │ │ │    │
    └───┘    │
    ┌───┐    │    ┌───┐          ┌───┐
    │ │ │  ┌─┴─┐  │ │ │ softmax  │ │ │   ┌───────┐
    │ Z ├─►│ × ├─►│ X ├─────────►│ P ├──►│CTCLOSS│◄── (seqLabel)
    │ │ │  └───┘  │ │ │          │ │ │   └───┬───┘
    └───┘         └─┬─┘          └─┬─┘       │
                    │              │+        ▼
                  ┌─┴─┐            ▼       ┌─┴─┐
                  │ │ │          ┌─┴─┐ -   │ │ │
                  │ δ │◄─────────┤ - │──◄──┤ r │
                  │ │ │          └───┘     │ │ │
                  └───┘                    └───┘
"""
function RNN_Batch_CTC_With_Softmax(x::Variable{Array{T}}, seqlabels::Vector, inputlens; blank=1, weight=1.0) where T
    batchsize = length(inputlens)
    loglikely = zeros(T, batchsize)
    p = zero(ᵛ(x))
    r = zero(ᵛ(x))

    Threads.@threads for b = 1:batchsize
        Tᵇ = inputlens[b]
        Lᵇ = length(seqlabels[b])
        p[:,1:Tᵇ,b] = softmax(x.value[:,1:Tᵇ,b]; dims=1)
        r[:,1:Tᵇ,b], loglikely[b] = CTC(p[:,1:Tᵇ,b], seqlabels[b], blank=blank)
        loglikely[b] /= Lᵇ * 2 + 1
    end

    if x.backprop
        function RNN_Batch_CTC_With_Softmax_Backward()
            if need2computeδ!(x)
                if weight==1.0
                    δ(x) .+=  p - r
                else
                    δ(x) .+= (p - r) .* weight
                end
            end
        end
        push!(graph.backward, RNN_Batch_CTC_With_Softmax_Backward)
    end
    return sum(loglikely)/batchsize
end


"""
    CRNN_Batch_CTC_With_Softmax(x::Variable{Array{T}}, seqlabels::Vector; blank=1, weight=1.0) where T

a batch of padded input sequence is processed by neural networks into `x`

# Inputs
`x`         : 3-D Variable (featdims,timesteps,batchsize), inputs of softmax\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`weight`    : weight for CTC loss

# Structure
    ┌───┐
    │ │ │
    │ W ├──►─┐
    │ │ │    │
    └───┘    │
    ┌───┐    │    ┌───┐          ┌───┐
    │ │ │  ┌─┴─┐  │ │ │ softmax  │ │ │   ┌───────┐
    │ Z ├─►│ × ├─►│ X ├─────────►│ P ├──►│CTCLOSS│◄── (seqLabel)
    │ │ │  └───┘  │ │ │          │ │ │   └───┬───┘
    └───┘         └─┬─┘          └─┬─┘       │
                    │              │+        ▼
                  ┌─┴─┐            ▼       ┌─┴─┐
                  │ │ │          ┌─┴─┐ -   │ │ │
                  │ δ │◄─────────┤ - │──◄──┤ r │
                  │ │ │          └───┘     │ │ │
                  └───┘                    └───┘
"""
function CRNN_Batch_CTC_With_Softmax(x::Variable{Array{T}}, seqlabels::Vector; blank=1, weight=1.0) where T
    featdims, timesteps, batchsize = size(x)
    loglikely = zeros(T, batchsize)
    p = softmax(ᵛ(x); dims=1)
    r = zero(ᵛ(x))

    Threads.@threads for b = 1:batchsize
        r[:,:,b], loglikely[b] = CTC(p[:,:,b], seqlabels[b], blank=blank)
        loglikely[b] /= length(seqlabels[b]) * 2 + 1
    end

    if x.backprop
        function CRNN_Batch_CTC_With_Softmax_Backward()
            if need2computeδ!(x)
                if weight==1.0
                    δ(x) .+=  p - r
                else
                    δ(x) .+= (p - r) .* weight
                end
            end
        end
        push!(graph.backward, CRNN_Batch_CTC_With_Softmax_Backward)
    end
    return sum(loglikely)/batchsize
end
