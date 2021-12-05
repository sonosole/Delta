export DNN_CTC
export DNN_Batch_CTC
export RNN_Batch_CTC
export CRNN_Batch_CTC


"""
    DNN_CTC(p::Variable, seq; blank=1, weight=1.0)
for case batchsize==1 for test case

# Inputs
`seq`    : 1-D Array, input sequence's label.\n
`weight` : weight for CTC loss
`p`      : 2-D Variable, var after softmax, maybe weighted like below, i.e. p = w .* softmax(x)\n

# Structure
                   ┌───┐
                   │ │ │
                   │ W ├──►──┐
                   │ │ │     │
                   └───┘     │
    ┌───┐          ┌───┐     │     ┌───┐
    │ │ │ softmax  │ │ │   ┌─┴─┐   │ │ │   ┌───────┐
    │ X ├─────────►│ Y ├──►│ × ├──►│ P ├──►│CTCLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └───┘   │ │ │   └───┬───┘
    └─┬─┘          └─┬─┘           └───┘       │
      │              │+                        ▼
    ┌─┴─┐            ▼                       ┌─┴─┐
    │ │ │          ┌─┴─┐ -                   │ │ │
    │ δ │◄─────────┤ - │─────────◄───────────┤ r │
    │ │ │          └───┘                     │ │ │
    └───┘                                    └───┘
"""
function DNN_CTC(p::Variable{Array{T}}, seq; blank=1, weight=1.0) where T
    L = length(seq) * 2 + 1
    r, loglikely = CTC(ᵛ(p), seq, blank=blank)
    if p.backprop
        function DNN_CTC_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .-= r ./ ᵛ(p)
                else
                    δ(p) .-= r ./ ᵛ(p) .* weight
                end
            end
        end
        push!(graph.backward, DNN_CTC_Backward)
    end
    return loglikely / L
end


"""
    DNN_Batch_CTC(p::Variable{Array{T}}, seqlabels::Vector, inputlens; blank=1, weight=1.0) where T

# Inputs
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`inputlens` : records each input sequence's length, like [20,17,...]\n
`weight`    : weight for CTC loss
`p`         : 2-D Variable, var after softmax, maybe weighted like below, i.e. p = w .* softmax(x)\n

# Structure
                   ┌───┐
                   │ │ │
                   │ W ├──►──┐
                   │ │ │     │
                   └───┘     │
    ┌───┐          ┌───┐     │     ┌───┐
    │ │ │ softmax  │ │ │   ┌─┴─┐   │ │ │   ┌───────┐
    │ X ├─────────►│ Y ├──►│ × ├──►│ P ├──►│CTCLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └───┘   │ │ │   └───┬───┘
    └─┬─┘          └─┬─┘           └───┘       │
      │              │+                        ▼
    ┌─┴─┐            ▼                       ┌─┴─┐
    │ │ │          ┌─┴─┐ -                   │ │ │
    │ δ │◄─────────┤ - │─────────◄───────────┤ r │
    │ │ │          └───┘                     │ │ │
    └───┘                                    └───┘
"""
function DNN_Batch_CTC(p::Variable{Array{T}}, seqlabels::Vector, inputlens; blank=1, weight=1.0) where T
    batchsize = length(inputLengths)
    loglikely = zeros(T, batchsize)
    I, F = indexbounds(inputlens)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        span = I[b]:F[b]
        r[:,span], loglikely[b] = CTC(p.value[:,span], seqlabels[b], blank=blank)
        loglikely[b] /= length(seqlabels[b]) * 2 + 1
    end

    if p.backprop
        function DNN_Batch_CTC_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .-= r ./ ᵛ(p)
                else
                    δ(p) .-= r ./ ᵛ(p) .* weight
                end
            end
        end
        push!(graph.backward, DNN_Batch_CTC_Backward)
    end
    return sum(loglikely)/batchsize
end


"""
    RNN_Batch_CTC(p::Variable, seqlabels::Vector, inputlens; blank=1, weight=1.0) where T
# Inputs
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`inputlens` : each input's length, like [19,97,...]\n
`weight`    : weight for CTC loss
`p`         : 3-D Variable with shape (featdims,timesteps,batchsize), a batch of padded input sequence.\n
              var after softmax, maybe weighted like below, i.e. p = w .* softmax(x)\n

# Structure
                   ┌───┐
                   │ │ │
                   │ W ├──►──┐
                   │ │ │     │
                   └───┘     │
    ┌───┐          ┌───┐     │     ┌───┐
    │ │ │ softmax  │ │ │   ┌─┴─┐   │ │ │   ┌───────┐
    │ X ├─────────►│ Y ├──►│ × ├──►│ P ├──►│CTCLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └───┘   │ │ │   └───┬───┘
    └─┬─┘          └─┬─┘           └───┘       │
      │              │+                        ▼
    ┌─┴─┐            ▼                       ┌─┴─┐
    │ │ │          ┌─┴─┐ -                   │ │ │
    │ δ │◄─────────┤ - │─────────◄───────────┤ r │
    │ │ │          └───┘                     │ │ │
    └───┘                                    └───┘
"""
function RNN_Batch_CTC(p::Variable{Array{T}}, seqlabels::Vector, inputlens; blank=1, weight=1.0) where T
    batchsize = length(inputlens)
    loglikely = zeros(T, batchsize)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        Tᵇ = inputlens[b]
        Lᵇ = length(seqlabels[b])
        r[:,1:Tᵇ,b], loglikely[b] = CTC(p.value[:,1:Tᵇ,b], seqlabels[b], blank=blank)
        loglikely[b] /= Lᵇ * 2 + 1
    end

    if p.backprop
        function RNN_Batch_CTC_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .-= r ./ ᵛ(p)
                else
                    δ(p) .-= r ./ ᵛ(p) .* weight
                end
            end
        end
        push!(graph.backward, RNN_Batch_CTC_Backward)
    end
    return sum(loglikely)/batchsize
end


"""
    CRNN_Batch_CTC(p::Variable{Array{T}}, seqlabels::Vector) where T -> LogLikely

# Inputs
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`weight`    : weight for CTC loss
`p`         : 3-D Variable with shape (featdims,timesteps,batchsize), a batch of padded input sequence.\n
              var after softmax, maybe weighted like below, i.e. p = w .* softmax(x)\n
# Structure
                   ┌───┐
                   │ │ │
                   │ W ├──►──┐
                   │ │ │     │
                   └───┘     │
    ┌───┐          ┌───┐     │     ┌───┐
    │ │ │ softmax  │ │ │   ┌─┴─┐   │ │ │   ┌───────┐
    │ X ├─────────►│ Y ├──►│ × ├──►│ P ├──►│CTCLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └───┘   │ │ │   └───┬───┘
    └─┬─┘          └─┬─┘           └───┘       │
      │              │+                        ▼
    ┌─┴─┐            ▼                       ┌─┴─┐
    │ │ │          ┌─┴─┐ -                   │ │ │
    │ δ │◄─────────┤ - │─────────◄───────────┤ r │
    │ │ │          └───┘                     │ │ │
    └───┘                                    └───┘
"""
function CRNN_Batch_CTC(p::Variable{Array{T}}, seqlabels::Vector; blank=1, weight=1.0) where T
    featdims, timesteps, batchsize = size(x)
    loglikely = zeros(T, batchsize)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        r[:,:,b], loglikely[b] = CTC(p.value[:,:,b], seqlabels[b], blank=blank)
        loglikely[b] /= length(seqlabels[b]) * 2 + 1
    end

    if p.backprop
        function CRNN_Batch_CTC_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .-= r ./ ᵛ(p)
                else
                    δ(p) .-= r ./ ᵛ(p) .* weight
                end
            end
        end
        push!(graph.backward, CRNN_Batch_CTC_Backward)
    end
    return sum(loglikely)/batchsize
end
