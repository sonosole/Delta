export CRNN_Batch_TCS
export DNN_Batch_TCS
export RNN_Batch_TCS


"""
    DNN_Batch_TCS(p::Variable{Array{T}},
                  seqlabels::Vector,
                  inputlens;
                  background::Int=1,
                  foreground::Int=2,
                  weight=1.0) where T

a batch of concatenated input sequence is processed by neural networks into `p`

# Inputs
`p`         : 2-D Variable, probability or weighted probability\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`inputlens` : records each input sequence's length, like [20,17,...]\n
`weight`    : weight for TCS loss

# Structure
                   ┌───┐
                   │ │ │
                   │ W ├──►──┐
                   │ │ │     │
                   └───┘     │
    ┌───┐          ┌───┐     │     ┌───┐
    │ │ │ softmax  │ │ │   ┌─┴─┐   │ │ │   ┌───────┐
    │ X ├─────────►│ Y ├──►│ × ├──►│ P ├──►│TCSLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └───┘   │ │ │   └───┬───┘
    └─┬─┘          └─┬─┘           └───┘       │
      │              │+                        ▼
    ┌─┴─┐            ▼                       ┌─┴─┐
    │ │ │          ┌─┴─┐ -                   │ │ │
    │ δ │◄─────────┤ - │─────────◄───────────┤ r │
    │ │ │          └───┘                     │ │ │
    └───┘                                    └───┘
"""
function DNN_Batch_TCS(p::Variable{Array{T}},
                       seqlabels::Vector,
                       inputlens;
                       background::Int=1,
                       foreground::Int=2,
                       weight=1.0) where T
    batchsize = length(seqlabels)
    loglikely = zeros(T, batchsize)
    I, F = indexbounds(inputlens)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        span = I[b]:F[b]
        r[:,span], loglikely[b] = TCS(p.value[:,span], seqlabels[b], background=background, foreground=foreground)
        loglikely[b] /= length(seqlabels[b]) * 3 + 1
    end

    if p.backprop
        function DNN_Batch_TCS_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .-= r ./ (ᵛ(p) .+ eps(T))
                else
                    δ(p) .-= r ./ (ᵛ(p) .+ eps(T)) .* weight
                end
            end
        end
        push!(graph.backward, DNN_Batch_TCS_Backward)
    end
    return sum(loglikely)/batchsize
end


"""
    RNN_Batch_TCS(p::Variable{Array{T}},
                  seqlabels::Vector,
                  inputlens;
                  background::Int=1,
                  foreground::Int=2,
                  weight=1.0) where T

a batch of padded input sequence is processed by neural networks into `p`

# Inputs
`p`         : 3-D Variable with shape (featdims,timesteps,batchsize), probability or weighted probability\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`inputlens` : records each input sequence's length, like [20,17,...]\n
`weight`    : weight for TCS loss

# Structure
                   ┌───┐
                   │ │ │
                   │ W ├──►──┐
                   │ │ │     │
                   └───┘     │
    ┌───┐          ┌───┐     │     ┌───┐
    │ │ │ softmax  │ │ │   ┌─┴─┐   │ │ │   ┌───────┐
    │ X ├─────────►│ Y ├──►│ × ├──►│ P ├──►│TCSLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └───┘   │ │ │   └───┬───┘
    └─┬─┘          └─┬─┘           └───┘       │
      │              │+                        ▼
    ┌─┴─┐            ▼                       ┌─┴─┐
    │ │ │          ┌─┴─┐ -                   │ │ │
    │ δ │◄─────────┤ - │─────────◄───────────┤ r │
    │ │ │          └───┘                     │ │ │
    └───┘                                    └───┘
"""
function RNN_Batch_TCS(p::Variable{Array{T}},
                       seqlabels::Vector,
                       inputlens;
                       background::Int=1,
                       foreground::Int=2,
                       weight=1.0) where T
    batchsize = length(seqlabels)
    loglikely = zeros(T, batchsize)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        Tᵇ = inputlens[b]
        Lᵇ = length(seqlabels[b])
        r[:,1:Tᵇ,b], loglikely[b] = TCS(p.value[:,1:Tᵇ,b], seqlabels[b], background=background, foreground=foreground)
        loglikely[b] /= Lᵇ * 3 + 1
    end

    if p.backprop
        function RNN_Batch_TCS_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .-= r ./ (ᵛ(p) .+ eps(T))
                else
                    δ(p) .-= r ./ (ᵛ(p) .+ eps(T)) .* weight
                end
            end
        end
        push!(graph.backward, RNN_Batch_TCS_Backward)
    end
    return sum(loglikely)/batchsize
end

"""
    CRNN_Batch_TCS(p::Variable{Array{T}},
                   seqlabels::Vector;
                   background::Int=1,
                   foreground::Int=2,
                   weight=1.0) where T

a batch of padded input sequence is processed by neural networks into `p`

# Main Inputs
`p`            : 3-D Variable with shape (featdims,timesteps,batchsize), probability or weighted probability\n
`seqlabels`    : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`weight`       : weight for TCS loss

# Structure
                   ┌───┐
                   │ │ │
                   │ W ├──►──┐
                   │ │ │     │
                   └───┘     │
    ┌───┐          ┌───┐     │     ┌───┐
    │ │ │ softmax  │ │ │   ┌─┴─┐   │ │ │   ┌───────┐
    │ X ├─────────►│ Y ├──►│ × ├──►│ P ├──►│TCSLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └───┘   │ │ │   └───┬───┘
    └─┬─┘          └─┬─┘           └───┘       │
      │              │+                        ▼
    ┌─┴─┐            ▼                       ┌─┴─┐
    │ │ │          ┌─┴─┐ -                   │ │ │
    │ δ │◄─────────┤ - │─────────◄───────────┤ r │
    │ │ │          └───┘                     │ │ │
    └───┘                                    └───┘
"""
function CRNN_Batch_TCS(p::Variable{Array{T}},
                        seqlabels::Vector;
                        background::Int=1,
                        foreground::Int=2,
                        weight=1.0) where T
    featdims, timesteps, batchsize = size(p)
    loglikely = zeros(T, batchsize)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        r[:,:,b], loglikely[b] = TCS(p.value[:,:,b], seqlabels[b], background=background, foreground=foreground)
        loglikely[b] /= length(seqlabels[b]) * 3 + 1
    end

    if p.backprop
        function CRNN_Batch_TCS_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .-= r ./ (ᵛ(p) .+ eps(T))
                else
                    δ(p) .-= r ./ (ᵛ(p) .+ eps(T)) .* weight
                end
            end
        end
        push!(graph.backward, CRNN_Batch_TCS_Backward)
    end
    return sum(loglikely)/batchsize
end
