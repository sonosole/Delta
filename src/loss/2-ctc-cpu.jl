export CTC
export CTCGreedySearch
export DNN_CTC_With_Softmax
export DNN_Batch_CTC_With_Softmax
export RNN_Batch_CTC_With_Softmax
export CRNN_Batch_CTC_With_Softmax
export indexbounds


function indexbounds(lengthArray)
    # assert lengthArray has no 0 element
    acc = 0
    num = length(lengthArray)
    s = ones(Int,num,1)
    e = ones(Int,num,1)
    for i = 1:num
        s[i] += acc
        e[i] = s[i] + lengthArray[i] - 1
        acc += lengthArray[i]
    end
    return (s,e)
end


"""
    CTC(p::Array{T,2}, seq) where T -> (target, lossvalue)
# inputs
`p`: probability of softmax output\n
`seq`: label seq like [2 3 6 5], 1 is blank, so minimum of it is 2. If `p` has no label (e.g. pure noise or oov) then `seq` is []

# outputs
`target`: target of softmax's output\n
`lossvalue`: negative log-likelyhood
"""
function CTC(p::Array{TYPE,2}, seq) where TYPE
    Log0 = LogZero(TYPE)   # approximate -Inf of TYPE
    ZERO = TYPE(0)         # typed zero,e.g. Float32(0)
    S, T = size(p)         # assert p is a 2-D tensor
    L = length(seq)*2 + 1  # topology length with blanks
    a = fill!(Array{TYPE,2}(undef,L,T), Log0)    # 𝜶 = p(s[k,t], x[1:t]), k in CTC topology's indexing
    b = fill!(Array{TYPE,2}(undef,L,T), Log0)    # 𝛃 = p(x[t+1:T] | s[k,t]), k in CTC topology's indexing
    r = fill!(Array{TYPE,2}(undef,S,T), ZERO)    # 𝜸 = p(s[k,t] | x[1:T]), k in softmax's indexing

    if L>1
        a[1,1] = log(p[    1, 1])
        a[2,1] = log(p[seq[1],1])
        b[L-1,T] = ZERO
        b[L-0,T] = ZERO
    else
        a[1,1] = log(p[1,1])
        b[L,T] = ZERO
    end

    # --- forward in log scale ---
    for t = 2:T
        first = max(1,L-2*(T-t)-1);
        lasst = min(2*t,L);
        for s = first:lasst
            i = div(s,2);
            if s==1
                a[s,t] = a[s,t-1] + log(p[1,t])
            elseif mod(s,2)==1
                a[s,t] = LogSum2Exp(a[s,t-1], a[s-1,t-1]) + log(p[1,t])
            elseif s==2
                a[s,t] = LogSum2Exp(a[s,t-1], a[s-1,t-1]) + log(p[seq[i],t])
            elseif seq[i]==seq[i-1]
				a[s,t] = LogSum2Exp(a[s,t-1], a[s-1,t-1]) + log(p[seq[i],t])
            else
                a[s,t] = LogSum3Exp(a[s,t-1], a[s-1,t-1], a[s-2,t-1]) + log(p[seq[i],t])
            end
        end
    end

    # --- backward in log scale ---
    for t = T-1:-1:1
        first = max(1,L-2*(T-t)-1)
        lasst = min(2*t,L)
        for s = first:lasst
            i = div(s,2)
            j = div(s+1,2)
            if s==L
                b[s,t] = b[s,t+1] + log(p[1,t+1])
            elseif mod(s,2)==1
                b[s,t] = LogSum2Exp(b[s,t+1] + log(p[1,t+1]), b[s+1,t+1] + log(p[seq[j],t+1]))
            elseif s==L-1
                b[s,t] = LogSum2Exp(b[s,t+1] + log(p[seq[i],t+1]), b[s+1,t+1] + log(p[1,t+1]))
            elseif seq[i]==seq[i+1]
				b[s,t] = LogSum2Exp(b[s,t+1] + log(p[seq[i],t+1]), b[s+1,t+1] + log(p[1,t+1]))
            else
                b[s,t] = LogSum3Exp(b[s,t+1] + log(p[seq[i],t+1]), b[s+1,t+1] + log(p[1,t+1]), b[s+2,t+1] + log(p[seq[i+1],t+1]))
            end
        end
    end

    logsum = Log0
    for s = 1:L
        logsum = LogSum2Exp(logsum, a[s,1] + b[s,1])
    end

    g = exp.((a + b) .- logsum)

    # deprecated inefficient implementation, although it's easier to understand
    # for s = 1:L
    #     if mod(s,2)==1
    #         r[1,:] .+= g[s,:]
    #     else
    #         i = div(s,2)
    #         r[seq[i],:] .+= g[s,:]
    #     end
    # end

    # reduce first line
    r[1,:] .+= g[1,:]
    # reduce rest lines
    for n = 1:length(seq)
        s = n<<1
        r[seq[n],:] += g[s,  t]
        r[1     ,:] += g[s+1,t]
    end

    return r, -logsum
end


"""
    CTCGreedySearch(x::Array)
remove repeats and blanks of argmax(x, dims=1)
"""
function CTCGreedySearch(x::Array)
    # blank --> 1
    hyp = Vector{Int}(undef,0)
    idx = argmax(x,dims=1)
    for i = 1:length(idx)
        maxid = idx[i][1]
        if !((i!=1 && idx[i][1]==idx[i-1][1]) || (idx[i][1]==1))
            push!(hyp, maxid)
        end
    end
    return hyp
end


"""
    DNN_CTC_With_Softmax(var::Variable, seq)

`var`: 2-D Variable, input sequence.\n
`seq`: 1-D Array, input sequence's label.
"""
function DNN_CTC_With_Softmax(var::Variable{Array{T}}, seq) where T
    # for case batchsize==1
    p = softmax(var.value; dims=1)
    L = length(seq) * 2 + 1
    r, loglikely = CTC(p, seq)
    if var.backprop
        function DNN_CTC_With_Softmax_Backward()
            if need2computeδ!(var)
                var.delta += p - r
            end
        end
        push!(graph.backward, DNN_CTC_With_Softmax_Backward)
    end
    return loglikely / L
end


"""
    DNN_Batch_CTC_With_Softmax(var::Variable, seq, inputLengths, labelLengths)

`var`: 2-D Variable, resulted by a batch of concatenated input sequence.\n
`seq`: 1-D Array, concatenated by a batch of input sequence label.\n
`inputLengths`: 1-D Array which records each input sequence's length.\n
`labelLengths`: 1-D Array which records input sequence label's length.
"""
function DNN_Batch_CTC_With_Softmax(var::Variable{Array{T}}, seq, inputLengths, labelLengths) where T
    batchsize = length(inputLengths)
    loglikely = zeros(T, batchsize)
    probs = softmax(var.value; dims=1)
    gamma = zero(probs)
    sidI,eidI = indexbounds(inputLengths)
    sidL,eidL = indexbounds(labelLengths)

    Threads.@threads for b = 1:batchsize
        IDI = sidI[b]:eidI[b]
        IDL = sidL[b]:eidL[b]
        gamma[:,IDI], loglikely[b] = CTC(probs[:,IDI], seq[IDL])
        loglikely[b] /= length(IDL) * 2 + 1
    end

    if var.backprop
        function DNN_Batch_CTC_With_Softmax_Backward()
            if need2computeδ!(var)
                var.delta += probs - gamma
            end
        end
        push!(graph.backward, DNN_Batch_CTC_With_Softmax_Backward)
    end
    return sum(loglikely)/batchsize
end


"""
    RNN_Batch_CTC_With_Softmax(var::Variable, seqlabels, inputLengths, labelLengths)

`var`: 3-D Variable with shape (featdims,timesteps,batchsize), resulted by a batch of padded input sequence.\n
`seqlabels`: 1-D Array concatenated by a batch of input sequence's label.\n
`inputLengths`: 1-D Array which records each input sequence's length.\n
`labelLengths`: 1-D Array which records all labels' length.\n
"""
function RNN_Batch_CTC_With_Softmax(var::Variable{Array{T}}, seqlabels, inputLengths, labelLengths) where T
    batchsize = length(inputLengths)
    loglikely = zeros(T, batchsize)
    probs = zero(var.value)
    gamma = zero(var.value)

    Threads.@threads for b = 1:batchsize
        Tᵇ = inputLengths[b]
        Lᵇ = labelLengths[b]
        probs[:,1:Tᵇ,b] = softmax(var.value[:,1:Tᵇ,b]; dims=1)
        gamma[:,1:Tᵇ,b], loglikely[b] = CTC(probs[:,1:Tᵇ,b], seqlabels[b])
        loglikely[   b] /= Lᵇ * 2 + 1
    end

    if var.backprop
        function RNN_Batch_CTC_With_Softmax_Backward()
            if need2computeδ!(var)
                var.delta += probs - gamma
            end
        end
        push!(graph.backward, RNN_Batch_CTC_With_Softmax_Backward)
    end
    return sum(loglikely)/batchsize
end


"""
    CRNN_CTCLoss_With_Softmax(var::Variable{Array{T}}, seqlabels::Vector) where T -> LogLikely

`var`: 3-D Variable (featdims,timesteps,batchsize), resulted by a batch of padded input sequence.\n
`seqlabels`: a vector contains a batch of 1-D Array labels.\n
"""
function CRNN_Batch_CTC_With_Softmax(var::Variable{Array{T}}, seqlabels::Vector) where T
    featdims, timesteps, batchsize = size(var)
    probs = softmax(var.value; dims=1)
    gamma = zero(var.value)
    loglikely = zeros(T, batchsize)

    Threads.@threads for b = 1:batchsize
        gamma[:,:,b], loglikely[b] = CTC(probs[:,:,b], seqlabels[b])
        loglikely[b] /= length(seqlabels[b])*2 + 1
    end

    if var.backprop
        function CRNN_Batch_CTC_With_Softmax_Backward()
            if need2computeδ!(var)
                var.delta += probs - gamma
            end
        end
        push!(graph.backward, CRNN_Batch_CTC_With_Softmax_Backward)
    end
    return sum(loglikely)/batchsize
end
