export CTC
export CTCGreedySearch
export DNN_CTC_With_Softmax
export DNN_Batch_CTC_With_Softmax
export RNN_Batch_CTC_With_Softmax


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


LogZero = -floatmax(Float32)


function LogSum2Exp(a::Real, b::Real)
	if a <= LogZero
        a = LogZero
    end
	if b <= LogZero
        b = LogZero
    end
	return (max(a,b) + log(1.0 + exp(-abs(a-b))));
end


function LogSum3Exp(a::Real, b::Real, c::Real)
    return LogSum2Exp(LogSum2Exp(a,b),c)
end


function LogSumExp(a)
    tmp = LogZero
    for i = 1:length(a)
        tmp = LogSum2Exp(tmp, a[i])
    end
    return tmp
end


"""
    CTC(p::Array{T,2}, seq) where T -> target, lossvalue
# inputs
`p`: probability of softmax output\n
`seq`: label seq like [2 3 6 5], 1 is blank, so minimum of it is 2.

# outputs
`target`: target of softmax's output\n
`lossvalue`: negative log-likelyhood
"""
function CTC(p::Array{TYPE,2}, seq) where TYPE
    S,T = size(p)           # assert p is a 2-D tensor
    L = length(seq)*2 + 1   # topology length with blanks
    a = fill(LogZero, L,T)  # 𝜶 = p(s[k,t], x[1:t])
    b = fill(LogZero, L,T)  # 𝛃  = p(x[t+1:T] | s[k,t])
    r = zero(p)             # 𝜸 = classWiseSum(𝜶 .* 𝛃)

    if L>1
        a[1,1] = log(p[    1, 1])
        a[2,1] = log(p[seq[1],1])
        b[L-1,T] = 0.0
        b[L-0,T] = 0.0
    else
        a[1,1] = log(p[1,1])
        b[L,T] = 0.0
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

    logsum = LogZero
    for s = 1:L
        logsum = LogSum2Exp(logsum, a[s,1] + b[s,1])
    end

	g = exp.((a + b) .- logsum)
    for s = 1:L
        if mod(s,2)==1
			r[1,:] .+= g[s,:]
        else
			i = div(s,2)
			r[seq[i],:] .+= g[s,:]
        end
    end
    return r, -logsum
end


"""
    CTCGreedySearch(x::Array)
remove repeats and blanks of argmax(x, dims=1)
"""
function CTCGreedySearch(x::Array)
	# blank 映射到 1
    hyp = []
    idx = argmax(x,dims=1)
    for i = 1:length(idx)
        maxid = idx[i][1]
        if !((i!=1 && idx[i][1]==idx[i-1][1]) || (idx[i][1]==1))
            push!(hyp,idx[i][1])
        end
    end
    return hyp
end


"""
    DNN_CTC_With_Softmax(var::Variable, seq)
    
`var`: 2-D Variable, input sequence.\n
`seq`: 1-D Array, input sequence's label.
"""
function DNN_CTCLoss_With_Softmax(var::Variable{Array{T}}, seq) where T
    # for case batchsize==1
	p = softmax(var.value; dims=1)
    C = eltype(p)((length(seq)+1) / var.shape[2])
    r, loglikely = CTC(p, seq)
    if var.backprop
        function DNN_CTCLoss_With_Softmax_Backward()
            if need2computeδ!(var)
                var.delta += (p - r) .* C
            end
        end
        push!(graph.backward, DNN_CTCLoss_With_Softmax_Backward)
    end
    return loglikely * C
end


"""
    DNN_Batch_CTC_With_Softmax(var::Variable, seq, inputLengths, labelLengths)
    
`var`: 2-D Variable, resulted by a batch of concatenated input sequence.\n
`seq`: 1-D Array, concatenated by a batch of input sequence label.\n
`inputLengths`: 1-D Array which records each input sequence's length.\n
`labelLengths`: 1-D Array which records input sequence label's length.
"""
function DNN_Batch_CTCLoss_With_Softmax(var::Variable{Array{T}}, seq, inputLengths, labelLengths) where T
	LogLikely = 0.0
	batchsize = length(inputLengths)
	probs = softmax(var.value; dims=1)
	gamma = zero(probs)
	sidI,eidI = indexbounds(inputLengths)
	sidL,eidL = indexbounds(labelLengths)

	Threads.@threads for i = 1:batchsize
		IDI = sidI[i]:eidI[i]
		IDL = sidL[i]:eidL[i]
		CST = (length(IDL)+1) / length(IDI)
		gamma[:,IDI], loglikely = CTC(probs[:,IDI], seq[IDL])
		gamma[:,IDI] .*= CST
		probs[:,IDI] .*= CST
		LogLikely += loglikely
	end

    if var.backprop
        function DNN_Batch_CTCLoss_With_Softmax_Backward()
            if need2computeδ!(var)
                var.delta += probs - gamma
            end
        end
        push!(graph.backward, DNN_Batch_CTCLoss_With_Softmax_Backward)
    end
    return LogLikely/batchsize
end


"""
    RNN_Batch_CTC_With_Softmax(var::Variable, seq, inputLengths, labelLengths)
    
`var`: 3-D Variable (featdims,timesteps,batchsize), resulted by a batch of padded input sequence.\n
`seq`: 1-D Array concatenated by a batch of input sequence label.\n
`inputLengths`: 1-D Array which records each input sequence's length.\n
`labelLengths`: 1-D Array which records input sequence label's length.\n
"""
function RNN_Batch_CTC_With_Softmax(var::Variable{Array{T}}, seqlabel, inputLengths, labelLengths) where T
    # assert (featdims,timesteps,batchsize) = size(var)
    probs = zero(var.value)
 	gamma = zero(var.value)
    LogLikely = 0.0

    Threads.@threads for b = 1:batchsize
        Tᵇ = inputLengths[b]
        C = (labelLengths[b]+1) / inputLengths[b]
        probs[:,1:Tᵇ,b] = softmax(var.value[:,1:Tᵇ,b]; dims=1)
        gamma[:,1:Tᵇ,b], loglikely = CTC(probs[:,1:Tᵇ,b], seqlabel[b])
        gamma[:,1:Tᵇ,b] .*= C
        probs[:,1:Tᵇ,b] .*= C
        LogLikely += loglikely
    end

    if var.backprop
        function RNN_Batch_CTCLoss_With_Softmax_Backward()
            if need2computeδ!(var)
                var.delta += probs - gamma
            end
        end
        push!(graph.backward, RNN_Batch_CTCLoss_With_Softmax_Backward)
    end
    return LogLikely/batchsize
end

