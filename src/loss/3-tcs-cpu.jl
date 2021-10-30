export TCS
export TCSGreedySearch
export CRNN_Batch_TCS_With_Softmax
export DNN_Batch_TCS_With_Softmax
export RNN_Batch_TCS_With_Softmax

"""
    TCS(p::Array{T,2}, seq) where T -> target, lossvalue
# inputs
`p`: probability of softmax output\n
`seq`: label seq like [1 2 3 1 2 3 1 2 6 1 2 5 1], of which 1 is background state 2 is foreground state.
       If `p` has no label (e.g. pure noise or oov) then `seq` is [1]

# outputs
`target`: target of softmax's output\n
`lossvalue`: negative log-likelyhood
"""
function TCS(p::Array{TYPE,2}, seq) where TYPE
    Log0 = LogZero(TYPE)   # approximate -Inf of TYPE
    ZERO = TYPE(0)         # typed zero,e.g. Float32(0)
    S, T = size(p)         # assert p is a 2-D tensor
    L = length(seq)        # topology length
    r = fill!(Array{TYPE,2}(undef,S,T), ZERO)  # 𝜸 = p(s[k,t] | x[1:T]), k in softmax's indexing

    if L == 1
        r[1,:] .= TYPE(1)
        return r, - sum(log.(p[seq[1],:]))
    end

    a = fill!(Array{TYPE,2}(undef,L,T), Log0)  # 𝜶 = p(s[k,t], x[1:t]), k in TCS topology's indexing
    b = fill!(Array{TYPE,2}(undef,L,T), Log0)  # 𝛃 = p(x[t+1:T] | s[k,t]), k in TCS topology's indexing
    a[1,1] = log(p[seq[1],1])
    a[2,1] = log(p[seq[2],1])
    b[L-1,T] = ZERO
    b[L-0,T] = ZERO

    # --- forward in log scale ---
	for t = 2:T
	    for s = 1:L
	        if s!=1
				R = mod(s,3)
	            if R==1 || s==2 || R==0
	                a[s,t] = LogSum2Exp(a[s,t-1], a[s-1,t-1])
	            elseif R==2
	                a[s,t] = LogSum3Exp(a[s,t-1], a[s-1,t-1], a[s-2,t-1])
	            end
	        else
	            a[s,t] = a[s,t-1]
	        end
	        a[s,t] += log(p[seq[s],t])
	    end
	end

    # --- backward in log scale ---
	for t = T-1:-1:1
		for s = L:-1:1
			Q = b[s,t+1] + log(p[seq[s],t+1])
			if s!=L
				R = mod(s,3)
				V = b[s+1,t+1] + log(p[seq[s+1],t+1])
				if R==1 || R==2 || s==L-1
					b[s,t] = LogSum2Exp(Q, V)
				elseif R==0
					b[s,t] = LogSum3Exp(Q, V, b[s+2,t+1] + log(p[seq[s+2],t+1]))
				end
			else
				b[s,t] = Q
			end
		end
	end

    # loglikely of TCS
    logsum = LogSum3Exp(Log0, a[1,1] + b[1,1], a[2,1] + b[2,1])

    # log weight --> normal probs
	g = exp.((a + b) .- logsum)

    # Reduce First line
    r[seq[1],:] .+= g[1,:]
    # Reduce other lines
    for n = 1:div(L-1,3)
        s = 3*n
        r[seq[s-1],:] .+= g[s-1,:]  # reduce forground states
        r[seq[s  ],:] .+= g[s,  :]  # reduce labels' states
        r[seq[s+1],:] .+= g[s+1,:]  # reduce background state
    end

    return r, -logsum
end


function TCSGreedySearch(x::Array)
	# Backgroud  --> 1 index
    # Foreground --> 2 index
    hyp = []
    idx = argmax(x,dims=1)
    for i = 1:length(idx)
        maxid = idx[i][1]
        if !((i!=1 && idx[i][1]==idx[i-1][1]) || (idx[i][1]==1) || (idx[i][1]==2))
            push!(hyp, maxid)
        end
    end
    return hyp
end


"""
    DNN_Batch_TCS_With_Softmax(var::Variable{Array{T}}, seq, inputLengths, labelLengths) where T

`var`: 2-D Variable, resulted by a batch of concatenated input sequence.\n
`seq`: 1-D Array, concatenated by a batch of sequential labels.\n
`inputLengths`: 1-D Array which records each input sequence's length.\n
`labelLengths`: 1-D Array which records input sequence label's length.
"""
function DNN_Batch_TCS_With_Softmax(var::Variable{Array{T}}, seq, inputLengths, labelLengths) where T
    batchsize = length(inputLengths)
    loglikely = zeros(T, batchsize)
    probs = softmax(var.value; dims=1)
    gamma = zero(probs)
    sidI,eidI = indexbounds(inputLengths)  # starting and ending indexes of inputs
    sidL,eidL = indexbounds(labelLengths)  # starting and ending indexes of labels

    Threads.@threads for b = 1:batchsize
        IDI = sidI[b]:eidI[b]
        IDL = sidL[b]:eidL[b]
        gamma[:,IDI], loglikely[b] = TCS(probs[:,IDI], seq[IDL])
        loglikely[b] /= length(IDL)
    end

    if var.backprop
        function DNN_Batch_TCS_With_Softmax_Backward()
            if need2computeδ!(var)
                var.delta += probs - gamma
            end
        end
        push!(graph.backward, DNN_Batch_TCS_With_Softmax_Backward)
    end
    return sum(loglikely)/batchsize
end


"""
    RNN_Batch_TCS_With_Softmax(var::Variable{Array{T}}, seqlabels, inputLengths, labelLengths) where T

`var`: 3-D Variable with shape (featdims,timesteps,batchsize), resulted by a batch of padded input sequence.\n
`seqlabels`: a Vector contains multiple 1-D sequential labels.\n
`inputLengths`: 1-D Array which records each input sequence's length.\n
`labelLengths`: 1-D Array which records all labels' length.\n
"""
function RNN_Batch_TCS_With_Softmax(var::Variable{Array{T}}, seqlabels, inputLengths, labelLengths) where T
    batchsize = length(inputLengths)
    loglikely = zeros(T, batchsize)
    probs = zero(var.value)
    gamma = zero(var.value)

    Threads.@threads for b = 1:batchsize
        Tᵇ = inputLengths[b]
        Lᵇ = labelLengths[b]
        probs[:,1:Tᵇ,b] = softmax(var.value[:,1:Tᵇ,b]; dims=1)
        gamma[:,1:Tᵇ,b], loglikely[b] = TCS(probs[:,1:Tᵇ,b], seqlabels[b])
        loglikely[   b] /= Lᵇ
    end

    if var.backprop
        function RNN_Batch_TCS_With_Softmax_Backward()
            if need2computeδ!(var)
                var.delta += probs - gamma
            end
        end
        push!(graph.backward, RNN_Batch_TCS_With_Softmax_Backward)
    end
    return sum(loglikely)/batchsize
end


function CRNN_Batch_TCS_With_Softmax(var::Variable{Array{T}}, seqlabels::Vector) where T
    featdims, timesteps, batchsize = size(var)
    probs = softmax(var.value; dims=1)
    gamma = zero(var.value)
    loglikely = zeros(T, batchsize)

    Threads.@threads for b = 1:batchsize
        gamma[:,:,b], loglikely[b] = TCS(probs[:,:,b], seqlabels[b])
        loglikely[b] /= length(seqlabels[b])
    end

    if var.backprop
        function CRNN_Batch_TCS_With_Softmax_Backward()
            if need2computeδ!(var)
                var.delta += probs - gamma
            end
        end
        push!(graph.backward, CRNN_Batch_TCS_With_Softmax_Backward)
    end
    return sum(loglikely)/batchsize
end
