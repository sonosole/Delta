# Time Delayed Classifier
export seqtdc, TDC
export TDCGreedySearch
export TDCGreedySearchWithTimestamp


function seqtdc(seq, blank::Int=1, front::Int=2)
    L = length(seq)       # sequence length
    if L ≠ 0
        N = 4 * L         # topology length
        label = zeros(Int, N)
        label[1:4:N] .= blank
        label[2:4:N] .= front
        label[3:4:N] .= seq
        label[4:4:N] .= blank
        return label
    else
        return fill!(zeros(Int, 1), blank)
    end
end


"""
    TDC(p::Array{T,2}, seqlabel; blank::Int=1, front::Int=2) where T

# Topology Example
     ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐             ┌─►─┐    ┌─►─┐    ┌─►─┐             ┌─►─┐    ┌─►─┐    ┌─►─┐
    ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌─────┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌─────┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐
    │blank├─►│front├─►│  A  ├─►│blank├─►│blank├─►│front├─►│  B  ├─►│blank├─►│blank├─►│front├─►│  C  ├─►│blank│
    └─────┘  └─────┘  └──┬──┘  └──┬──┘  └─────┘  └┬───┬┘  └──┬──┘  └──┬──┘  └─────┘  └┬───┬┘  └─────┘  └─────┘
                         │        └────────►──────┘   │      │        └────────►──────┘   │
                         └─────────────────►──────────┘      └─────────────────►──────────┘

"""
function TDC(p::Array{TYPE,2}, seqlabel; blank::Int=1, front::Int=2) where TYPE
    seq  = seqtdc(seqlabel, blank, front)
    Log0 = LogZero(TYPE)   # approximate -Inf of TYPE
    ZERO = TYPE(0)         # typed zero, e.g. Float32(0)
    S, T = size(p)         # assert p is a 2-D tensor
    L = length(seq)        # topology length
    r = fill!(Array{TYPE,2}(undef,S,T), ZERO)  # 𝜸 = p(s[k,t] | x[1:T]), k in softmax's indexing

    if L == 1
        r[seq[1],:] .= TYPE(1)
        return r, - sum(log.(p[seq[1],:]))
    end

    a = fill!(Array{TYPE,2}(undef,L,T), Log0)  # 𝜶 = p(s[k,t], x[1:t]), k in TDC topology's indexing
    b = fill!(Array{TYPE,2}(undef,L,T), Log0)  # 𝛃 = p(x[t+1:T] | s[k,t]), k in TDC topology's indexing
    a[1,1] = log(p[seq[1],1])                  # blank entrance
    a[2,1] = log(p[seq[2],1])                  # front entrance
    b[L-1,T] = ZERO                            # label exit
    b[L  ,T] = ZERO                            # blank exit

    # --- forward in log scale ---
	for t = 2:T
        τ = t-1
	    for s = 1:L
	        if s≠1
				R = mod(s,4)
	            if R==3 || R==0 || s==2
	                a[s,t] = LogSum2Exp(a[s,τ], a[s-1,τ])
                elseif R==2
                    a[s,t] = LogSum4Exp(a[s,τ], a[s-1,τ], a[s-2,τ], a[s-3,τ])
                elseif R==1
                    a[s,t] = a[s-1,τ]
	            end
	        else
	            a[s,t] = a[s,τ]
	        end
	        a[s,t] += log(p[seq[s],t])
	    end
	end

    # --- backward in log scale ---
	for t = T-1:-1:1
        τ = t+1
		for s = L:-1:1
			Q⁰ = b[s,τ] + log(p[seq[s],τ])
			if s≠L
				R = mod(s,4)
				Q¹ = b[s+1,τ] + log(p[seq[s+1],τ])
				if R==2 || s==1 || s==L-1
					b[s,t] = LogSum2Exp(Q⁰, Q¹)
                elseif R==0
                    Q² = b[s+2,τ] + log(p[seq[s+2],τ])
                    b[s,t] = LogSum3Exp(Q⁰, Q¹, Q²)
                elseif R==3
                    Q³ = b[s+3,τ] + log(p[seq[s+3],τ])
                    b[s,t] = LogSum3Exp(Q⁰, Q¹, Q³)
                elseif R==1
                    b[s,t] = Q¹
				end
			else
				b[s,t] = Q⁰
			end
		end
	end

    # loglikely of TCS
    logsum = LogSum2Exp(a[1,1] + b[1,1], a[2,1] + b[2,1])

    # log weight --> normal probs
	g = exp.((a + b) .- logsum)

    # reduce lines
    for n = 1:div(L,4)
        s = n<<2
        r[seq[s-3],:] .+= g[s-3,:]  # reduce blank states
        r[seq[s-2],:] .+= g[s-2,:]  # reduce front states
        r[seq[s-1],:] .+= g[s-1,:]  # reduce labels' states
        r[seq[s  ],:] .+= g[s,  :]  # reduce blank state
    end

    return r, -logsum
end


"""
    TDCGreedySearch(x::Array; dims=1, blank::Int=1, front::Int=2) -> hypothesis
"""
function TDCGreedySearch(x::Array; dims=1, blank::Int=1, front::Int=2)
    hyp = Vector{Int}(undef, 0)
    idx = argmax(x, dims=dims)
    for t = 1:length(idx)
        previous = idx[t≠1 ? t-1 : t][1]
        current  = idx[t][1]
        if !((current==previous && t≠1) ||
             (current==blank) ||
             (current==front))
            push!(hyp, current)
        end
    end
    return hyp
end


"""
    TDCGreedySearchWithTimestamp(x::Array; dims=1, blank::Int=1, front::Int=2) -> hypothesis, timestamp
"""
function TDCGreedySearchWithTimestamp(x::Array; dims=1, blank::Int=1, front::Int=2)
    hyp = Vector{Int}(undef, 0)
    stp = Vector{Float32}(undef, 0)
    idx = argmax(x, dims=dims)
    T   = length(idx)
    for t = 1:T
        previous = idx[t≠1 ? t-1 : t][1]
        current  = idx[t][1]
        if !((current==previous && t≠1) ||
             (current==blank) ||
             (current==front))
            push!(hyp, current)
            push!(stp, t / T)
        end
    end
    return hyp, stp
end



export CRNN_TDC_With_Softmax


function CRNN_TDC_With_Softmax(x::Variable{Array{T}},
                               seqlabels::Vector;
                               blank::Int=1,
                               front::Int=2,
                               reduction::String="seqlen",
                               weight::Float64=1.0) where T
    featdims, timesteps, batchsize = size(x)
    loglikely = zeros(T, batchsize)
    p = softmax(ᵛ(x); dims=1)
    r = zero(ᵛ(x))

    Threads.@threads for b = 1:batchsize
        r[:,:,b], loglikely[b] = TDC(p[:,:,b], seqlabels[b], blank=blank, front=front)
    end

    Δ = p - r
    reduce3d(Δ, loglikely, seqlabels, reduction)

    if x.backprop
        function CRNN_TDC_With_Softmax_Backward()
            if need2computeδ!(x)
                if weight==1.0
                    δ(x) .+= Δ
                else
                    δ(x) .+= Δ .* weight
                end
            end
        end
        push!(graph.backward, CRNN_TDC_With_Softmax_Backward)
    end
    return sum(loglikely)
end
