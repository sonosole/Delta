export BoundedCTC, seqbctc
export BoundedCTCGreedySearch
export BoundedCTCGreedySearchWithTimestamp

"""
    seqbctc(seq, blank::Int=1, risebound::Int=2, fallbound::Int=3) -> newseq
expand `seq` with `blank`, `risebound` and `fallbound`'s indexes. For example, if `seq` is [i,j,k], then
`newseq` is
`[B, ↑,i,↓, B, ↑,j,↓, B, ↑,k,↓, B]`,
of which B is `blank` index, ↑ is `risebound` index and ↓ is `fallbound` index.

# Example
    julia> seqbctc([7,9,5])'
    1×13 adjoint(::Vector{Int64}) with eltype Int64:
     1  2  7  3  1  2  9  3  1  2  5  3  1
"""
function seqbctc(seq, blank::Int=1, risebound::Int=2, fallbound::Int=3)
    L = length(seq)       # sequence length
    N = 4 * L + 1         # topology length
    label = zeros(Int, N)
    label[1:4:N] .= blank
    label[2:4:N] .= risebound
    label[3:4:N] .= seq
    label[4:4:N] .= fallbound
    return label
end


function BoundedCTC(p::Array{TYPE,2}, seqlabel; blank::Int=1, risebound::Int=2, fallbound::Int=3) where TYPE
    seq  = seqbctc(seqlabel, blank, risebound, fallbound)
    Log0 = LogZero(TYPE)   # approximate -Inf of TYPE
    ZERO = TYPE(0.0)       # typed zero, e.g. Float32(0)
    S, T = size(p)         # assert p is a 2-D tensor
    L = length(seq)        # topology length
    a = fill(Log0, L,T)    # 𝜶 = p(s[k,t], x[1:t]), k in TCS topology's indexing
    b = fill(Log0, L,T)    # 𝛃 = p(x[t+1:T] | s[k,t]), k in TCS topology's indexing
    r = zero(p)            # 𝜸 = p(s[k,t] | x[1:T]), k in softmax's indexing

	if L>1
		a[1,1] = log(p[seq[1],1])
		a[2,1] = log(p[seq[2],1])
		b[L-1,T] = ZERO
		b[L  ,T] = ZERO
	else
		a[1,1] = log(p[seq[1],1])
		b[L,T] = ZERO
	end

    # --- forward in log scale ---
	for t = 2:T
        τ = t-1
	    for s = 1:L
	        if s≠1
				R = mod(s,4)
	            if (R==3 || R==0 || R==1) || s==2
	                a[s,t] = LogSum2Exp(a[s,τ], a[s-1,τ])
	            elseif R==2
	                a[s,t] = LogSum3Exp(a[s,τ], a[s-1,τ], a[s-2,τ])
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
				if (R==1 || R==2 || R==3) || s==L-1
					b[s,t] = LogSum2Exp(Q⁰, Q¹)
				elseif R==0
                    Q² = b[s+2,τ] + log(p[seq[s+2],τ])
					b[s,t] = LogSum3Exp(Q⁰, Q¹, Q²)
				end
			else
				b[s,t] = Q⁰
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
    for n = 1:div(L-1,4)
        s = n<<2
        r[seq[s-2],:] .+= g[s-2,:]  # reduce blank states
        r[seq[s-1],:] .+= g[s-1,:]  # reduce risebund states
        r[seq[s  ],:] .+= g[s,  :]  # reduce labels' states
        r[seq[s+1],:] .+= g[s+1,:]  # reduce fallbound state
    end

    return r, -logsum
end


function BoundedCTCGreedySearch(x::Array, blank::Int=1, risebound::Int=2, fallbound::Int=3)
    hyp = Vector{Int}(undef, 0)
    idx = argmax(x,dims=1)
    for t = 1:length(idx)
        previous = idx[t≠1 ? t-1 : t][1]
        current  = idx[t][1]
        if !((current==previous && t≠1) ||
             (current==blank) ||
             (current==risebound) ||
             (current==fallbound))
            push!(hyp, current)
        end
    end
    return hyp
end


"""
    BoundedCTCGreedySearchWithTimestamp(x::Array, blank::Int=1, risebound::Int=2, fallbound::Int=3) -> hypothesis, timestamp
"""
function BoundedCTCGreedySearchWithTimestamp(x::Array, blank::Int=1, risebound::Int=2, fallbound::Int=3)
    hyp = Vector{Int}(undef, 0)
    stp = Vector{Float32}(undef, 0)
    idx = argmax(x,dims=1)
    T   = length(idx)
    for t = 1:T
        previous = idx[t≠1 ? t-1 : t][1]
        current  = idx[t][1]
        if !((current==previous && t≠1) ||
             (current==blank) ||
             (current==risebound) ||
             (current==fallbound))
            push!(hyp, current)
            push!(stp, t / T)
        end
    end
    return hyp, stp
end
