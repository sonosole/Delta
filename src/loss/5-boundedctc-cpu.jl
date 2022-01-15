export BoundedCTC, seqboundedctc
export BoundedCTCGreedySearch
export BoundedCTCGreedySearchWithTimestamp

"""
    seqboundedctc(seq, blank::Int=1, bound::Int=2) -> newseq

expand `seq` with `blank`, `risebound` and `fallbound`'s indexes. For example, if `seq` is [i,j,k], then
`newseq` is
`[B, ↑,i,↓, B, ↑,j,↓, B, ↑,k,↓, B]`,
of which B is `blank` index, ↑ and ↓ are `bound` index.

# Example
    julia> seqboundedctc([7,9,5])'
    1×13 adjoint(::Vector{Int64}) with eltype Int64:
     1  2  7  2  1  2  9  2  1  2  5  2  1
"""
function seqboundedctc(seq, blank::Int=1, bound::Int=2)
    L = length(seq)       # sequence length
    N = 4 * L + 1         # topology length
    label = zeros(Int, N)
    label[1:4:N] .= blank
    label[2:4:N] .= bound
    label[3:4:N] .= seq
    label[4:4:N] .= bound
    return label
end


"""
    BoundedCTC(p::Array{T,2}, seqlabel; blank::Int=1, bound::Int=2) where T

# Topology Example
      ┌─►─┐                 ┌─►─┐                 ┌─►─┐                 ┌─►─┐                  ┌─►─┐
    ┌─┴───┴─┐  ┌───────┐  ┌─┴───┴─┐  ┌───────┐  ┌─┴───┴─┐  ┌───────┐  ┌─┴───┴─┐  ┌───────┐   ┌─┴───┴─┐
    │ blank ├─►│   ↑   ├─►│ Hello ├─►│   ↓   ├─►│ blank ├─►│   ↑   ├─►│ World ├─►│   ↓   ├──►│ blank │
    └───────┘  └───────┘  └───┬───┘  └───┬───┘  └───────┘  └─┬───┬─┘  └───────┘  └───────┘   └───────┘
                              │          └──────────►────────┘   │
                              └─────────────────────►────────────┘

"""
function BoundedCTC(p::Array{TYPE,2}, seqlabel; blank::Int=1, bound::Int=2) where TYPE
    seq  = seqboundedctc(seqlabel, blank, bound)
    Log0 = LogZero(TYPE)   # approximate -Inf of TYPE
    ZERO = TYPE(0.0)       # typed zero, e.g. Float32(0)
    S, T = size(p)         # assert p is a 2-D tensor
    L = length(seq)        # topology length
    r = zero(p)            # 𝜸 = p(s[k,t] | x[1:T]), k in softmax's indexing

    if L == 1
        r[seq[1],:] .= TYPE(1)
        return r, - sum(log.(p[seq[1],:]))
    end

    a = fill!(Array{TYPE,2}(undef,L,T), Log0)
    b = fill!(Array{TYPE,2}(undef,L,T), Log0)
	a[1,1] = log(p[seq[1],1])
	a[2,1] = log(p[seq[2],1])
    a[3,1] = log(p[seq[3],1])
    b[L-2,T] = ZERO
	b[L-1,T] = ZERO
	b[L  ,T] = ZERO

    # --- forward in log scale ---
	for t = 2:T
        τ = t-1
	    for s = 1:L
            R = mod(s,4)
	        if s≠1 && s≠2 && R≠0
	            if R==3 || R==1
	                a[s,t] = LogSum2Exp(a[s,τ], a[s-1,τ])
	            elseif R==2
	                a[s,t] = LogSum3Exp(a[s-1,τ], a[s-2,τ], a[s-3,τ])
	            end
	        elseif s≠1
	            a[s,t] = a[s-1,τ]
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
				if R==2 || s==L-1
					b[s,t] = Q¹
				elseif R==1 || s==L-2
					b[s,t] = LogSum2Exp(Q⁰, Q¹)
                elseif R==0
                    Q² = b[s+2,τ] + log(p[seq[s+2],τ])
                    b[s,t] = LogSum2Exp(Q¹, Q²)
                elseif R==3
                    Q³ = b[s+3,τ] + log(p[seq[s+3],τ])
                    b[s,t] = LogSum3Exp(Q⁰, Q¹, Q³)
				end
			else
				b[s,t] = Q⁰
			end
		end
	end

    # loglikely of TCS
    logsum = LogSum3Exp(a[1,1] + b[1,1], a[2,1] + b[2,1], a[3,1] + b[3,1])

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


"""
    BoundedCTCGreedySearch(x::Array; blank::Int=1, bound::Int=2) -> hypothesis
"""
function BoundedCTCGreedySearch(x::Array; blank::Int=1, bound::Int=2)
    hyp = Vector{Int}(undef, 0)
    idx = argmax(x,dims=1)
    for t = 1:length(idx)
        previous = idx[t≠1 ? t-1 : t][1]
        current  = idx[t][1]
        if !((current==previous && t≠1) ||
             (current==blank) ||
             (current==bound))
            push!(hyp, current)
        end
    end
    return hyp
end


"""
    BoundedCTCGreedySearchWithTimestamp(x::Array; blank::Int=1, bound::Int=2) -> hypothesis, timestamp
"""
function BoundedCTCGreedySearchWithTimestamp(x::Array; blank::Int=1, bound::Int=2)
    hyp = Vector{Int}(undef, 0)
    stp = Vector{Float32}(undef, 0)
    idx = argmax(x,dims=1)
    T   = length(idx)
    for t = 1:T
        previous = idx[t≠1 ? t-1 : t][1]
        current  = idx[t][1]
        if !((current==previous && t≠1) ||
             (current==blank) ||
             (current==bound))
            push!(hyp, current)
            push!(stp, t / T)
        end
    end
    return hyp, stp
end
