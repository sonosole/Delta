export FastCTC, seqfastctc
export FastCTCGreedySearch
export FastCTCGreedySearchWithTimestamp
export CRNN_FastCTC_With_Softmax


function seqfastctc(seq, blank::Int=1)
    L = length(seq)       # sequence length
    N = 2 * L + 1         # topology length
    label = zeros(Int, N)
    label[1:2:N] .= blank
    label[2:2:N] .= seq
    return label
end


"""
    FastCTC(p::Array{T,2}, seqlabel; blank::Int=1) where T

# Topology Example
     ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐
    ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐
    │blank├─►│  C  ├─►│blank├─►│  A  ├─►│blank├─►│  T  ├─►│blank│
    └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘
"""
function FastCTC(p::Array{TYPE,2}, seqlabel; blank::Int=1) where TYPE
    seq  = seqfastctc(seqlabel, blank)
    Log0 = LogZero(TYPE)   # approximate -Inf of TYPE
    ZERO = TYPE(0)         # typed zero,e.g. Float32(0)
    S, T = size(p)         # assert p is a 2-D tensor
    L = length(seq)        # topology length with blanks
    r = fill!(Array{TYPE,2}(undef,S,T), ZERO)    # 𝜸 = p(s[k,t] | x[1:T]), k in softmax's indexing

    if L == 1
        r[blank,:] .= TYPE(1)
        return r, - sum(log.(p[blank,:]))
    end

    a = fill!(Array{TYPE,2}(undef,L,T), Log0)    # 𝜶 = p(s[k,t], x[1:t]), k in FastCTC topology's indexing
    b = fill!(Array{TYPE,2}(undef,L,T), Log0)    # 𝛃 = p(x[t+1:T] | s[k,t]), k in FastCTC topology's indexing
    a[1,1] = log(p[seq[1],1])
    a[2,1] = log(p[seq[2],1])
    b[L-1,T] = ZERO
    b[L  ,T] = ZERO

    # --- forward in log scale ---
    for t = 2:T
        τ = t-1
        first = max(1, L-2*(T-t)-1)
        lasst = min(2*t, L)
        for s = first:lasst
            if s≠1
                a[s,t] = LogSum2Exp(a[s,τ], a[s-1,τ])
            else
                a[s,t] = a[s,τ]
            end
            a[s,t] += log(p[seq[s],t])
        end
    end

    # --- backward in log scale ---
    for t = T-1:-1:1
        τ = t+1
        first = max(1, L-2*(T-t)-1)
        lasst = min(2*t, L)
        for s = first:lasst
            Q = b[s,τ] + log(p[seq[s],τ])
            if s≠L
                b[s,t] = LogSum2Exp(Q, b[s+1,τ] + log(p[seq[s+1],τ]))
            else
                b[s,t] = Q
            end
        end
    end

    logsum = LogSum2Exp(a[1,1] + b[1,1], a[2,1] + b[2,1])
    g = exp.((a + b) .- logsum)

    # reduce first line of g
    r[blank,:] .+= g[1,:]
    # reduce rest lines of g
    for n = 1:div(L-1,2)
        s = n<<1
        r[seq[s],:] .+= g[s,  :]
        r[blank, :] .+= g[s+1,:]
    end

    return r, -logsum
end


"""
    FastCTCGreedySearch(x::Array; blank::Int=1, dim::Int=1) -> hypothesis
remove repeats and blanks of argmax(x, dims=dims)
"""
function FastCTCGreedySearch(x::Array; blank::Int=1, dim::Int=1)
    hyp = Vector{Int}(undef, 0)
    idx = argmax(x, dims=dim)
    for t = 1:length(idx)
        previous = idx[t≠1 ? t-1 : t][1]
        current  = idx[t][1]
        if !((t≠1 && current==previous) || (current==blank))
            push!(hyp, current)
        end
    end
    return hyp
end


"""
    FastCTCGreedySearchWithTimestamp(x::Array; blank::Int=1, dim::Int=1) -> hypothesis, timestamp
"""
function FastCTCGreedySearchWithTimestamp(x::Array; blank::Int=1, dim::Int=1)
    hyp = Vector{Int}(undef, 0)
    stp = Vector{Float32}(undef, 0)
    idx = argmax(x, dims=dim)
    T   = length(idx)
    for t = 1:T
        previous = idx[t≠1 ? t-1 : t][1]
        current  = idx[t][1]
        if !((t≠1 && current==previous) || (current==blank))
            push!(hyp, current)
            push!(stp, t / T)
        end
    end
    return hyp, stp
end



function CRNN_FastCTC_With_Softmax(x::Variable{Array{T}},
                                   seqlabels::Vector;
                                   blank::Int=1,
                                   weight::Float64=1.0,
                                   reduction::String="seqlen") where T
    featdims, timesteps, batchsize = size(x)
    loglikely = zeros(T, batchsize)
    p = softmax(ᵛ(x); dims=1)
    r = zero(ᵛ(x))

    Threads.@threads for b = 1:batchsize
        r[:,:,b], loglikely[b] = FastCTC(p[:,:,b], seqlabels[b], blank=blank)
    end

    Δ = p - r
    reduce3d(Δ, loglikely, seqlabels, reduction)

    if x.backprop
        function CRNN_FastCTC_With_Softmax_Backward()
            if need2computeδ!(x)
                if weight==1.0
                    δ(x) .+= Δ
                else
                    δ(x) .+= Δ .* weight
                end
            end
        end
        push!(graph.backward, CRNN_FastCTC_With_Softmax_Backward)
    end
    return sum(loglikely)
end
