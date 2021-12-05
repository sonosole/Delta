export CTC
export CTCGreedySearch
export indexbounds

"""
    indexbounds(lengthArray)
`lengthArray` records length of each sequence, i.e. labels or features
# example
    julia> indexbounds([2,0,3,2])
    ([1; 3; 3; 6], [2; 2; 5; 7])
"""
function indexbounds(lengthArray)
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
    CTC(p::Array{T,2}, seq; blank=1) where T -> (target, lossvalue)
# Inputs
    p   : probability of softmax output\n
    seq : label seq like [9 3 6 15] which contains no blank. If p
          has no label (e.g. pure noise or oov) then seq is []
# Outputs
    target    : target of softmax's output\n
    lossvalue : negative log-likelyhood
"""
function CTC(p::Array{TYPE,2}, seq; blank=1) where TYPE
    Log0 = LogZero(TYPE)   # approximate -Inf of TYPE
    ZERO = TYPE(0)         # typed zero,e.g. Float32(0)
    S, T = size(p)         # assert p is a 2-D tensor
    L = length(seq)*2 + 1  # topology length with blanks
    r = fill!(Array{TYPE,2}(undef,S,T), ZERO)    # 𝜸 = p(s[k,t] | x[1:T]), k in softmax's indexing

    if L == 1
        r[blank,:] .= TYPE(1)
        return r, - sum(log.(p[blank,:]))
    end

    a = fill!(Array{TYPE,2}(undef,L,T), Log0)    # 𝜶 = p(s[k,t], x[1:t]), k in CTC topology's indexing
    b = fill!(Array{TYPE,2}(undef,L,T), Log0)    # 𝛃 = p(x[t+1:T] | s[k,t]), k in CTC topology's indexing
    a[1,1] = log(p[blank, 1])
    a[2,1] = log(p[seq[1],1])
    b[L-1,T] = ZERO
    b[L-0,T] = ZERO

    # --- forward in log scale ---
    for t = 2:T
        first = max(1,L-2*(T-t)-1);
        lasst = min(2*t,L);
        for s = first:lasst
            i = div(s,2);
            if s==1
                a[s,t] = a[s,t-1] + log(p[blank,t])
            elseif mod(s,2)==1
                a[s,t] = LogSum2Exp(a[s,t-1], a[s-1,t-1]) + log(p[blank,t])
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
                b[s,t] = b[s,t+1] + log(p[blank,t+1])
            elseif mod(s,2)==1
                b[s,t] = LogSum2Exp(b[s,t+1] + log(p[blank,t+1]), b[s+1,t+1] + log(p[seq[j],t+1]))
            elseif s==L-1
                b[s,t] = LogSum2Exp(b[s,t+1] + log(p[seq[i],t+1]), b[s+1,t+1] + log(p[blank,t+1]))
            elseif seq[i]==seq[i+1]
                b[s,t] = LogSum2Exp(b[s,t+1] + log(p[seq[i],t+1]), b[s+1,t+1] + log(p[blank,t+1]))
            else
                b[s,t] = LogSum3Exp(b[s,t+1] + log(p[seq[i],t+1]), b[s+1,t+1] + log(p[blank,t+1]), b[s+2,t+1] + log(p[seq[i+1],t+1]))
            end
        end
    end

    logsum = LogSum3Exp(Log0, a[1,1] + b[1,1], a[2,1] + b[2,1])
    g = exp.((a + b) .- logsum)

    # reduce first line of g
    r[blank,:] .+= g[1,:]
    # reduce rest lines of g
    for n = 1:length(seq)
        s = n<<1
        r[seq[n],:] .+= g[s,  :]
        r[blank, :] .+= g[s+1,:]
    end

    return r, -logsum
end


"""
    CTCGreedySearch(x::Array; blank=1, dims=1)
remove repeats and blanks of argmax(x, dims=dims)
"""
function CTCGreedySearch(x::Array; blank::Int=1, dims=1)
    hyp = Vector{Int}(undef, 0)
    idx = argmax(x, dims=dims)
    for i = 1:length(idx)
        previous = idx[i≠1 ? i-1 : i][1]
        current  = idx[i][1]
        if !((i≠1 && current==previous) || (current==blank))
            push!(hyp, current)
        end
    end
    return hyp
end
