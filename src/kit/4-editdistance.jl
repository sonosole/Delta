export editcost


"""
    editcost(tar, src, subcost=1)

Minimum edit distance discribes how many steps to transform `src` into `tar`.
Editing operations are `Insertion(i)`, `Deletion(d)` and `Substitution(s)`, e.g.\n
    src: I N T E * N T I O N
    tar: * E X E C U T I O N
         d s s   i s
"""
function editcost(tar, src, subcost=1)
    T = length(tar)+1
    S = length(src)+1
    d = zeros(Int, T, S)
    d[1,2:S] = 1:S-1
    d[2:T,1] = 1:T-1
    for t = 2:T
        for s = 2:S
            c = ifelse(src[s-1]!=tar[t-1], subcost, 0)
            d[t,s] = min(d[t-1,s]+1, d[t,s-1]+1, d[t-1,s-1]+c)
            #               ins      del         sub
        end
    end
    return d[T,S]
end
