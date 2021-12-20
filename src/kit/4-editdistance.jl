export editcost
export editcosts


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
            # -----------ins --------del---------sub----------
        end
    end
    return d[T,S]
end


"""
    editcosts(tar, src, subcost=1) -> d, ins,del,sub

`d` = `ins` + `del` + `sub`, where `d` is total edit distance from `src` to `tar`,
`ins` is insertion operations, `del` is deletion operations and `sub` is substitution
operations

# Example
    julia> d,ins,del,sub = editcosts("ABCDEfghi","abcABCdE")
    (8, 4, 3, 1)
"""
function editcosts(tar, src, subcost=1)
    T = length(tar)+1
    S = length(src)+1
    d = zeros(Int, T, S)
    row = zeros(Int, T, S)
    col = zeros(Int, T, S)

    # if tar=[], Deletions on src
    for t=1, s=2:S
        d[t,s] = s - 1
        row[t,s] = t
        col[t,s] = s - 1
    end
    # if src=[], Insertions on src
    for t=2:T, s=1
        d[t,s] = t - 1
        row[t,s] = t - 1
        col[t,s] = s
    end

    # forward
    for t = 2:T
        for s = 2:S
            c = ifelse(src[s-1]!=tar[t-1], subcost, 0)
            way3 = [d[t-1,s]+1, d[t,s-1]+1, d[t-1,s-1]+c]
            # ------ins --------del---------sub----------
            indx = argmin(way3)
            d[t,s] = way3[indx]
            if indx==1
                row[t,s], col[t,s] = t-1, s   # ins
            elseif indx==2
                row[t,s], col[t,s] = t, s-1   # del
            elseif indx==3
                row[t,s], col[t,s] = t-1, s-1 # sub
            end
        end
    end

    # trace-back
    trace = [];
    push!(trace, (T, S))
    push!(trace, (row[T,S], col[T,S]))
    while trace[end] != (0,0)
        t,s = trace[end]
        push!(trace, (row[t,s], col[t,s]))
    end

    ins, del, sub = 0, 0, 0
    for (t,s) in trace[end-1:-1:1]
        r, c = row[t,s], col[t,s]
        if r==0 || c==0 continue end
        𝕥 = t!=1 ? t-1 : t  # edge case
        𝕤 = s!=1 ? s-1 : s  # edge case
        if d[t,s]==d[r,c]
            continue
        elseif d[t,s]==d[𝕥,s]+1
            ins += 1
        elseif d[t,s]==d[t,𝕤]+1
            del += 1
        elseif d[t,s]==d[r,c]+subcost
            sub += 1
        end
    end
    return d[T,S], ins,del,sub
end
