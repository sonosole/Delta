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


"""
    align2seq(tar, src, subcost=1, show=false; cn=false) -> marks::Array{String,2}

align `src` and `tar`, `subcost` is the substitution cost, `show` decides whether\n
to display on the terminal, `cn` stands for chinese. `marks` is the aligned results\n
recorded in a 3×N Array{String,2}, of which the 1st row is `tar`, 2nd is `src` and\n
3rd is edit operations.

# Example
julia> r = align2seq(

       ["are","are","you","ok","bu"],

       ["ah","r","are","u","OKAY"], 1, true; cn=false);
───────────────────
are     ah      sub
        r       del
are     are
you     u       sub
ok      OKAY    sub
bu              ins
"""
function align2seq(tar, src, subcost=1, show=false; cn=false)
    T = length(tar) + 1
    S = length(src) + 1
    d =   zeros(Int, T, S)
    row = zeros(Int, T, S)
    col = zeros(Int, T, S)

    Tmax = 0
    Smax = 0
    Tlen = zeros(Int, T-1)
    Slen = zeros(Int, S-1)
    for t=1, s=2:S
        𝕤 = s - 1
        d[t,s] = 𝕤
        row[t,s] = t
        col[t,s] = 𝕤
        Slen[𝕤] = length(src[𝕤])
        if Smax < Slen[𝕤] Smax=Slen[𝕤] end
    end
    for t=2:T, s=1
        𝕥 = t - 1
        d[t,s] = 𝕥
        row[t,s] = 𝕥
        col[t,s] = s
        Tlen[𝕥] = length(tar[𝕥])
        if Tmax < Tlen[𝕥] Tmax=Tlen[𝕥] end
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

    steps = length(trace)-2
    marks = Array{String,2}(undef, 3, steps)
    INS = cn ? "插" : "ins"
    DEL = cn ? "删" : "del"
    SUB = cn ? "替" : "sub"
    NIL = cn ? "无" : "nil"
    _V_ = "\t"

    cn ? println(repeat("─",floor(Int,(Tmax+Smax+12)*3/2))) : println(repeat("─",Tmax+Smax+12))
    for (i,(t,s)) in enumerate(trace[end-1:-1:1])
        r, c = row[t,s], col[t,s]
        if r==0 || c==0 continue end
        𝕥 = t!=1 ? t-1 : t  # edge case
        𝕤 = s!=1 ? s-1 : s  # edge case
        𝕚 = i - 1
        if d[t,s]==d[r,c]
            marks[1,𝕚] = tar[𝕥]
            marks[2,𝕚] = src[𝕤]
            marks[3,𝕚] = NIL
            TAR = tar[𝕥] * repeat(" ", Tmax - Tlen[𝕥])
            SRC = src[𝕤] * repeat(" ", Smax - Slen[𝕤])
            show && println(TAR, _V_, SRC) # Correct
        elseif d[t,s]==d[𝕥,s]+1
            marks[1,𝕚] = tar[𝕥]
            marks[2,𝕚] = ""
            marks[3,𝕚] = INS
            TAR = tar[𝕥] * repeat(" ", Tmax - Tlen[𝕥])
            SRC = repeat(" ", Smax)
            show && println(TAR, _V_, SRC, _V_, "ins"|>blue) # Insertion
        elseif d[t,s]==d[t,𝕤]+1
            marks[1,𝕚] = ""
            marks[2,𝕚] = src[𝕤]
            marks[3,𝕚] = DEL
            TAR = repeat(" ", Tmax)
            SRC = src[𝕤] * repeat(" ", Smax - length(src[𝕤]))
            show && println(TAR, _V_, SRC, _V_, "del"|>red) # Deletion
        elseif d[t,s]==d[r,c]+subcost
            marks[1,𝕚] = tar[𝕥]
            marks[2,𝕚] = src[𝕤]
            marks[3,𝕚] = SUB
            TAR = tar[𝕥] * repeat(" ", Tmax - Tlen[𝕥])
            SRC = src[𝕤] * repeat(" ", Smax - Slen[𝕤])
            show && println(TAR, _V_, SRC, _V_, "sub"|>yellow) # Substitution
        end
    end
    return marks
end
