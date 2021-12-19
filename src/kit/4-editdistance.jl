export editcost
export editcosts
export editerrs
export alignops
export alignerrs


"""
    editcost(tar, src, subcost=1)

Minimum edit distance discribes how many operations to transform `src` into `tar`.
Editing operations are `Insertion(i)`, `Deletion(d)` and `Substitution(s)`, e.g.\n
    src: I N T E * N T I O N
    tar: * E X E C U T I O N
         d s s   i s          (Note: the transformation direction is src -> tar)
`subcost`=2 is so called Levenshtein distance. `subcost`=1 is often used in speech recognition tasks.
"""
function editcost(tar::V, src::V, subcost=1) where V
    T = length(tar)+1
    S = length(src)+1
    d = zeros(Int, T, S)
    d[1,2:S] = 1:S-1
    d[2:T,1] = 1:T-1
    for t = 2:T
        for s = 2:S
            c = ifelse(src[s-1]!=tar[t-1], subcost, 0)
            d[t,s] = min(d[t-1,s]+1, d[t,s-1]+1, d[t-1,s-1]+c)
            # -----------ins---------del---------sub----------
        end
    end
    return d[T,S]
end


"""
    editcosts(tar, src, subcost=1) -> d, ins,del,sub

`d` = `ins` + `del` + `sub`, where `d` is the total edit distance from `src` to `tar`,
`ins` is insertion operations, `del` is deletion operations and `sub` is substitution
operations. `subcost`=2 is so called Levenshtein distance. e.g.\n
    src: I N T E * N T I O N
    tar: * E X E C U T I O N
         d s s   i s          (Note: the transformation direction is src -> tar)

# Example
    julia> d,ins,del,sub = editcosts("ABCDEfghi","abcABCdE")
    (8, 4, 3, 1)
"""
function editcosts(tar::V, src::V, subcost=1) where V
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
            #-------ins---------del---------sub----------
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
    editerrs(correct, estimated, subcost=1) -> d, ins,del,sub

`d` = `ins` + `del` + `sub`, where `d` is the total edit errors made from editing
`correct` to `estimated`, `ins` is insertion(i) errors, `del` is deletion(d) errors
and `sub` is substitution(s) errors, e.g.\n
    correct   : A * C D
    estimated : a b C
                s i   d  (Note: the edit direction is correct -> estimated)
`subcost`=1 is mostly used in speech transcription task to calculate word error rate.
`subcost`=2 is called Levenshtein distance.
# Example
    julia> d,ins,del,sub = editerrs("ABCDEfghi","abcABCdE")
    (8, 3, 4, 1)
    julia> d,ins,del,sub = editcosts("ABCDEfghi","abcABCdE")
    (8, 4, 3, 1)
"""
function editerrs(correct::V, estimated::V, subcost=1) where V
    d, ins,del,sub = editcosts(estimated, correct, subcost)
    return d, ins,del,sub
end


"""
    alignops(tar, src; subcost=1, show=true, cn=false) -> marks::Array{String,2}

align `src` and `tar`, `subcost` is the substitution cost, `show` decides whether
to display, `cn` stands for chinese. `marks` is the aligned results recorded in a
3×N Array{String,2}, of which the 1st row is `tar`, 2nd is `src` and 3rd is the
operations to transform `src` into `tar`. `subcost`=2 is Levenshtein distance.

# Example
    julia> r = alignops(
           ["are","are","you","ok","bu"], # tar
           ["ah","r","are","u","OKAY"]);  # src
    ───────────────────
    are     ah      sub
            r       del
    are     are
    you     u       sub
    ok      OKAY    sub
    bu              ins
"""
function alignops(tar::V, src::V; subcost=1, show=true, cn=false) where V
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
            #-------ins---------del---------sub----------
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
            show && println(TAR, _V_, SRC, _V_, "ins"|>blue!) # Insertion
        elseif d[t,s]==d[t,𝕤]+1
            marks[1,𝕚] = ""
            marks[2,𝕚] = src[𝕤]
            marks[3,𝕚] = DEL
            TAR = repeat(" ", Tmax)
            SRC = src[𝕤] * repeat(" ", Smax - length(src[𝕤]))
            show && println(TAR, _V_, SRC, _V_, "del"|>red!) # Deletion
        elseif d[t,s]==d[r,c]+subcost
            marks[1,𝕚] = tar[𝕥]
            marks[2,𝕚] = src[𝕤]
            marks[3,𝕚] = SUB
            TAR = tar[𝕥] * repeat(" ", Tmax - Tlen[𝕥])
            SRC = src[𝕤] * repeat(" ", Smax - Slen[𝕤])
            show && println(TAR, _V_, SRC, _V_, "sub"|>yellow!) # Substitution
        end
    end
    return marks
end


"""
    alignerrs(tar, src; subcost=1, show=true, cn=false) -> marks::Array{String,2}

align `src`(estimated) and `tar`(desired), `subcost` is the substitution cost, `show`
decides whether to display, `cn` stands for chinese. `marks` is the aligned results
recorded in a 3×N Array{String,2}, of which the 1st row is `tar`, the 2nd is `src` and
the 3rd is the errors to align `src` to `tar`. `subcost`=2 is Levenshtein distance.

# Example
    julia> r = alignerrs(
           ["are","are","you","ok","bu"],
           ["ah" ,"r"  ,"are","u" ,"OKAY"]);
    ───────────────────
    are     ah      sub
            r       ins
    are     are
    you     u       sub
    ok      OKAY    sub
    bu              del
"""
function alignerrs(tar::V, src::V; subcost=1, show=true, cn=false) where V
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
            #-------ins---------del---------sub----------
            indx = argmin(way3)
            d[t,s] = way3[indx]
            if indx==1
                row[t,s], col[t,s] = t-1, s   # ins op on src
            elseif indx==2
                row[t,s], col[t,s] = t, s-1   # del op on src
            elseif indx==3
                row[t,s], col[t,s] = t-1, s-1 # sub op on src
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

    # ops on src to transfer into tar, can be viewed as the
    # error edits that leads tar turning into src, so ins/del op
    # from src to tar are equivalent to del/ins op from tar to src
    steps = length(trace)-2
    marks = Array{String,2}(undef, 3, steps)
    INS = cn ? "删" : "del"
    DEL = cn ? "插" : "ins"
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
            show && println(TAR, _V_, SRC, _V_, "del"|>red!) # Deletion errors
        elseif d[t,s]==d[t,𝕤]+1
            marks[1,𝕚] = ""
            marks[2,𝕚] = src[𝕤]
            marks[3,𝕚] = DEL
            TAR = repeat(" ", Tmax)
            SRC = src[𝕤] * repeat(" ", Smax - length(src[𝕤]))
            show && println(TAR, _V_, SRC, _V_, "ins"|>blue!) # Insertion errors
        elseif d[t,s]==d[r,c]+subcost
            marks[1,𝕚] = tar[𝕥]
            marks[2,𝕚] = src[𝕤]
            marks[3,𝕚] = SUB
            TAR = tar[𝕥] * repeat(" ", Tmax - Tlen[𝕥])
            SRC = src[𝕤] * repeat(" ", Smax - Slen[𝕤])
            show && println(TAR, _V_, SRC, _V_, "sub"|>yellow!) # Substitution errors
        end
    end
    return marks
end
