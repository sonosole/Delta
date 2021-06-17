export edistance

function edistance(tar,src)
    T = length(tar)+1
    S = length(src)+1
    d = zeros(Int, T, S)
    d[1,2:S] = 1:S-1
    d[2:T,1] = 1:T-1
    for t = 2:T
        for s = 2:S
            if src[s-1]==tar[t-1]
                d[t,s] = d[t-1,s-1]
            else
                d[t,s] = 1 + min(d[t-1,s],d[t,s-1],d[t-1,s-1])
                #                ins      del      sub
            end
        end
    end
    return d[T,S]
end
