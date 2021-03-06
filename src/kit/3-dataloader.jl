using Random
abstract type DataSet end

export DataSet
export DataLoader

"""
    DataLoader(dataset::T;
               batchsize=1,
               shuffle=true,
               droplast=true,
               collatefn::Union{Function,Nothing}=nothing) where {T<:DataSet}

Suppose each element of the dataset returns a tuple sample (feat, label), everytime the DataLoader instance
fetchs a minibatch of data from a dataset and collates them into a batched sample (feats, labels).
"""
mutable struct DataLoader{T}
    data::T
    batchsize::Int
    batchnums::Int
    droplast::Bool
    shuffle::Bool
    imax::Int
    len::Int
    indices::Vector{Int}
    collate::Union{Function,Nothing}

    function DataLoader(dataset::T;
                        batchsize=1,
                        shuffle=true,
                        droplast=true,
                        collatefn::Union{Function,Nothing}=nothing) where {T<:DataSet}
        if batchsize <= 0
            throw(ArgumentError("batchsize should be positive, but got $batchsize"))
        end
        n = length(dataset)
        if n < batchsize
            @warn "# observations < batchsize, decreasing batchsize to $n"
            batchsize = n
        end
        rest = mod(n, batchsize)
        imax = (droplast && rest!=0) ? (n - rest) : n
        batchnums = ceil(Int, imax / batchsize)
        new{T}(dataset, batchsize, batchnums, droplast, shuffle, imax, n, 1:n, collatefn)
    end
end


function Base.iterate(d::DataLoader, i=0)
    i >= d.imax && return nothing
    if d.shuffle && i == 0
        shuffle!(d.indices)
    end
    next  = min(i + d.batchsize, d.len)
    idxs  = d.indices[i+1 : next]
    batch = [d.data[k] for k in idxs]

    if d.collate ≠ nothing
        return (d.collate(batch), next)
    else
        return (batch, next)
    end
end


function Base.length(d::DataLoader)
    n = d.len / d.batchsize
    d.droplast ? floor(Int, n) : ceil(Int, n)
end


function Base.getindex(d::DataLoader, k::Int)
    k > d.batchnums && return nothing
    if d.shuffle && k == 1
        shuffle!(d.indices)
    end
    start = 1 + (k-1)*d.batchsize
    final = min(k*d.batchsize, d.len)
    batch = [d.data[i] for i in d.indices[start:final]]
    if d.collate ≠ nothing
        return d.collate(batch)
    else
        return batch
    end
end


# pretty printing
function Base.show(io::IO, d::DataLoader{T}) where T
    println("DataLoader{$T}")
    println("————————————————————————")
    println("     data →  $T")
    println("batchsize →  $(d.batchsize)")
    println("batchnums →  $(d.batchnums)")
    println(" droplast →  $(d.droplast)")
    println("  shuffle →  $(d.shuffle)")
    println("     imax →  $(d.imax)")
    println("      len →  $(d.len)")
    println("  indices →  $(d.indices[1]):$(d.indices[2]-d.indices[1]):$(d.indices[end])")
    println("  collate →  $(d.collate)")
    print("————————————————————————")
end
