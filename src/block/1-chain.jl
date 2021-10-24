export Chain
export popitems

"""
    Chain(blocks...)

Chain multiple blocks / functions together, so that they are called in sequence on a given input.
`Chain` also supports indexing and slicing, e.g. `c[2]` or `c[1:3]` or c[[1,3]].

# Examples
    julia> m = Chain(dense(256,128), lstm(128,64), dense(64,10));
"""
mutable struct Chain
    blocks::Vector
    function Chain(sequence::Vector)
        blocknum = length(sequence)
        blocks = Vector(undef,blocknum)
        for i = 1:blocknum
            blocks[i] = sequence[i]
        end
        new(blocks)
    end
    function Chain(sequence...)
        blocknum = length(sequence)
        blocks = Vector(undef,blocknum)
        for i = 1:blocknum
            blocks[i] = sequence[i]
        end
        new(blocks)
    end
end


Base.length(c::Chain)     = length(c.blocks)
Base.lastindex(c::Chain)  = length(c)
Base.firstindex(c::Chain) = 1
Base.getindex(c::Chain, k...)      = c.blocks[k...]
Base.setindex!(c::Chain, v, k...) = (c.blocks[k...] = v)
Base.iterate(c::Chain, i=firstindex(c)) = i>length(c) ? nothing : (c[i], i+1)


function Base.show(io::IO, c::Chain)
    print(io,  "Chain\n(\n      ")
    join(io, c.blocks, "\n      ")
    print(io,               "\n)")
end


function popitems(blocks::Vector{T}, list) where T
    lenb = length(blocks)
    lenl = length(list)
    @assert(lenb>lenl)
    newblocks = Vector(undef,0)
    for i = 1:lenb
        if i ∉ list
            push!(newblocks,blocks[i])
        end
    end
    return newblocks
end


"""
    unbiasedof(model::Chain)

unbiased weights of Chain block
"""
function unbiasedof(model::Chain)
    weights = Vector(undef, 0)
    for m in model
        append!(weights, unbiasedof(m))
    end
    return weights
end


function paramsof(c::Chain)
    params = Vector{Variable}(undef,0)
    for i = 1:length(c)
        p = paramsof(c[i])
        if p ≠ nothing
            append!(params, p)
        end
    end
    return params
end


function xparamsof(c::Chain)
    params = Vector{XVariable}(undef,0)
    for i = 1:length(c)
        p = xparamsof(c[i])
        if p ≠ nothing
            append!(params, p)
        end
    end
    return params
end


function nparamsof(c::Chain)
    k = 0
    for i = 1:length(c)
        k += nparamsof(c[i])
    end
    return k
end


function resethidden(c::Chain)
    for i = 1:length(c)
        if typeof(c[i]) ∈ RNNLIST
            resethidden(c[i])
        end
    end
end


function forward(c::Chain, x::Variable)
    for i = 1:length(c)
        x = forward(c[i], x)
    end
    return x
end


function predict(c::Chain, x)
    for i = 1:length(c)
        x = predict(c[i], x)
    end
    return x
end


function to(type::Type, c::Chain)
    for m in c
        m = to(type, m)
    end
    return c
end


function to!(type::Type, c::Chain)
    for m in c
        m = to(type, m)
    end
    return nothing
end
