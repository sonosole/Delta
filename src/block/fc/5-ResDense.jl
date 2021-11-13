mutable struct ResDense <: Block
    blocks::Vector
    function ResDense(i::Int, n::Int, m::Int; type::Type=Array{Float32})
        l1 = linear(i,n, type=type)
        l2 = linear(n,m, type=type)
        l3 = linear(m,i, type=type)
        new([l1, l2, l3])
    end
    function ResDense()
        new(Vector(undef,3))
    end
end

function Base.show(io::IO, c::ResDense)
    print(io,  "ResDense\n(\n      ")
    join(io, c.blocks, "\n      ")
    print(io,               "\n)")
end

Base.length(c::ResDense)     = 3
Base.lastindex(c::ResDense)  = 3
Base.firstindex(c::ResDense) = 1
Base.getindex(c::ResDense, k...)     =  c.blocks[k...]
Base.setindex!(c::ResDense, v, k...) = (c.blocks[k...] = v)
Base.iterate(c::ResDense, i=1) = i>3 ? nothing : (c[i], i+1)

function paramsof(m::ResDense)
    params = Vector{Variable}(undef,0)
    for i = 1:length(m)
        append!(params, paramsof(m[i]))
    end
    return params
end

function xparamsof(m::ResDense)
    xparams = Vector{XVariable}(undef,0)
    for i = 1:length(m)
        append!(xparams, xparamsof(m[i]))
    end
    return xparams
end

function nparamsof(model::ResDense)
    nparams = 0
    for m in model
        nparams += nparamsof(m)
    end
    return nparams
end

function bytesof(model::ResDense, unit::String="MB")
    n = nparamsof(model)
    u = uppercase(unit)
    if u == "KB" return n * sizeof(eltype(model[1].w)) / 1024 end
    if u == "MB" return n * sizeof(eltype(model[1].w)) / 1048576 end
    if u == "GB" return n * sizeof(eltype(model[1].w)) / 1073741824 end
    if u == "TB" return n * sizeof(eltype(model[1].w)) / 1099511627776 end
end

function clone(this::ResDense; type::Type=Array{Float32})
    cloned = ResDense()
    cloned[1] = clone(this[1], type=type)
    cloned[2] = clone(this[2], type=type)
    cloned[3] = clone(this[3], type=type)
    return cloned
end

function forward(m::ResDense, x0)
    x1 = relu(forward(m[1], x0))
    x2 = relu(forward(m[2], x1))
    x3 = x0 + forward(m[3], x2)
    return relu(x3)
end

function predict(m::ResDense, x0)
    x1 = relu(predict(m[1], x0))
    x2 = relu(predict(m[2], x1))
    x3 = x0 + predict(m[3], x2)
    return relu(x3)
end
