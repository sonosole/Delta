mutable struct Maxout <: Block
    w::VarOrNil # input to middle hidden weights
    b::VarOrNil # bias of middle hidden units
    h::Int
    k::Int
    function Maxout(inputSize::Int, hiddenSize::Int; k::Int=2, type::Type=Array{Float32})
        @assert (k>=2) "# of Affine layers should no less than 2"
        T = eltype(type)
        d = hiddenSize * k
        w = randn(T, d, inputSize) .* sqrt( T(1/d) )
        b = zeros(T, d, 1)
        new(Variable{type}(w,true,true,true),
            Variable{type}(b,true,true,true), hiddenSize, k)
    end
    function Maxout(hiddenSize::Int; k::Int=2)
        @assert (k>=2) "# of Affine layers should no less than 2"
        new(nothing, nothing, hiddenSize, k)
    end
end


function clone(this::Maxout; type::Type=Array{Float32})
    cloned = Maxout(this.h, k=this.k)
    cloned.w = clone(this.w, type=type)
    cloned.b = clone(this.b, type=type)
    return cloned
end


# pretty show
function Base.show(io::IO, m::Maxout)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    maxk = m.k
    print(io, "Maxout($(SIZE[2]), $(SIZE[1]÷maxk); k=$maxk, type=$TYPE)")
end


"""
    unbiasedof(m::Maxout)

unbiased weights of `Maxout` block
"""
function unbiasedof(m::Maxout)
    weights = Vector(undef, 1)
    weights[1] = m.w.value
    return weights
end


function paramsof(m::Maxout)
    params = Vector{Variable}(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function xparamsof(m::Maxout)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', m.w)
    xparams[2] = ('b', m.b)
    return xparams
end


function nparamsof(m::Maxout)
    lw = length(m.w)
    lb = length(m.b)
    return (lw + lb)
end


function bytesof(model::Maxout, unit::String="MB")
    n = nparamsof(model) * elsizeof(model.w)
    return blocksize(n, uppercase(unit))
end


function forward(model::Maxout, x::Variable{T}) where T
    h = model.h
    k = model.k
    w = model.w
    b = model.b
    c = size(x, 2)
    x = matAddVec(w * x, b)         # dim=(h*k, c)
    temp = reshape(x.value, h,k,c)  # dim=(h,k,c)
    maxv = maximum(temp, dims=2)    # dim=(h,1,c)
    mask = temp .== maxv            # dim=(h,k,c)
    out  = Variable{T}(reshape(maxv, h,c), x.backprop)
    if x.backprop
        function maxoutBackward()
            if need2computeδ!(x)
                x.delta += reshape(mask .* reshape(out.delta, h,1,c), h*k,c)
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, maxoutBackward)
    end
    return out
end


function predict(model::Maxout, x::AbstractArray)
    h = model.h
    k = model.k
    w = model.w.value
    b = model.b.value
    c = size(x, 2)
    x = w * x .+ b                  # dim=(h*k, c)
    temp = reshape(x, h,k,c)        # dim=(h,k,c)
    maxv = maximum(temp, dims=2)    # dim=(h,1,c)
    out  = reshape(maxv, h,c)       # dim=(h,  c)
    return out
end


function to(type::Type, m::Maxout)
    m.w = to(type, m.w)
    m.b = to(type, m.b)
    return m
end


function to!(type::Type, m::Maxout)
    m.w = to(type, m.w)
    m.b = to(type, m.b)
    return nothing
end
