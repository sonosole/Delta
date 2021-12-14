"""
Vanilla RNN, i.e. ⤦\n
    hᵗ = f(w*xᵗ + u*hᵗ⁻¹ .+ b)
"""
mutable struct RNN <: Block
    w::VarOrNil # input to hidden weights
    b::VarOrNil # bias of hidden units
    u::VarOrNil # recurrent weights
    f::Function # activation function
    h::Any      # hidden variable
    function RNN(inputSize::Int, hiddenSize::Int, fn::Function=relu; type::Type=Array{Float32})
        T = eltype(type)
        w = randn(T, hiddenSize, inputSize) .* sqrt( T(2/inputSize) )
        b = zeros(T, hiddenSize, 1)
        u = randdiagonal(T, hiddenSize; from=-0.1990, to=0.1997)
        new(Variable{type}(w,true,true,true),
            Variable{type}(b,true,true,true),
            Variable{type}(u,true,true,true), fn, nothing)
    end
    function RNN(fn::Function)
        new(nothing, nothing, nothing, fn, nothing)
    end
end


function clone(this::RNN; type::Type=Array{Float32})
    cloned = RNN(this.f)
    cloned.w = clone(this.w, type=type)
    cloned.b = clone(this.b, type=type)
    cloned.u = clone(this.u, type=type)
    return cloned
end


mutable struct RNNs <: Block
    layers::Vector{RNN}
    function RNNs(topology::Vector{Int}, fn::Array{F}; type::Type=Array{Float32}) where F
        n = length(topology) - 1
        layers = Vector{RNN}(undef, n)
        for i = 1:n
            layers[i] = RNN(topology[i], topology[i+1], fn[i]; type=type)
        end
        new(layers)
    end
end


Base.getindex(m::RNNs,     k...) =  m.layers[k...]
Base.setindex!(m::RNNs, v, k...) = (m.layers[k...] = v)
Base.length(m::RNNs)       = length(m.layers)
Base.lastindex(m::RNNs)    = length(m.layers)
Base.firstindex(m::RNNs)   = 1
Base.iterate(m::RNNs, i=firstindex(m)) = i>length(m) ? nothing : (m[i], i+1)


function Base.show(io::IO, m::RNN)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    print(io, "RNN($(SIZE[2]), $(SIZE[1]), $(m.f); type=$TYPE)")
end


function Base.show(io::IO, m::RNNs)
    print(io, "RNNs\n      (\n          ")
    join(io, m.layers, ",\n          ")
    print(io, "\n      )")
end


function resethidden(m::RNN)
    m.h = nothing
end


function resethidden(model::RNNs)
    for m in model
        resethidden(m)
    end
end


function forward(m::RNN, x::Variable{T}) where T
    f = m.f  # activition function
    w = m.w  # input's weights
    b = m.b  # input's bias
    u = m.u  # memory's weights
    h = m.h ≠ nothing ? m.h : Variable{T}(Zeros(T, size(w,1), size(x,2)))
    x = f(matAddVec(w*x + u*h, b))
    m.h = x
    return x
end


function forward(model::RNNs, x::Variable)
    for m in model
        x = forward(m, x)
    end
    return x
end


function predict(m::RNN, x::T) where T
    f = m.f        # activition function
    w = m.w.value  # input's weights
    b = m.b.value  # input's bias
    u = m.u.value  # memory's weights
    h = m.h ≠ nothing ? m.h : Zeros(T, size(w,1), size(x,2))
    x = f(w*x + u*h .+ b)
    m.h = x
    return x
end


function predict(model::RNNs, x)
    for m in model
        x = predict(m, x)
    end
    return x
end


"""
    unbiasedof(m::RNN)

unbiased weights of RNN block
"""
function unbiasedof(m::RNN)
    weights = Vector(undef, 1)
    weights[1] = m.w.value
    return weights
end


function weightsof(m::RNN)
    weights = Vector(undef,3)
    weights[1] = m.w.value
    weights[2] = m.b.value
    weights[3] = m.u.value
    return weights
end


"""
    unbiasedof(model::RNNs)

unbiased weights of RNNs block
"""
function unbiasedof(model::RNNs)
    weights = Vector(undef, 0)
    for m in model
        append!(weights, unbiasedof(m))
    end
    return weights
end


function weightsof(m::RNNs)
    weights = Vector(undef,0)
    for i = 1:length(m)
        append!(weights, weightsof(m[i]))
    end
    return weights
end


function gradsof(m::RNN)
    grads = Vector(undef,3)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    grads[3] = m.u.delta
    return grads
end


function gradsof(m::RNNs)
    grads = Vector(undef,0)
    for i = 1:length(m)
        append!(grads, gradsof(m[i]))
    end
    return grads
end


function zerograds!(m::RNN)
    for v in gradsof(m)
        v .= zero(v)
    end
end


function zerograds!(m::RNNs)
    for v in gradsof(m)
        v .= zero(v)
    end
end


function paramsof(m::RNN)
    params = Vector{Variable}(undef,3)
    params[1] = m.w
    params[2] = m.b
    params[3] = m.u
    return params
end


function xparamsof(m::RNN)
    xparams = Vector{XVariable}(undef,3)
    xparams[1] = ('w', m.w)
    xparams[2] = ('b', m.b)
    xparams[3] = ('u', m.u)
    return xparams
end


function paramsof(m::RNNs)
    params = Vector{Variable}(undef,0)
    for i = 1:length(m)
        append!(params, paramsof(m[i]))
    end
    return params
end


function xparamsof(m::RNNs)
    xparams = Vector{XVariable}(undef,0)
    for i = 1:length(m)
        append!(xparams, xparamsof(m[i]))
    end
    return xparams
end


function nparamsof(m::RNN)
    lw = length(m.w)
    lb = length(m.b)
    lu = length(m.u)
    return (lw + lb + lu)
end


function bytesof(model::RNN, unit::String="MB")
    n = nparamsof(model) * elsizeof(model.w)
    return blocksize(n, uppercase(unit))
end


function nparamsof(m::RNNs)
    num = 0
    for i = 1:length(m)
        num += nparamsof(m[i])
    end
    return num
end


function bytesof(model::RNNs, unit::String="MB")
    n = nparamsof(model) * elsizeof(model[1].w)
    return blocksize(n, uppercase(unit))
end


function to(type::Type, m::RNN)
    m.w = to(type, m.w)
    m.b = to(type, m.b)
    m.u = to(type, m.u)
    return m
end


function to!(type::Type, m::RNN)
    m = to(type, m)
    return nothing
end


function to(type::Type, m::RNNs)
    for layer in m
        layer = to(type, layer)
    end
    return m
end


function to!(type::Type, m::RNNs)
    for layer in m
        to!(type, layer)
    end
end
