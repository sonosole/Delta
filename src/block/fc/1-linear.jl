export linear

mutable struct linear <: Block
    w::VarOrNil # input to hidden weights
    b::VarOrNil # bias of hidden units
    function linear(inputSize::Int, hiddenSize::Int; type::Type=Array{Float32})
        T = eltype(type)
        A = sqrt(T(1/hiddenSize))
        w = randn(T, hiddenSize, inputSize) .* A
        b = randn(T, hiddenSize,         1) .* A
        new(Variable{type}(w,true,true,true),
            Variable{type}(b,true,true,true))
    end
    function linear()
        new(nothing, nothing)
    end
end


function clone(this::linear; type::Type=Array{Float32})
    cloned = linear()
    cloned.w = clone(this.w, type=type)
    cloned.b = clone(this.b, type=type)
    return cloned
end


# pretty show
function Base.show(io::IO, m::linear)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    print(io, "linear($(SIZE[2]), $(SIZE[1]); type=$TYPE)")
end

"""
    unbiasedof(m::linear)

unbiased weights of linear block
"""
function unbiasedof(m::linear)
    weights = Vector(undef, 1)
    weights[1] = m.w.value
    return weights
end

function weightsof(m::linear)
    weights = Vector(undef, 2)
    weights[1] = m.w.value
    weights[2] = m.b.value
    return weights
end


function gradsof(m::linear)
    grads = Vector(undef, 2)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    return grads
end


function zerograds!(m::linear)
    for v in gradsof(m)
        v .= 0.0
    end
end


function paramsof(m::linear)
    params = Vector{Variable}(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function xparamsof(m::linear)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', m.w)
    xparams[2] = ('b', m.b)
    return xparams
end


function nparamsof(m::linear)
    lw = length(m.w)
    lb = length(m.b)
    return (lw + lb)
end


function forward(m::linear, x::Variable)
    w = m.w
    b = m.b
    return matAddVec(w * x, b)
end


function predict(m::linear, x)
    w = m.w.value
    b = m.b.value
    return (w * x .+ b)
end


function to(type::Type, m::linear)
    m.w = to(type, m.w)
    m.b = to(type, m.b)
    return m
end


function to!(type::Type, m::linear)
    m.w = to(type, m.w)
    m.b = to(type, m.b)
    return nothing
end
