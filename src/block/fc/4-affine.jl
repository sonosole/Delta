mutable struct affine <: Block
    w::VarOrNil # input to hidden weights
    function affine(inputSize::Int, hiddenSize::Int; type::Type=Array{Float32})
        T = eltype(type)
        A = sqrt(T(1/hiddenSize))
        w = randn(T, hiddenSize, inputSize) .* A
        new(Variable{type}(w,true,true,true))
    end
    function affine()
        new(nothing)
    end
end


function clone(this::affine; type::Type=Array{Float32})
    cloned = affine()
    cloned.w = clone(this.w, type=type)
    return cloned
end


# pretty show
function Base.show(io::IO, m::affine)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    print(io, "affine($(SIZE[2]), $(SIZE[1]); type=$TYPE)")
end

"""
    unbiasedof(m::affine)

unbiased weights of affine block
"""
function unbiasedof(m::affine)
    weights = Vector(undef, 1)
    weights[1] = m.w.value
    return weights
end

function weightsof(m::affine)
    weights = Vector(undef, 1)
    weights[1] = m.w.value
    return weights
end


function gradsof(m::affine)
    grads = Vector(undef, 1)
    grads[1] = m.w.delta
    return grads
end


function zerograds!(m::affine)
    for v in gradsof(m)
        v .= 0.0
    end
end


function paramsof(m::affine)
    params = Vector{Variable}(undef,1)
    params[1] = m.w
    return params
end


function xparamsof(m::affine)
    xparams = Vector{XVariable}(undef,1)
    xparams[1] = ('w', m.w)
    return xparams
end


function nparamsof(m::affine)
    return length(m.w)
end


function bytesof(model::affine, unit::String="MB")
    n = nparamsof(model)
    u = uppercase(unit)
    if u == "KB" return n * sizeof(eltype(model.w)) / 1024 end
    if u == "MB" return n * sizeof(eltype(model.w)) / 1048576 end
    if u == "GB" return n * sizeof(eltype(model.w)) / 1073741824 end
    if u == "TB" return n * sizeof(eltype(model.w)) / 1099511627776 end
end


function forward(m::affine, x::Variable)
    w = m.w
    return w * x
end


function predict(m::affine, x)
    w = m.w.value
    return w * x
end


function to(type::Type, m::affine)
    m.w = to(type, m.w)
    return m
end


function to!(type::Type, m::affine)
    m.w = to(type, m.w)
    return nothing
end
