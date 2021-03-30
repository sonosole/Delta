export linear

mutable struct linear <: Block
    w::Variable # input to hidden weights
    b::Variable # bias of hidden units
    function linear(inputSize::Int, hiddenSize::Int; type::Type=Array{Float32})
        w = sqrt(1/hiddenSize) .* randn(hiddenSize, inputSize)
        b = sqrt(1/hiddenSize) .* randn(hiddenSize,         1)
        new(Variable{type}(w,true,true,true),
            Variable{type}(b,true,true,true))
    end
end


# pretty show
function Base.show(io::IO, m::linear)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    print(io, "linear($(SIZE[2]), $(SIZE[1]); type=$TYPE)")
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


function zerograds(m::linear)
    for v in gradsof(m)
        v .= 0.0
    end
end


function paramsof(m::linear)
    params = Vector(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
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
