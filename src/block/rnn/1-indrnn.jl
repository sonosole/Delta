export indrnn
export INDRNN
export uniform

function uniform(dtype::Type, shape::Tuple; from=dtype(0.0), to=dtype(1.0))
    From = dtype(from)
    To   = dtype(to)
    if from==dtype(0.0) && to==dtype(1.0)
        return rand(dtype, shape)
    else
        return rand(dtype, shape) .* (To - From) .+ From
    end
end


mutable struct indrnn <: Block
    w::Variable # input to hidden weights
    b::Variable # bias of hidden units
    u::Variable # recurrent weights
    f::Function # activation function
    h::Any      # hidden variable
    function indrnn(inputSize::Int, hiddenSize::Int, fn::Function=relu; type::Type=Array{Float32})
        T = eltype(type)
        w = randn(T, hiddenSize, inputSize) .* sqrt( T(2/inputSize) )
        b = zeros(T, hiddenSize, 1)
        u = uniform(T, (hiddenSize, 1); from=-0.1990, to=0.1997)
        new(Variable{type}(w,true,true,true),
            Variable{type}(b,true,true,true),
            Variable{type}(u,true,true,true), fn, nothing)
    end
end

mutable struct INDRNN <: Block
    layers::Vector{indrnn}
    function INDRNN(topology::Vector{Int}, fn::Array{F}; type::Type=Array{Float32}) where F
        n = length(topology) - 1
        layers = Vector{indrnn}(undef, n)
        for i = 1:n
            layers[i] = indrnn(topology[i], topology[i+1], fn[i]; type=type)
        end
        new(layers)
    end
end


Base.getindex(m::INDRNN,     k...) =  m.layers[k...]
Base.setindex!(m::INDRNN, v, k...) = (m.layers[k...] = v)
Base.length(m::INDRNN) = length(m.layers)


function Base.show(io::IO, m::indrnn)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    print(io, "indrnn($(SIZE[2]), $(SIZE[1]), $(m.f); type=$TYPE)")
end


function Base.show(io::IO, m::INDRNN)
    print(io, "INDRNN\n      (\n          ")
    join(io,      m.layers, ",\n          ")
    print(io,                   "\n      )")
end


function resethidden(m::indrnn)
    m.h = nothing
end


function resethidden(m::INDRNN)
    for i = 1:length(m)
        m[i].h = nothing
    end
end


function forward(m::indrnn, x::Variable{T}) where T
    f = m.f  # activition function
    w = m.w  # input's weights
    b = m.b  # input's bias
    u = m.u  # memory's weights
    h = m.h ≠ nothing ? m.h : Variable{T}(zeros(size(w,1),size(x,2)))
    x = f(matAddVec(matMulVec(h, u) + w*x, b))
    m.h = x
    return x
end


function forward(m::INDRNN, input::Variable)
    x = forward(m[1], input)
    for i = 2:length(m)
        x = forward(m[i], x)
    end
    return x
end


function predict(m::indrnn, x::AbstractArray{T}) where T
    f = m.f        # activition function
    w = m.w.value  # input's weights
    b = m.b.value  # input's bias
    u = m.u.value  # memory's weights
    h = m.h != nothing ? m.h : zeros(T,size(w,1),size(x,2))
    x = f(w*x + h .* u .+ b)
    m.h = x
    return x
end


function predict(m::INDRNN, input)
    x = predict(m[1], input)
    for i = 2:length(m)
        x = predict(m[i], x)
    end
    return x
end


function weightsof(m::indrnn)
    weights = Vector(undef,3)
    weights[1] = m.w.value
    weights[2] = m.b.value
    weights[3] = m.u.value
    return weights
end


function weightsof(m::INDRNN)
    weights = Vector(undef,0)
    for i = 1:length(m)
        append!(weights, weightsof(m[i]))
    end
    return weights
end


function gradsof(m::indrnn)
    grads = Vector(undef,3)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    grads[3] = m.u.delta
    return grads
end


function gradsof(m::INDRNN)
    grads = Vector(undef,0)
    for i = 1:length(m)
        append!(grads, gradsof(m[i]))
    end
    return grads
end


function zerograds(m::indrnn)
    for v in gradsof(m)
        v .= zero(v)
    end
end


function zerograds(m::INDRNN)
    for v in gradsof(m)
        v .= zero(v)
    end
end


function paramsof(m::indrnn)
    params = Vector{Variable}(undef,3)
    params[1] = m.w
    params[2] = m.b
    params[3] = m.u
    return params
end


function paramsof(m::INDRNN)
    params = Vector{Variable}(undef,0)
    for i = 1:length(m)
        append!(params, paramsof(m[i]))
    end
    return params
end


function nparamsof(m::indrnn)
    lw = length(m.w)
    lb = length(m.b)
    lu = length(m.u)
    return (lw + lb + lu)
end


function nparamsof(m::INDRNN)
    num = 0
    for i = 1:length(m)
        num += nparamsof(m[i])
    end
    return num
end



function to(type::Type, m::indrnn)
    m.w = to(type, m.w)
    m.b = to(type, m.b)
    m.u = to(type, m.u)
    return m
end


function to!(type::Type, m::indrnn)
    m.w = to(type, m.w)
    m.b = to(type, m.b)
    m.u = to(type, m.u)
    return nothing
end
