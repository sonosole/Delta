export dense
export MLP

mutable struct dense <: Block
    w::Variable
    b::Variable
    f::Function
    # type specilized may be CuArray/AFArray/ClArray/Array etc
    function dense(inputSize::Int, hiddenSize::Int, fn::Function=relu; type::Type=Array{Float32})
        T = eltype(type)
        a = sqrt(T(2/inputSize))
        w = randn(T, hiddenSize, inputSize) .* a
        b = randn(T, hiddenSize,         1) .* a
        new(Variable{type}(w,true,true,true), Variable{type}(b,true,true,true), fn)
    end
end


# pretty show
function Base.show(io::IO, m::dense)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    print(io, "dense($(SIZE[2]), $(SIZE[1]), $(m.f); type=$TYPE)")
end


mutable struct MLP <: Block
    layers::Vector{dense}

    function MLP(topology::Vector{Int}; type::Type=Array{Float32})
        n = length(topology) - 1
        layers = Vector{dense}(undef, n)
        for i = 1:n
            layers[i] = dense(topology[i], topology[i+1], relu; type=type)
        end
        new(layers)
    end

    function MLP(topology::Vector{Int}, fn::Vector{F}; type::Type=Array{Float32}) where F
        n = length(topology) - 1
        layers = Vector{dense}(undef, n)
        for i = 1:n
            layers[i] = dense(topology[i], topology[i+1], fn[i]; type=type)
        end
        new(layers)
    end
end


# pretty show
function Base.show(io::IO, m::MLP)
    print(io, "MLP\n      (\n          ")
    join(io,    m.layers, "\n          ")
    print(io,                "\n      )")
end


Base.length(m::MLP)             = length(m.layers)
Base.getindex(m::MLP, k...)     =        m.layers[k...]
Base.setindex!(m::MLP, v, k...) =       (m.layers[k...] = v)


function forward(m::dense, x::Variable)
    f = m.f
    w = m.w
    b = m.b
    return f( matAddVec(w*x, b) )
end


function forward(m::MLP, input::Variable)
    x = forward(m[1], input)
    for i = 2:length(m)
        x = forward(m[i], x)
    end
    return x
end


function predict(m::dense, x)
    f = m.f
    w = m.w.value
    b = m.b.value
    return f(w * x .+ b)
end


function predict(m::MLP, input)
    x = predict(m[1], input)
    for i = 2:length(m)
        x = predict(m[i], x)
    end
    return x
end


function weightsof(m::dense)
    weights = Vector(undef, 2)
    weights[1] = m.w.value
    weights[2] = m.b.value
    return weights
end


function weightsof(m::MLP)
    weights = Vector(undef,0)
    for i = 1:length(m)
        append!(weights, weightsof(m[i]))
    end
    return weights
end


function gradsof(m::dense)
    grads = Vector(undef, 2)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    return grads
end


function gradsof(m::MLP)
    grads = Vector(undef,0)
    for i = 1:length(m)
        append!(grads, gradsof(m[i]))
    end
    return grads
end


function zerograds(m::dense)
    for v in gradsof(m)
        v .= 0.0
    end
end


function zerograds(m::MLP)
    for v in gradsof(m)
        v .= 0.0
    end
end


function paramsof(m::dense)
    params = Vector{Variable}(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function paramsof(m::MLP)
    params = Vector{Variable}(undef,0)
    for i = 1:length(m)
        append!(params, paramsof(m[i]))
    end
    return params
end


function nparamsof(m::dense)
    lw = length(m.w)
    lb = length(m.b)
    return (lw + lb)
end


function nparamsof(m::MLP)
    num = 0
    for i = 1:length(m)
        num += nparamsof(m[i])
    end
    return num
end


function to(type::Type, m::dense)
    m.w = to(type, m.w)
    m.b = to(type, m.b)
    return m
end


function to!(type::Type, m::dense)
    m.w = to(type, m.w)
    m.b = to(type, m.b)
    return nothing
end
