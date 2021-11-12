mutable struct SelfLoopResNet <: Block
    self::Union{Block,Nothing}
    degree::Int
    function SelfLoopResNet(self::Block, n::Int)
        new(self, n)
    end
    function SelfLoopResNet(n::Int)
        new(nothing, n)
    end
end

function forward(m::SelfLoopResNet, x::Variable)
    z = x
    for i = 1:m.degree
        z = forward(m.self, z) + x
    end
    return z
end

function predict(m::SelfLoopResNet,  x::AbstractArray)
    z = x
    for i = 1:m.degree
        z = predict(m.self, z) + x
    end
    return z
end

function clone(this::SelfLoopResNet; type::Type=Array{Float32})
    cloned = SelfLoopResNet(this.degree)
    cloned.self = clone(this.self, type=type)
    return cloned
end

function unbiasedof(m::SelfLoopResNet)
    return unbiasedof(m.self)
end

function weightsof(m::SelfLoopResNet)
    return weightsof(m.self)
end

function gradsof(m::SelfLoopResNet)
    return gradsof(m.self)
end

function zerograds!(m::SelfLoopResNet)
    return zerograds!(m.self)
end

function paramsof(m::SelfLoopResNet)
    return paramsof(m.self)
end

function xparamsof(m::SelfLoopResNet)
    return xparamsof(m.self)
end

function nparamsof(m::SelfLoopResNet)
    return nparamsof(m.self)
end

function bytesof(m::SelfLoopResNet)
    return bytesof(m.self)
end

function to(type::Type, m::SelfLoopResNet)
    m.self = to(type, m.self)
    return m
end

function to!(type::Type, m::SelfLoopResNet)
    m.self = to(type, m.self)
    return nothing
end
