export SelfLoopCumulativeResNet

mutable struct SelfLoopCumulativeResNet
    self::Union{Block,Nothing}
    degree::Int
    function SelfLoopCumulativeResNet(self::Block, n::Int)
        new(self, n)
    end
    function SelfLoopCumulativeResNet(n::Int)
        new(nothing, n)
    end
end

function forward(m::SelfLoopCumulativeResNet, x::Variable)
    for i = 1:m.degree
        x = forward(m.self, x) + x
    end
    return x
end

function predict(m::SelfLoopCumulativeResNet,  x::AbstractArray)
    for i = 1:m.degree
        x = predict(m.self, x) + x
    end
    return x
end

function clone(this::SelfLoopCumulativeResNet; type::Type=Array{Float32})
    cloned = SelfLoopCumulativeResNet(this.degree)
    cloned.self = clone(this.self, type=type)
    return cloned
end

function unbiasedof(m::SelfLoopCumulativeResNet)
    return unbiasedof(m.self)
end

function weightsof(m::SelfLoopCumulativeResNet)
    return weightsof(m.self)
end

function gradsof(m::SelfLoopCumulativeResNet)
    return gradsof(m.self)
end

function zerograds!(m::SelfLoopCumulativeResNet)
    return zerograds!(m.self)
end

function paramsof(m::SelfLoopCumulativeResNet)
    return paramsof(m.self)
end

function xparamsof(m::SelfLoopCumulativeResNet)
    return xparamsof(m.self)
end

function nparamsof(m::SelfLoopCumulativeResNet)
    return nparamsof(m.self)
end

function bytesof(m::SelfLoopCumulativeResNet)
    return bytesof(m.self)
end

function to(type::Type, m::SelfLoopCumulativeResNet)
    m.self = to(type, m.self)
    return m
end

function to!(type::Type, m::SelfLoopCumulativeResNet)
    m.self = to(type, m.self)
    return nothing
end
