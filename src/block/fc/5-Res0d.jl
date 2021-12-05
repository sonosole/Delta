"""
    Res0d <: Block
Res0d is a residual block whose neuron deals with 0-d data, but without BatchNorm.

# Structure
       Rⁱ → Rᵐ         Rᵐ → Rⁱ
        ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐
    ─┬─►│ W ├──►│f()├──►│ W ├──►│ + ├──►│f()├──►
     │  └───┘   └───┘   └───┘   └─┬─┘   └───┘
     └────────────────►───────────┘
"""
mutable struct Res0d <: Block
    blocks::Vector
    function Res0d(i::Int, m::Int; type::Type=Array{Float32})
        l1 = linear(i, m, type=type)
        l2 = linear(m, i, type=type)
        new([l1, l2])
    end
    function Res0d()
        new(Vector(undef,2))
    end
end

@extend(Res0d, blocks)

function clone(this::Res0d; type::Type=Array{Float32})
    cloned = Res0d()
    cloned[1] = clone(this[1], type=type)
    cloned[2] = clone(this[2], type=type)
    return cloned
end

function forward(m::Res0d, x0)
    x1 = relu(forward(m[1], x0))
    x2 = x0 + forward(m[2], x1)
    return relu(x2)
end

function predict(m::Res0d, x0)
    x1 = relu(predict(m[1], x0))
    x2 = x0 + predict(m[2], x1)
    return relu(x2)
end
