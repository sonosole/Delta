"""
    Res0dWithBN <: Block
Res0dWithBN is a residual whose neuron processing 0-d data. which has 2 BatchNorm0d blocks

# Structure
       Rⁱ → Rᵐ                  Rᵐ → Rⁱ
        ┌───┐   ┌────┐   ┌───┐   ┌───┐   ┌────┐   ┌───┐   ┌───┐
    ─┬─►│ W ├──►│ BN ├──►│f()├──►│ W ├──►│ BN ├──►│ + ├──►│f()├──►
     │  └───┘   └────┘   └───┘   └───┘   └────┘   └─┬─┘   └───┘
     └─────────────────────────►────────────────────┘
"""
mutable struct Res0dWithBN <: Block
    blocks::Vector
    function Res0dWithBN(i::Int, m::Int; type::Type=Array{Float32})
        a1 = affine(i, m, type=type)
        b1 = BatchNorm0d(m)
        a2 = affine(m, i, type=type)
        b2 = BatchNorm0d(i)
        new([a1, b1, a2, b2])
    end
    function Res0dWithBN()
        new(Vector(undef,4))
    end
end

@extend(Res0dWithBN, blocks)

function clone(this::Res0dWithBN; type::Type=Array{Float32})
    cloned = Res0dWithBN()
    cloned[1] = clone(this[1], type=type)
    cloned[2] = clone(this[2], type=type)
    cloned[3] = clone(this[3], type=type)
    cloned[4] = clone(this[4], type=type)
    return cloned
end

function forward(m::Res0dWithBN, x0)
    x1 = forward(m[1], x0) # affine
    x2 = forward(m[2], x1) # batchnorm0d
    x3 =          relu(x2) # activation
    x4 = forward(m[3], x3) # affine
    x5 = forward(m[4], x4) # batchnorm0d
    return relu(x5 + x0)   # short connectin + activation
end

function predict(m::Res0dWithBN, x0)
    x1 = predict(m[1], x0) # affine
    x2 = predict(m[2], x1) # batchnorm0d
    x3 =          relu(x2) # activation
    x4 = predict(m[3], x3) # affine
    x5 = predict(m[4], x4) # batchnorm0d
    return relu(x5 + x0)   # short connectin + activation
end
