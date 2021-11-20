mutable struct ResDense <: Block
    blocks::Vector
    function ResDense(i::Int, m::Int; type::Type=Array{Float32})
        l1 = linear(i, m, type=type)
        l2 = linear(m, i, type=type)
        new([l1, l2])
    end
    function ResDense()
        new(Vector(undef,2))
    end
end

@extend(ResDense, blocks)

function clone(this::ResDense; type::Type=Array{Float32})
    cloned = ResDense()
    cloned[1] = clone(this[1], type=type)
    cloned[2] = clone(this[2], type=type)
    return cloned
end

function forward(m::ResDense, x0)
    x1 = relu(forward(m[1], x0))
    x2 = x0 + forward(m[2], x1)
    return relu(x2)
end

function predict(m::ResDense, x0)
    x1 = relu(predict(m[1], x0))
    x2 = x0 + predict(m[2], x1)
    return relu(x2)
end
