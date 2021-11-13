mutable struct MeanNormResDense <: Block
    blocks::Vector
    function MeanNormResDense(ichannels,  # input #channels, equals to output #channels
                              mchannels;  # intermediary #channels (or dimension)
                              type::Type=Array{Float32})

        linear1 = affine(ichannels, mchannels, type=type)
        normer1 = MeanNorm(ndims=2, keptdims=1, keptsize=mchannels, type=type)
        linear2 = affine(mchannels, ichannels, type=type)
        normer2 = MeanNorm(ndims=2, keptdims=1, keptsize=ichannels, type=type)
        new([linear1, normer1, linear2, normer2])
    end
    function MeanNormResDense()
        new(Vector(undef,4))
    end
end

Base.length(m::MeanNormResDense)     = 4
Base.lastindex(m::MeanNormResDense)  = 4
Base.firstindex(m::MeanNormResDense) = 1
Base.getindex(m::MeanNormResDense, k...)     =  m.blocks[k...]
Base.setindex!(m::MeanNormResDense, v, k...) = (m.blocks[k...] = v)
Base.iterate(m::MeanNormResDense, i=1) = i>4 ? nothing : (m[i], i+1)


@basic(MeanNormResDense, blocks)

function clone(this::MeanNormResDense; type::Type=Array{Float32})
    cloned = MeanNormResDense()
    for i = 1:4
        cloned[i] = clone(this[i], type=type)
    end
    return cloned
end

function forward(m::MeanNormResDense, x0::Variable{T}) where T
    x =      forward(m[1], x0)
    x = relu(forward(m[2], x))
    x =      forward(m[3], x)
    x = relu(forward(m[4], x) + x0)
    return x
end

function predict(m::MeanNormResDense, x0::AbstractArray)
    x =      predict(m[1], x0)
    x = relu(predict(m[2], x))
    x =      predict(m[3], x)
    x = relu(predict(m[4], x) + x0)
    return x
end
