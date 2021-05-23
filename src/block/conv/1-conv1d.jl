export conv1d
export conv1dReceptiveField


"""
    Computes a 1-D convolution given 3-D input and 3-D filter tensors.
    Input 3D-tensor of shape (ichannels, timeSteps, batchsize)
    Filter 3D-tensor of shape (ochannels, ichannels, kernel) but
    actually reshaped to 2D-tensor of shape (ochannels, ichannels*kernel) for convenient.
"""
mutable struct conv1d <: Block
    w::Variable # input to hidden weights
    b::Variable # bias of hidden units
    k::Int      # kernel size
    s::Int      # stride size
    p::Int      # padding size
    function conv1d(ichannels::Int, ochannels::Int, kernel::Int;
        stride::Int = 1,
        padding::Int = 0,
        type::Type=Array{Float32})

        dtype = eltype(type)
        filterSize = ichannels * kernel
        amplitude = sqrt(dtype(2/filterSize))
        w = amplitude * randn(dtype, ochannels, filterSize)
        b = amplitude * randn(dtype, ochannels,          1)
        new(Variable{type}(w,true,true,true),
            Variable{type}(b,true,true,true),
            kernel, stride, padding)
    end
end


# pretty show
function Base.show(io::IO, m::conv1d)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    print(io, "conv1d($(Int(SIZE[2]/m.k)), $(SIZE[1]), kernel=$(m.k), stride=$(m.s); type=$TYPE)")
end


function weightsof(m::conv1d)
    weights = Vector(undef, 2)
    weights[1] = m.w.value
    weights[2] = m.b.value
    return weights

end


function gradsof(m::conv1d)
    grads = Vector(undef, 2)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    return grads
end


function zerograds(m::conv1d)
    for v in gradsof(m)
        v .= 0.0
    end
end


function paramsof(m::conv1d)
    params = Vector(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function nparamsof(m::conv1d)
    lw = length(m.w)
    lb = length(m.b)
    return (lw + lb)
end


"""
    conv1dReceptiveField(StrideKernelPair::Vector{NTuple{2,Int}})
```julia
julia> conv1dReceptiveField([(3,2),(3,1),(4,2)]
(1:13, 5:17)
```
"""
function conv1dReceptiveField(StrideKernelPair::Vector{NTuple{2,Int}})
    # 输入是从底层到顶层的(kernel,stride)列表
    # 计算感受野时从顶层往底层计算,为了流式计算时候缓存空间的设计
    # 本函数返回：顶层第一个时间步感受到的底层时间步范围
    #            顶层第二个时间步感受到的底层时间步范围
    t1 = 1
    t2 = 2
    for (kernel,stride) in StrideKernelPair[end:-1:1]
        t1 = (t1-1) * stride + kernel
        t2 = (t2-1) * stride + kernel
    end
    return (1:t1,t2-t1+1:t2)
end


function conv1dReceptiveField(chain::Chain)
    # 计算感受野时从顶层往底层计算,为了流式计算时候缓存空间的设计
    # 本函数返回：顶层第一个时间步感受到的底层时间步范围
    #            顶层第二个时间步感受到的底层时间步范围
    t1 = 1
    t2 = 2
    for i = length(chain):-1:1
        @assert (typeof(chain[i]) <: conv1d) "$(typeof(chain[i])) <: conv1d"
        t1 = (t1-1) * chain[i].s + chain[i].k
        t2 = (t2-1) * chain[i].s + chain[i].k
    end
    return (1:t1,t2-t1+1:t2)
end


# in2col for predict
function in2col(var::Array{T}, kernel::Int, stride::Int) where T
    # from (ichannels,width,batchsize) to (ichannels*kernel,cols)
    (ichannels,width,batchsize) = size(var)
    step = floor(Int,(width-kernel)/stride + 1)
    cols = step * batchsize
    rows = ichannels * kernel
    out  = zeros(T, rows, cols)
    Threads.@threads for b = 1:batchsize
        index = 1 + (b-1)*step
        start = 1
        final = kernel
        for s = 1:step
            out[:,index] = reshape(var[:,start:final,b], (rows,1))
            start += stride
            final += stride
            index += 1
        end
    end
    return out
end

# in2col for training
function in2col(var::Variable{Array{T}}, kernel::Int, stride::Int) where T
    # var from (ichannels,width,batchsize) to (ichannels*kernel,cols)
    # in which cols = (width – kernel + 1) * batchsize
    (ichannels,width,batchsize) = size(var)
    step = floor(Int,(width-kernel)/stride + 1)
    cols = step * batchsize
    rows = ichannels * kernel
    out  = Variable{Array{T}}(zeros(T, rows, cols), var.backprop)

    Threads.@threads for b = 1:batchsize
        index = 1 + (b-1)*step
        start = 1
        final = kernel
        for s = 1:step
            out.value[:,index] = reshape(var.value[:,start:final,b], (rows,1))
            start += stride
            final += stride
            index += 1
        end
    end

    if var.backprop
        function in2colBackward()
            if need2computeδ!(var)
                Threads.@threads for b = 1:batchsize
                    index = 1 + (b-1)*step
                    start = 1
                    final = kernel
                    for s = 1:step
                        var.delta[:,start:final,b] += reshape(out.delta[:,index], (ichannels, kernel))
                        start += stride
                        final += stride
                        index += 1
                    end
                end
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, in2colBackward)
    end
    return out
end

# col2out for predict
function col2out(x::AbstractArray, batchsize::Int)
    # from (ochannels,width*batchsize) to (ochannels,width,batchsize)
    (ochannels, cols) = size(x)
    width = div(cols, batchsize)
    return reshape(x, (ochannels, width, batchsize))
end

# col2out for training
function col2out(x::Variable, batchsize::Int)
    # from (ochannels,width*batchsize) to (ochannels,width,batchsize)
    (ochannels, cols) = size(x)
    width = div(cols, batchsize)
    return reshape(x, (ochannels, width, batchsize))
end


function forward(model::conv1d, x::Variable{T}) where T
    # size(x) == (ichannels,width,batchsize)
    @assert ndims(x)==3 "input shape is of (ichannels,width,batchsize)"
    batchsize = size(x,3)
    w = model.w
    b = model.b
    x = in2col(x, model.k, model.s)
    x = matAddVec(w * x, b)
    return col2out(x, batchsize)
end


function predict(model::conv1d, x::AbstractArray)
    # size(x) == (ichannels,width,batchsize)
    @assert ndims(x)==3 "input shape is of (ichannels,width,batchsize)"
    batchsize = size(x,3)
    w = model.w.value
    b = model.b.value
    x = in2col(x, model.k, model.s)
    x = w * x .+ b
    return col2out(x, batchsize)
end
