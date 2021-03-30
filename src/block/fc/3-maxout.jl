export maxout


mutable struct maxout <: Block
    w::Variable # input to middle hidden weights
    b::Variable # bias of middle hidden units
    h::Int
    k::Int
    function maxout(inputSize::Int, hiddenSize::Int; k::Int=2, type::Type=Array{Float32})
        @assert (k>=2) "# of affine layers should no less than 2"
        d = hiddenSize * k
        w = randn(d, inputSize) .* sqrt( 1 / d )
        b = zeros(d, 1)
        new(Variable{type}(w,true,true,true),
            Variable{type}(b,true,true,true), hiddenSize, k)
    end
end

# pretty show
function Base.show(io::IO, m::maxout)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    maxk = m.k
    print(io, "maxout($(SIZE[2]), $(SIZE[1]÷maxk); k=$maxk, type=$TYPE)")
end


function paramsof(m::maxout)
    params = Vector{Variable}(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function nparamsof(m::maxout)
    lw = length(m.w)
    lb = length(m.b)
    return (lw + lb)
end


function forward(model::maxout, x::Variable{T}) where T
    h = model.h
    k = model.k
    w = model.w
    b = model.b
    c = size(x, 2)
    x = matAddVec(w * x, b)         # dim=(h*k, c)
    temp = reshape(x.value, h,k,c)  # dim=(h,k,c)
    maxv = maximum(temp, dims=2)    # dim=(h,1,c)
    mask = temp .== maxv            # dim=(h,k,c)
    out  = Variable{T}(reshape(maxv, h,c), x.backprop)
    if x.backprop
        function maxoutBackward()
            if need2computeδ!(x)
                x.delta += reshape(mask .* reshape(out.delta, h,1,c), h*k,c)
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, maxoutBackward)
    end
    return out
end


function predict(model::maxout, x::AbstractArray)
    h = model.h
    k = model.k
    w = model.w.value
    b = model.b.value
    c = size(x, 2)
    x = w * x .+ b                  # dim=(h*k, c)
    temp = reshape(x, h,k,c)        # dim=(h,k,c)
    maxv = maximum(temp, dims=2)    # dim=(h,1,c)
    out  = reshape(maxv, h,c)       # dim=(h,  c)
    return out
end
