function Base.maximum(var::Variable{T}; dims::Union{Int,NTuple{N,Int}}) where {T,N}
    out = Variable{T}(maximum(var.value, dims=dims), var.backprop)
    if var.backprop
        mask = var.value .== out.value
        function maximumBackward()
            if need2computeδ!(var)
                var.delta += out.delta .* mask
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, maximumBackward)
    end
    return out
end

function Base.minimum(var::Variable{T}; dims::Union{Int,NTuple{N,Int}}) where {T,N}
    out = Variable{T}(minimum(var.value, dims=dims), var.backprop)
    if var.backprop
        mask = var.value .== out.value
        function minimumBackward()
            if need2computeδ!(var)
                var.delta += out.delta .* mask
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, minimumBackward)
    end
    return out
end

function Base.sum(var::Variable{T}; dims::Union{Int,NTuple{N,Int}}) where {T,N}
    out = Variable{T}(sum(var.value, dims=dims), var.backprop)
    if var.backprop
        function sumBackward()
            if need2computeδ!(var)
                var.delta .+= out.delta
            end
            ifNotKeepδThenFreeδ!(out)
        end
        push!(graph.backward, sumBackward)
    end
    return out
end


function mean(x::Variable{T}; dims::Union{Int,NTuple{N,Int}}) where {T,N}
    n = eltype(x)(1) / prod([size(x, i) for i in dims])
    μ = Variable{T}(sum(x.value, dims=dims) .* n, x.backprop)
    if x.backprop
        function meanBackward()
            if need2computeδ!(x)
                x.delta .+= μ.delta .* n
            end
            ifNotKeepδThenFreeδ!(μ);
        end
        push!(graph.backward, meanBackward)
    end
    return μ
end


function maxmin(var::Variable{T}; dims1::Int, dims2::Int) where T
    tmp = minimum(maximum(var.value, dims=dims1), dims=dims2)
    out = Variable{T}(tmp, var.backprop)
    if var.backprop
        mask = var.value .== out.value
        function maxminBackward()
            if need2computeδ!(var)
                var.delta .+= out.delta .* mask
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, maxminBackward)
    end
    return out
end


function maxmin(x::AbstractArray; dims1::Int, dims2::Int)
    return minimum( maximum(x, dims=dims1), dims=dims2)
end

function Base.minmax(var::Variable{T}; dims1::Int, dims2::Int) where T
    return maxmin(var; dims1=dims2, dims2=dims1)
end


function Base.minmax(x::AbstractArray; dims1::Int, dims2::Int)
    return maximum(minimum(x, dims=dims1), dims=dims2)
end


function linearpool(x::Variable{T}; dims::Union{Int,NTuple{N,Int}}) where {T,N}
    Σxᵢ² = sum(x.value .* x.value, dims=dims)     # Σ xᵢ·xᵢ
    Σxᵢ  = sum(x.value,            dims=dims)     # Σ xᵢ
    y    = Variable{T}(Σxᵢ² ./ Σxᵢ, x.backprop)
    if x.backprop
        TWO = eltype(x)(2.0f0)
        function linearpoolBackward()
            if need2computeδ!(x)
                x.delta += (TWO .* x.value .- y.value) ./ Σxᵢ .* y.delta
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, linearpoolBackward)
    end
    return y
end


function linearpool(x::AbstractArray; dims::Union{Int,NTuple{N,Int}}) where N
    return sum(x .* x, dims=dims) ./ sum(x, dims=dims)
end


function exppool(x::Variable{T}; dims::Union{Int,NTuple{N,Int}}) where {T,N}
    eˣ  = exp.(x.value)
    Σeˣⁱxᵢ = sum(eˣ .* x.value, dims=dims)         # Σ exp(xᵢ)·xᵢ
    Σeˣⁱ = sum(eˣ, dims=dims)                      # Σ exp(xᵢ)
    y  = Variable{T}(Σeˣⁱxᵢ ./ Σeˣⁱ, x.backprop)
    if x.backprop
        ONE = eltype(x)(1.0f0)
        function exppoolBackward()
            if need2computeδ!(x)
                x.delta += eˣ ./ Σeˣⁱ .* (ONE .+ x.value .- y.value) .* y.delta
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, exppoolBackward)
    end
    return y
end


function exppool(x::AbstractArray; dims::Union{Int,NTuple{N,Int}}) where N
    e = exp.(x)
    return sum(e .* x, dims=dims) ./ sum(e, dims=dims)
end



export mean
export maxmin
export linearpool
export exppool
