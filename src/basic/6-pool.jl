import Base.maximum
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

import Base.minimum
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

import Base.sum
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


function mean(var::Variable{T}; dims::Union{Int,NTuple{N,Int}}) where {T,N}
    CST = eltype(var)(1.0) / prod([size(var, i) for i in dims])
    out = Variable{T}(sum(var.value, dims=dims) .* CST, var.backprop)
    if var.backprop
        function meanBackward()
            if need2computeδ!(var)
                var.delta .+= out.delta .* CST
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, meanBackward)
    end
    return out
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

import Base.minmax
function Base.minmax(var::Variable{T}; dims1::Int, dims2::Int) where T
    return maxmin(var; dims1=dims2, dims2=dims1)
end


function Base.minmax(x::AbstractArray; dims1::Int, dims2::Int)
    return maximum(minimum(x, dims=dims1), dims=dims2)
end


function linearpool(var::Variable{T}; dims::Union{Int,NTuple{N,Int}}) where {T,N}
    sum1 = sum(var.value .* var.value, dims=dims)     # Σ xᵢ·xᵢ
    sum2 = sum(var.value,              dims=dims)     # Σ xᵢ
    out  = Variable{T}(sum1 ./ sum2, var.backprop)
    if var.backprop
        TWO = eltype(var)(2.0)
        function linearpoolBackward()
            if need2computeδ!(var)
                var.delta += (TWO .* var.value .- out.value) ./ sum2 .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, linearpoolBackward)
    end
    return out
end


function linearpool(x::AbstractArray; dims::Union{Int,NTuple{N,Int}}) where N
    return sum(x .* x, dims=dims) ./ sum(x, dims=dims)
end


function exppool(var::Variable{T}; dims::Union{Int,NTuple{N,Int}}) where {T,N}
    tmp  = exp.(var.value)
    sum1 = sum(tmp .* var.value, dims=dims)         # Σ exp(xᵢ)·xᵢ
    sum2 = sum(tmp,              dims=dims)         # Σ exp(xᵢ)
    out  = Variable{T}(sum1 ./ sum2, var.backprop)
    if var.backprop
        ONE = eltype(var)(1.0)
        function exppoolBackward()
            if need2computeδ!(var)
                var.delta += tmp ./ sum2 .* (ONE .+ var.value .- out.value) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, exppoolBackward)
    end
    return out
end


function exppool(x::AbstractArray; dims::Union{Int,NTuple{N,Int}}) where N
    e = exp.(x)
    return sum(e .* x, dims=dims) ./ sum(e, dims=dims)
end
