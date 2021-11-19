function Base.maximum(x::Variable{T}; dims::Union{Int,NTuple{N,Int}}) where {T,N}
    y = Variable{T}(maximum(ᵛ(x), dims=dims), x.backprop)
    if x.backprop
        mask = ᵛ(x) .== ᵛ(y)
        function maximumBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* mask
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, maximumBackward)
    end
    return y
end

function Base.minimum(x::Variable{T}; dims::Union{Int,NTuple{N,Int}}) where {T,N}
    y = Variable{T}(minimum(ᵛ(x), dims=dims), x.backprop)
    if x.backprop
        mask = ᵛ(x) .== ᵛ(y)
        function minimumBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* mask
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, minimumBackward)
    end
    return y
end

function Base.sum(x::Variable{T}; dims::Union{Int,NTuple{N,Int}}) where {T,N}
    y = Variable{T}(sum(ᵛ(x), dims=dims), x.backprop)
    if x.backprop
        function sumBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        push!(graph.backward, sumBackward)
    end
    return y
end


function mean(x::Variable{T}; dims::Union{Int,NTuple{N,Int}}) where {T,N}
    n = eltype(x)(1) / prod([size(x, i) for i in dims])
    μ = Variable{T}(sum(ᵛ(x), dims=dims) .* n, x.backprop)
    if x.backprop
        function meanBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(μ) .* n
            end
            ifNotKeepδThenFreeδ!(μ);
        end
        push!(graph.backward, meanBackward)
    end
    return μ
end


function maxmin(x::Variable{T}; dims1::Int, dims2::Int) where T
    t = minimum(maximum(ᵛ(x), dims=dims1), dims=dims2)
    y = Variable{T}(t, x.backprop)
    if x.backprop
        mask = ᵛ(x) .== ᵛ(y)
        function maxminBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* mask
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, maxminBackward)
    end
    return y
end


function maxmin(x::AbstractArray; dims1::Int, dims2::Int)
    return minimum( maximum(x, dims=dims1), dims=dims2)
end

function Base.minmax(x::Variable{T}; dims1::Int, dims2::Int) where T
    return maxmin(x; dims1=dims2, dims2=dims1)
end


function Base.minmax(x::AbstractArray; dims1::Int, dims2::Int)
    return maximum(minimum(x, dims=dims1), dims=dims2)
end


function linearpool(x::Variable{T}; dims::Union{Int,NTuple{N,Int}}) where {T,N}
    Σxᵢ² = sum(ᵛ(x) .* ᵛ(x), dims=dims)     # Σ xᵢ·xᵢ
    Σxᵢ  = sum(ᵛ(x),         dims=dims)     # Σ xᵢ
    y    = Variable{T}(Σxᵢ² ./ Σxᵢ, x.backprop)
    if x.backprop
        TWO = eltype(x)(2.0f0)
        function linearpoolBackward()
            if need2computeδ!(x)
                δ(x) .+= (TWO .* ᵛ(x) .- ᵛ(y)) ./ Σxᵢ .* δ(y)
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
    eˣ  = exp.(ᵛ(x))
    Σeˣⁱxᵢ = sum(eˣ .* ᵛ(x), dims=dims)   # Σ exp(xᵢ)·xᵢ
    Σeˣⁱ = sum(eˣ, dims=dims)             # Σ exp(xᵢ)
    y  = Variable{T}(Σeˣⁱxᵢ ./ Σeˣⁱ, x.backprop)
    if x.backprop
        ONE = eltype(x)(1.0f0)
        function exppoolBackward()
            if need2computeδ!(x)
                δ(x) .+= eˣ ./ Σeˣⁱ .* (ONE .+ ᵛ(x) .- ᵛ(y)) .* δ(y)
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
