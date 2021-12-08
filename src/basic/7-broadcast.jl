"""
    axes2reduce(z, x)
z = broadcast(::typeof(+-*/), x, y)
(3,4,5),(1,4,)  -> (1,3)
(3,4,5),(1,4,1) -> (1,3)
"""
function axes2reduce(z, x)
    a = Int[]
    for i = 1:ndims(z)
        if size(x, i) == 1
            push!(a, i)
        end
    end
    return a
end


"""
    unbcast(δx::AbstractArray, x::AbstractArray) -> ∇x

reduced `δx` to `∇x` according to shape difference from `x` and `δx`

# Params
`x`  : comes from `z = broadcast(::typeof(+-*/...), x, y)`\n
`δx` : unreduced gradient, i.e. `δx = δz .* ∂z/∂x`\n
`∇x` : reduced gradient, i.e. ⤓\n
       Δx = sum(δx, dims=axes2reduce(δx, x)) # reduced but still has redundant dimensions\n
       ∇x = reshape(Δx, size(x))
"""
function unbcast(δx::AbstractArray, x::AbstractArray)
    if size(δx) == size(x)
        return δx
    elseif length(δx) == length(x)
        return reshape(δx, size(x))
    else
        Δx = sum(δx, dims=axes2reduce(δx,x))
        return reshape(Δx, size(x))
    end
end




function Base.Broadcast.broadcasted(::typeof(+), x::Variable{T1}, y::Variable{T2}) where {T1,T2}
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    backprop = (x.backprop || y.backprop)
    z = Variable{T}(ᵛ(x) .+ ᵛ(y), backprop)
    if backprop
        function DotAddBackward()
            if need2computeδ!(x)
                δx = δ(z)
                δ(x) .+= unbcast(δx, ᵛ(x))
            end
            if need2computeδ!(y)
                δy = δ(z)
                δ(y) .+= unbcast(δy, ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(z);
        end
        push!(graph.backward, DotAddBackward)
    end
    return z
end


function Base.Broadcast.broadcasted(::typeof(-), x::Variable{T1}, y::Variable{T2}) where {T1,T2}
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    backprop = (x.backprop || y.backprop)
    z = Variable{T}(ᵛ(x) .- ᵛ(y), backprop)
    if backprop
        function DotMinusBackward()
            if need2computeδ!(x)
                δx = δ(z)
                δ(x) .+= unbcast(δx, ᵛ(x))
            end
            if need2computeδ!(y)
                δy = - δ(z)
                δ(y) .+= unbcast(δy, ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(z);
        end
        push!(graph.backward, DotMinusBackward)
    end
    return z
end


function Base.Broadcast.broadcasted(::typeof(*), x::Variable{T1}, y::Variable{T2}) where {T1,T2}
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    backprop = (x.backprop || y.backprop)
    z = Variable{T}(ᵛ(x) .* ᵛ(y), backprop)
    if backprop
        function DotMulBackward()
            if need2computeδ!(x)
                δx = δ(z) .* ᵛ(y)
                δ(x) .+= unbcast(δx, ᵛ(x))
            end
            if need2computeδ!(y)
                δy = δ(z) .* ᵛ(x)
                δ(y) .+= unbcast(δy, ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(z);
        end
        push!(graph.backward, DotMulBackward)
    end
    return z
end


function Base.Broadcast.broadcasted(::typeof(/), x::Variable{T1}, y::Variable{T2}) where {T1,T2}
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    backprop = (x.backprop || y.backprop)
    z = Variable{T}(ᵛ(x) ./ ᵛ(y), backprop)
    if backprop
        function DotDivBackward()
            δx = δ(z) ./ y
            if need2computeδ!(x)
                δ(x) .+= unbcast(δx, ᵛ(x))
            end
            if need2computeδ!(y)
                δy = - δx .* ᵛ(z)
                δ(y) .+= unbcast(δy, ᵛ(y))
            end
            ifNotKeepδThenFreeδ!(z);
        end
        push!(graph.backward, DotDivBackward)
    end
    return z
end
