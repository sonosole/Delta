"""
    axis2reduce(z, x)
z = broadcast(::typeof(+-*/), x, y)
(3,4,5),(1,4,)  -> (1,3)
(3,4,5),(1,4,1) -> (1,3)
"""
function axis2reduce(z, x)
    a = Int[]
    for i = 1:ndims(z)
        if size(x, i) == 1
            push!(a, i)
        end
    end
    return a
end


"""
    axis2recover(δx, x)

`δx` 与 `x` 有相同的元素数，但 `δx` 的形状里有更多的1，删除 `δx` 的维度形状里比`x`的维度形状里多的1并返回

# Example
julia> axis2recover(ones(1,3,3,1,1), ones(1,3,3))
(1, 3, 3)
julia> axis2recover(ones(1,3,3,1,1), ones(1,3,3,1))
(1, 3, 3, 1)
"""
function axis2recover(δx, x)
    return ntuple(i -> size(δx, i), ndims(x))
end


function unbcast(x::AbstractArray, δx::AbstractArray)
    #  z = broadcast(::typeof(+-*/...), x, y)
    # δx = δz .* ∂z/∂x
    if size(δx) == size(x)
        return δx
    elseif length(δx) == length(x)
        return reshape(δx, axis2recover(δx, x))
    else
        return reshape(sum(δx, dims=axis2reduce(δx,x)), axis2recover(δx, x))
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
                δ(x) .+= unbcast(ᵛ(x), δx)
            end
            if need2computeδ!(y)
                δy = δ(z)
                δ(y) .+= unbcast(ᵛ(y), δy)
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
                δ(x) .+= unbcast(ᵛ(x), δx)
            end
            if need2computeδ!(y)
                δy = - δ(z)
                δ(y) .+= unbcast(ᵛ(y), δy)
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
                δ(x) .+= unbcast(ᵛ(x), δx)
            end
            if need2computeδ!(y)
                δy = δ(z) .* ᵛ(x)
                δ(y) .+= unbcast(ᵛ(y), δy)
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
            y⁻¹ = T(1) ./ ᵛ(y)
            y⁻² =  y⁻¹ .* y⁻¹
            if need2computeδ!(x)
                δx = δ(z) .* y⁻¹
                δ(x) .+= unbcast(ᵛ(x), δx)
            end
            if need2computeδ!(y)
                δy = - δ(z) .* ᵛ(x) .* y⁻²
                δ(y) .+= unbcast(ᵛ(y), δy)
            end
            ifNotKeepδThenFreeδ!(z);
        end
        push!(graph.backward, DotDivBackward)
    end
    return z
end
