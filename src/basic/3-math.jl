# ------------------------------------------------
# basic math operators like dot mul,dot add,etc ..
# ------------------------------------------------
import Base.+
import Base.-
import Base.*
import Base.^

export dotAdd
export dotMul
export matAddVec
export matMulVec

function Base.:+(var::Variable{T}, constant) where T
    # a matrix add a constant element by element
    constant = eltype(var.value)(constant)
    out = Variable{T}(var.value .+ constant, var.backprop)
    if var.backprop
        function matAddScalarBackward()
            if need2computeδ!(var) var.delta += out.delta end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, matAddScalarBackward)
    end
    return out
end


function Base.:+(constant, var::Variable{T}) where T
    return var + constant;
end


function Base.:-(var::Variable{T}, constant) where T
    # a matrix minus a constant element by element
    constant = eltype(var.value)(constant)
    out = Variable{T}(var.value .- constant, var.backprop)
    if var.backprop
        function matMinusScalarBackward()
            if need2computeδ!(var) var.delta += out.delta end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, matMinusScalarBackward)
    end
    return out
end


function Base.:-(constant, var::Variable{T}) where T
    # a matrix minus a constant element by element
    constant = eltype(var.value)(constant)
    out = Variable{T}(constant .- var.value, var.backprop)
    if var.backprop
        function scalarMinusMatBackward()
            if need2computeδ!(var) var.delta -= out.delta end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, scalarMinusMatBackward)
    end
    return out
end


function Base.:*(var::Variable{T}, constant) where T
    # a matrix multiplies a constant element by element
    constant = eltype(var.value)(constant)
    out = Variable{T}(var.value .* constant, var.backprop)
    if var.backprop
        function matMulScalarBackward()
            if need2computeδ!(var) var.delta += out.delta .* constant end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, matMulScalarBackward)
    end
    return out
end


function Base.:*(constant, var::Variable{T}) where T
    return var * constant
end


function Base.:^(var::Variable{T}, n::Int) where T
    # 矩阵、列向量与常数按元素做幂指数运算
    n = eltype(var)(n)
    out = Variable{T}(var.value .^ n, var.backprop)
    if var.backprop
        function powerBackward()
            if need2computeδ!(var)
                var.delta += n .* out.value ./ var.value .* out.delta;
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, powerBackward)
    end
    return out
end


function Base.:+(var1::Variable{T1}, var2::Variable{T2}) where {T1,T2}
     # a matrix add a matrix element by element
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    @assert (var1.shape == var2.shape) "2 inputs shall be the same size"
    backprop = (var1.backprop || var2.backprop)
    out = Variable{T}(var1.value + var2.value, backprop)
    if backprop
        function add2varBackward()
            if need2computeδ!(var1) var1.delta += out.delta end
            if need2computeδ!(var2) var2.delta += out.delta end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, add2varBackward)
    end
    return out
end


function Base.:-(var1::Variable{T1}, var2::Variable{T2}) where {T1,T2}
    # a matrix minus a matrix element by element
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    @assert (var1.shape == var2.shape) "2 inputs shall be the same size"
    backprop = (var1.backprop || var2.backprop)
    out = Variable{T}(var1.value - var2.value, backprop)
    if backprop
        function minus2varBackward()
            if need2computeδ!(var1) var1.delta += out.delta end
            if need2computeδ!(var2) var2.delta -= out.delta end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, minus2varBackward)
    end
    return out
end


"""
    dotAdd(var1::Variable{T1}, var2::Variable{T2}) where {T1,T2}
    a tensor add a tensor element by element
"""
function dotAdd(var1::Variable{T1}, var2::Variable{T2}) where {T1,T2}
    # a tensor add a tensor element by element
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    @assert (var1.shape == var2.shape) "2 inputs shall be the same size"
    backprop = (var1.backprop || var2.backprop)
    out = Variable{T}(var1.value .+ var2.value, backprop)
    if backprop
        function dotAddBackward()
            if need2computeδ!(var1) var1.delta += out.delta end
            if need2computeδ!(var2) var2.delta += out.delta end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, dotAddBackward)
    end
    return out
end


"""
    dotMul(var1::Variable{T1}, var2::Variable{T2}) where {T1,T2}
    a tensor multiplies a tensor element by element
"""
function dotMul(var1::Variable{T1}, var2::Variable{T2}) where {T1,T2}
    # a tensor multiplies a tensor element by element
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    @assert (var1.shape == var2.shape) "2 inputs shall be the same size"
    backprop = (var1.backprop || var2.backprop)
    out = Variable{T}(var1.value .* var2.value, backprop)
    if backprop
        function dotMulBackward()
            if need2computeδ!(var1) var1.delta += out.delta .* var2.value end
            if need2computeδ!(var2) var2.delta += out.delta .* var1.value end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, dotMulBackward)
    end
    return out
end


function Base.:*(var1::Variable{T1}, var2::Variable{T2}) where {T1,T2}
    # matrix var1 multiplies matrix var2
    # 矩阵相乘 Y[i,j] = sum(W[i,k]*X[k,j],k=...)
    # var1 -- 权重矩阵
    # var2 -- n个输入列向量组成的矩阵
    # out  -- n个输出列向量组成的矩阵
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    backprop = (var1.backprop || var2.backprop)
    out = Variable{T}(var1.value * var2.value, backprop)
    if backprop
        function matMulBackward()
            if need2computeδ!(var1) var1.delta += out.delta * var2.value' end
            if need2computeδ!(var2) var2.delta += var1.value' * out.delta end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, matMulBackward)
    end
    return out
end


"""
    matAddVec(var1::Variable{T1}, var2::Variable{T2}) where {T1,T2}
    a matrix tensor `var1` adds a vector tensor `var2`
"""
function matAddVec(var1::Variable{T1}, var2::Variable{T2}) where {T1,T2}
    # var1 -- 充当和节点，非学习的参数
    # var2 -- 偏置列向量，要学习的参数
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    @assert (var1.shape[1]==var2.shape[1] && var2.shape[2]==1)
    backprop = (var1.backprop || var2.backprop)
    out = Variable{T}(var1.value .+ var2.value, backprop)
    if backprop
        function matAddVecBackward()
            if need2computeδ!(var1)
                var1.delta += out.delta
            end
            if need2computeδ!(var2)
                var2.delta += sum(out.delta, dims=2)
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, matAddVecBackward)
    end
    return out
end


"""
    matAddVec(var1::Variable{T1}, var2::Variable{T2}) where {T1,T2}
    a matrix tensor `var1` multiplies a vector tensor `var2`
"""
function matMulVec(var1::Variable{T1}, var2::Variable{T2}) where {T1,T2}
    # var1 -- 一般充当激活节点，非网络需要学习的参数
    # var2 -- 列向量，循环权重，是网络需要学习的参数
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T1 : T2
    @assert (var1.shape[1]==var2.shape[1] && var2.shape[2]==1)
    backprop = (var1.backprop || var2.backprop)
    out = Variable{T}(var1.value .* var2.value, backprop)
    if backprop
        function matMulVecBackward()
            if need2computeδ!(var1) var1.delta += out.delta .* var2.value end
            if need2computeδ!(var2) var2.delta += sum(out.delta .* var1.value, dims=2) end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, matMulVecBackward)
    end
    return out
end
