function Zero(::Type{T}, x...) where T
    return fill!(T(undef, x...), 0.0)
end


"""
    mutable struct Variable{T}
# Fields
- `value::T`                : value in forward
- `delta::Union{Nothing,T}` : gradients collected in backprop
- `shape::Tuple`            : shape of `value`
- `keepsgrad::Bool`         : whether keeps grad after backprop
- `isleaf::Bool`            : whether leaf node
"""
mutable struct Variable{T}
    value::T
    delta::Union{Nothing,T}
    shape::Tuple
    backprop::Bool      # 是否反向传播，依赖的子节点都跟其一样
    keepsgrad::Bool     # 是否保留梯度，可能用于训练或者其他
    isleaf::Bool        # 可训练的，代表可学习的参数
    function Variable{T}(x, backprop::Bool=true,
                            keepsgrad::Bool=false,
                            isleaf::Bool=false) where T
        s = size(x)
        δ = nothing
        new{T}(x, δ, s, backprop, keepsgrad, isleaf)
    end
end


# Convenient abstract-type-specilized constructors for data on GPU/CPU/xPU etc....
function Variable(x; backprop::Bool=true,
                     keepsgrad::Bool=false,
                     type::Type=Array{Float32})
    isleaf = true    # any user defined Variable is a leaf Variable
    return Variable{type}(x, backprop, keepsgrad, isleaf)
end


# pretty printing
function Base.show(io::IO, var::Variable{T}) where T
    if  var.isleaf println(cyan("\n≡≡≡ Leaf Variable ≡≡≡")) end
    if !var.isleaf println(cyan("\n≡≡≡ None Leaf Variable ≡≡≡")) end

    print(blue("\nvalue is "))
    display(var.value)
    print(green("\ndelta is "))
    display(var.delta)
end


function zeroDelta(var::Variable{T}) where T
    # 要切断某些反向传播路径的时候将其初始化为零
    if var.delta==nothing
        var.delta = Zero(T, var.shape);
    end
end


function need2computeδ!(var::Variable{T}) where T
    # 需要反向传播的时候就需要初始化
    if !(var.isleaf && !var.keepsgrad)
        if var.delta==nothing
            var.delta = Zero(T, var.shape);
        end
        return true
    else
        return false
    end
end


function ifNotKeepδThenFreeδ!(var::Variable{T}) where T
    if !var.keepsgrad
        var.delta = nothing
    end
end


import Base.size
import Base.ndims
import Base.length
import Base.strides
import Base.eltype
import Base.similar
import Base.getindex
import Base.setindex!
import Base.copy
import Base.deepcopy


Base.size(x::Variable)           =    size(x.value)
Base.size(x::Variable, dim::Int) =    size(x.value, dim)
Base.ndims(x::Variable)          =   ndims(x.value)
Base.length(x::Variable)         =  length(x.value)
Base.strides(x::Variable)        = strides(x.value)
Base.eltype(x::Variable)         =  eltype(x.value)
Base.getindex(x::Variable,     k...)  =    x.value[k...]
Base.setindex!(x::Variable, v, k...)  =   (x.value[k...] = v)
Base.similar(x::Variable{T})  where T = Variable{T}( similar(x.value), x.backprop, x.keepsgrad, x.isleaf)
Base.copy(x::Variable{T})     where T = Variable{T}(    copy(x.value), x.backprop, x.keepsgrad, x.isleaf)
Base.deepcopy(x::Variable{T}) where T = Variable{T}(deepcopy(x.value), x.backprop, x.keepsgrad, x.isleaf)


function (v::Variable)(i...)
    if v.delta!==nothing
        return v.delta[i...]
    else
        return nothing
    end
end


"""
    to(type::Type, var::Variable{T}, show::Bool=false) where T -> Variable{type}
Type conversions between data types in Variable. (like from Array{Float64} to CuArray{Float32} )
## Example
`x = Variable(ones(1,4),type=Array{Float64})`

`y = to(CuArray{Float16},x,true)`
"""
function to(type::Type, var::Variable{T}, show::Bool=false) where T
    if type <: T || type >: T
        COLOR_TYPE = yellow(T)
        COLOR_DEST = yellow(type)
        @info "you don't need to convert data from $COLOR_TYPE to $COLOR_DEST.\n"
        return var
    else
        if !show
            return Variable{type}(var.value, var.backprop, var.keepsgrad, var.isleaf)
        else
            COLOR_TYPE = green(T)
            COLOR_DEST = green(type)
            @info "converting data from $COLOR_TYPE to $COLOR_DEST.\n"
            return Variable{type}(var.value, var.backprop, var.keepsgrad, var.isleaf)
        end
    end
end


export Variable
export zeroDelta
export Zero
export to

export need2computeδ!
export ifNotKeepδThenFreeδ!
