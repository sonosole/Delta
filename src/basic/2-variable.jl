"""
    Zeros(::Type{T}, shape...) where T
return an all-zero-elements-array of type T which has shape `shape...`

# Example
    julia> Zeros(Array{Float64}, 2, 5)
    2×5 Array{Float64,2}:
     0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0
 """
function Zeros(::Type{T}, shape...) where T
    return fill!(T(undef, shape...), 0.0)
end


"""
    Ones(::Type{T}, shape...) where T
return an all-one-elements-array of type T which has shape `shape...`

# Example
    julia> 7Ones(Array{Float64}, 2, 5)
    2×5 Array{Float64,2}:
     7.0  7.0  7.0  7.0  7.0
     7.0  7.0  7.0  7.0  7.0
 """
function Ones(::Type{T}, shape...) where T
    return fill!(T(undef, shape...), 1.0)
end


"""
    mutable struct Variable{T}
# Fields
+ `value::T`                : value in forward
+ `delta::Union{Nothing,T}` : gradients collected in backprop
+ `shape::Tuple`            : shape of `value`
+ `keepsgrad::Bool`         : whether keeps grad after backprop
+ `isleaf::Bool`            : whether leaf node
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


function clone(x::Variable; type::Type=Array{Float32})
    return Variable{type}(type(x.value), x.backprop, x.keepsgrad, x.isleaf)
end


function zeroDelta(var::Variable{T}) where T
    # 要切断某些反向传播路径的时候将其初始化为零
    if var.delta === nothing
        var.delta = Zeros(T, var.shape);
    end
end


function need2computeδ!(var::Variable{T}) where T
    # 需要反向传播的时候就需要初始化
    if !(var.isleaf && !var.keepsgrad)
        if var.delta===nothing
            var.delta = Zeros(T, var.shape);
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



Base.sizeof(x::Variable)         =  sizeof(x.value)
Base.size(x::Variable)           =    size(x.value)
Base.size(x::Variable, dim::Int) =    size(x.value, dim)
Base.ndims(x::Variable)          =   ndims(x.value)
Base.length(x::Variable)         =  length(x.value)
Base.strides(x::Variable)        = strides(x.value)
Base.eltype(x::Variable)         =  eltype(x.value)
Base.similar(x::Variable{T})  where T = Variable{T}( similar(x.value), x.backprop, x.keepsgrad, x.isleaf)
Base.copy(x::Variable{T})     where T = Variable{T}(    copy(x.value), x.backprop, x.keepsgrad, x.isleaf)
Base.deepcopy(x::Variable{T}) where T = Variable{T}(deepcopy(x.value), x.backprop, x.keepsgrad, x.isleaf)

Base.setindex!(x::Variable, v::Number,        k...) = (x.value[k...] .= v)
Base.setindex!(x::Variable, v::AbstractArray, k...) = (x.value[k...]  = v)

function Base.getindex(x::Variable{T}, k...) where T
    y = Variable{T}(x.value[k...], x.backprop, x.keepsgrad, x.isleaf)
    if x.backprop
        function getindexBackward()
            if need2computeδ!(x)
                x.delta[k...] .+= y.delta
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, getindexBackward)
    end
    return y
end


function (v::Variable)(i...)
    if v.delta ≠ nothing
        return v.delta[i...]
    else
        return nothing
    end
end


"""
    to(type::Type, var::Variable{T}, show::Bool=false) where T -> Variable{type}
Type conversions between data types in `Variable`. (like from Array{Float64} to CuArray{Float32} )
# Example
    x = Variable(ones(1,4), type=Array{Float64})
    y = to(CuArray{Float16}, x, true)
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


"""
    to(type::Type, vars::Vector{Variable}, show::Bool=false)
Type conversions between data types in `Vector{Variable}`
# Example
    julia> v = Vector{Variable}(undef,2);
    julia> v[1] = Variable(4ones(2,4),type=Array{Float64});
    julia> v[2] = Variable(3ones(2,3),type=Array{Float64});

    julia> typeof(v[1])
    Variable{Array{Float64,N} where N}

    julia> v = to(Array{Float32},v);

    julia> typeof(v[1])
    Variable{Array{Float32,N} where N}
"""
function to(type::Type, vars::Vector{Variable}, show::Bool=false)
    for i = 1:length(vars)
        vars[i] = to(type, vars[i], show);
    end
    return vars
end


function to!(type::Type, vars::Vector{Variable}, show::Bool=false)
    vars = to(type, vars, show)
    return nothing
end


export Variable
export zeroDelta
export Zeros, Ones
export to, to!
export clone
export need2computeδ!
export ifNotKeepδThenFreeδ!
export elsizeof
export value, delta, ᵛ, ᵟ, δ

export XVariable, VarOrNil
const  XVariable = Tuple{Char, Variable}
const  VarOrNil  = Union{Variable, Nothing}

# pretty printing
function Base.show(io::IO, xvar::XVariable)
    c, var = xvar
    if  var.isleaf println(cyan("\n≡≡≡ Leaf Variable ($c) ≡≡≡")) end
    if !var.isleaf println(cyan("\n≡≡≡ None Leaf Variable ≡≡≡")) end

    print(blue("\nvalue is "))
    display(var.value)
    print(green("\ndelta is "))
    display(var.delta)
end


elsizeof(x::Variable) = sizeof(eltype(x))


# lazy showing way of Variable's main vars
@inline ᵛ(x::Variable) = x.value
@inline ᵟ(x::Variable) = x.delta
@inline δ(x::Variable) = x.delta
@inline value(x::Variable) = x.value
@inline delta(x::Variable) = x.delta
