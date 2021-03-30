# ----------------------------------------
#     Non-linear activation functions
# ----------------------------------------
import Base.abs
import Base.reshape
import Base.vcat
import Base.sind
import Base.sinpi

import Base.exp2
import Base.exp10

import Base.log2
import Base.log10

export abs
export reshape
export sind
export sinpi

export exp2
export exp10

export log2
export log10

export relu, relu!
# -------------------------------------------------------- relu
function relu!(x::AbstractArray)
    @. x = max(0.0, x)
end


function relu(x::AbstractArray)
    O = eltype(x)(0.0)
    return max.(O, x)
end


function relu!(var::Variable{T}) where T
    out = Variable{T}(relu!(var.value), var.backprop)
    if var.backprop
        o2i = var.value .> 0.0
        function reluBackward()
            if need2computeδ!(var)
                var.delta += out.delta .* o2i
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, reluBackward)
    end
    return out
end


function relu(var::Variable{T}) where T
    out = Variable{T}(relu(var.value), var.backprop)
    if var.backprop
        o2i = var.value .> 0.0
        function reluBackward()
            if need2computeδ!(var)
                var.delta += out.delta .* o2i
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, reluBackward)
    end
    return out
end

export relu1, relu1!
# -------------------------------------------------------- relu1
function relu1!(x::AbstractArray)
    @. x = min(1.0, max(0.0, x))
end


function relu1(x::AbstractArray)
    T = eltype(x)
    ONE  = T(1.0)
    ZERO = T(0.0)
    return min.(ONE, max.(ZERO, x))
end


function relu1!(var::Variable{T}) where T
    out = Variable{T}(relu1!(var.value), var.backprop)
    if var.backprop
        o2i = 0.0 .< var.value .< 1.0
        function relu1Backward()
            if need2computeδ!(var)
                var.delta += out.delta .* o2i
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, relu1Backward)
    end
    return out
end


function relu1(var::Variable{T}) where T
    out = Variable{T}(relu1(var.value), var.backprop)
    if var.backprop
        o2i = 0.0 .< var.value .< 1.0
        function relu1Backward()
            if need2computeδ!(var)
                var.delta += out.delta .* o2i
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, relu1Backward)
    end
    return out
end

export relu6, relu6!
# -------------------------------------------------------- relu6
function relu6!(x::AbstractArray)
    @. x = min(6.0, max(0.0, x))
end


function relu6(x::AbstractArray)
    T = eltype(x)
    SIX  = T(6.0)
    ZERO = T(0.0)
    return min.(SIX, max.(ZERO, x))
end


function relu6!(var::Variable{T}) where T
    out = Variable{T}(relu6!(var.value), var.backprop)
    if var.backprop
        o2i = 0.0 .< var.value .< 6.0
        function relu1Backward()
            if need2computeδ!(var)
                var.delta += out.delta .* o2i
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, relu1Backward)
    end
    return out
end


function relu6(var::Variable{T}) where T
    out = Variable{T}(relu6(var.value), var.backprop)
    if var.backprop
        o2i = 0.0 .< var.value .< 6.0
        function relu1Backward()
            if need2computeδ!(var)
                var.delta += out.delta .* o2i
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, relu1Backward)
    end
    return out
end

export line, line!
# -------------------------------------------------------- line
function line!(x::AbstractArray)
    @. x = (-1.0 < x < 1.0) * x
end


function line(x::AbstractArray)
    T = eltype(x)
    ONEP = T( 1.0)
    ONEN = T(-1.0)
    return (ONEN .< x .< ONEP) .* x
end


function line!(var::Variable{T}) where T
    o2i = -1.0 .< var.value .< 1.0
    var.value .*= o2i
    out = Variable{T}(var.value, var.backprop)
    if var.backprop
        function lineBackward()
            if need2computeδ!(var)
                var.delta += out.delta .* o2i
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, lineBackward)
    end
    return out
end


function line(var::Variable{T}) where T
    o2i = -1.0 .< var.value .< 1.0
    out = Variable{T}(var.value .* o2i, var.backprop)
    if var.backprop
        function lineBackward()
            if need2computeδ!(var)
                var.delta += out.delta .* o2i
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, lineBackward)
    end
    return out
end


export hardtanh, hardtanh!
# -------------------------------------------------------- hardtanh
function hardtanh!(x::AbstractArray)
    T = eltype(x)
    ONEP = T( 1.0)
    ONEN = T(-1.0)
    @. x = min(ONEP, max(ONEN, x))
end


function hardtanh(x::AbstractArray)
    T = eltype(x)
    ONEP = T( 1.0)
    ONEN = T(-1.0)
    return min.(ONEP, max.(ONEN, x))
end


function hardtanh!(var::Variable{T}) where T
    out = Variable{T}(hardtanh!(var.value), var.backprop)
    if var.backprop
        o2i = abs(var.value) .< 1.0
        function hardtanhBackward()
            if need2computeδ!(var)
                var.delta += out.delta .* o2i
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, hardtanhBackward)
    end
    return out
end


function hardtanh(var::Variable{T}) where T
    out = Variable{T}(hardtanh(var.value), var.backprop)
    if var.backprop
        o2i = abs(var.value) .< 1.0
        function hardtanhBackward()
            if need2computeδ!(var)
                var.delta += out.delta .* o2i
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, hardtanhBackward)
    end
    return out
end


export leakyrelu, leakyrelu!
# -------------------------------------------------------- leakyrelu
function leakyrelu!(x::AbstractArray)
    ZPONE = eltype(x)(0.1)
    @. x = max(ZPONE * x, x)
end


function leakyrelu(x::AbstractArray)
    ZPONE = eltype(x)(0.1)
    return max.(ZPONE .* x, x)
end


function leakyrelu!(var::Variable{T}) where T
    ZPONE = eltype(var)(0.1)
    tempv = var.value .* ZPONE
    @. var.value = max(var.value, tempv)
    out = Variable{T}(var.value, var.backprop)
    if var.backprop
        mask1 = var.value .> tempv
        mask2 = .!mask1
        function leakyreluBackward()
            if need2computeδ!(var)
                var.delta = (mask1 + ZPONE .* mask2) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, leakyreluBackward)
    end
    return out
end


function leakyrelu(var::Variable{T}) where T
    ZPONE = eltype(var)(0.1)
    tempv = var.value .* ZPONE
    mask1 = var.value .> tempv
    mask2 = .!mask1
    out  = Variable{T}(max.(tempv, var.value), var.backprop)
    if var.backprop
        function leakyreluBackward()
            if need2computeδ!(var)
                var.delta = (mask1 + ZPONE .* mask2) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, leakyreluBackward)
    end
    return out
end


export sigmoid, sigmoid!
# -------------------------------------------------------- sigmoid
function sigmoid!(x::AbstractArray)
    ONE = eltype(x)(1.0)
    @. x = ONE / (ONE + exp(-x))
end


function sigmoid(x::AbstractArray)
    ONE = eltype(x)(1.0)
    return ONE ./ (ONE .+ exp.(-x))
end


function sigmoid!(var::Variable{T}) where T
    out = Variable{T}(sigmoid!(var.value), var.backprop)
    if var.backprop
        ONE = eltype(var)(1.0)
        function sigmoidBackward()
            if need2computeδ!(var)
                var.delta += out.value .* (ONE .- out.value) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, sigmoidBackward)
    end
    return out
end


function sigmoid(var::Variable{T}) where T
    out = Variable{T}(sigmoid(var.value), var.backprop)
    if var.backprop
        ONE = eltype(var)(1.0)
        function sigmoidBackward()
            if need2computeδ!(var)
                var.delta += out.value .* (ONE .- out.value) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, sigmoidBackward)
    end
    return out
end


export swish, swish!
# -------------------------------------------------------- swish
function swish!(x::AbstractArray)
    ONE = eltype(x)(1.0)
    @. x = x / (ONE + exp(-x))
end


function swish(x::AbstractArray)
    ONE = eltype(x)(1.0)
    return  x ./ (ONE .+ exp.(-x))
end


function swish!(var::Variable{T}) where T
    return dotMul(sigmoid(var), var)
end


function swish(var::Variable{T}) where T
    return dotMul(sigmoid(var), var)
end


export softmax

function softmax(x::AbstractArray; dims::Union{Int,NTuple{N,Int}}) where N
    out = exp.(x .- maximum(x, dims=dims))
    SUM = eltype(x)(1.0) ./ sum(out, dims=dims)
    return out .* SUM
end


function softmax(var::Variable{T}; dims::Union{Int,NTuple{N,Int}}) where {T,N}
    out = Variable{T}(softmax(var.value; dims=dims), var.backprop)
    if var.backprop
        function softmaxBackward()
            if need2computeδ!(var)
                ẏy = out.delta .* out.value;
                var.delta += ẏy .- out.value .* sum(ẏy, dims=dims);
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, softmaxBackward)
    end
    return out
end


# -----------------
# 不常用激活函数....
# -----------------
export softplus, softplus! 
function softplus!(x::AbstractArray)
    @. x = log(1.0 + exp(x))
end


function softplus(x::AbstractArray)
    ONE = eltype(x)(1.0)
    return log.( ONE .+ exp.(x) )
end


function softplus!(var::Variable{T}) where T
    out = Variable{T}(softplus(var.value), var.backprop)
    if var.backprop
        ONE = eltype(var)(1.0)
        function softplusBackward()
            if need2computeδ!(var)
                var.delta += out.delta ./ (ONE .+ exp.(-var.value))
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, softplusBackward)
    end
    return out
end


function softplus(var::Variable{T}) where T
    out = Variable{T}(softplus(var.value), var.backprop)
    if var.backprop
        ONE = eltype(var)(1.0)
        function softplusBackward()
            if need2computeδ!(var)
                var.delta += out.delta ./ (ONE .+ exp.(-var.value))
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, softplusBackward)
    end
    return out
end


export exp! 
function exp!(var::Variable{T}) where T
    out = Variable{T}(exp!(var.value), var.backprop)
    if var.backprop
        function expBackward()
            if need2computeδ!(var)
                var.delta += out.value .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, expBackward)
    end
    return out
end


function Base.:exp(var::Variable{T}) where T
    out = Variable{T}(exp(var.value), var.backprop)
    if var.backprop
        function expBackward()
            if need2computeδ!(var)
                var.delta += out.value .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, expBackward)
    end
    return out
end


export log!
function log!(var::Variable{T}) where T
    out = Variable{T}(log(var.value), var.backprop)
    if var.backprop
        function logBackward()
            if need2computeδ!(var)
                var.delta += out.delta ./ var.value
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, logBackward)
    end
    return out
end


function Base.:log(var::Variable{T}) where T
    out = Variable{T}(log(var.value), var.backprop)
    if var.backprop
        function logBackward()
            if need2computeδ!(var)
                var.delta += out.delta ./ var.value
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, logBackward)
    end
    return out
end


export abs!
function abs!(x::AbstractArray)
    @. x = abs(x)
end


function Base.:abs(x::AbstractArray)
    return abs.(x)
end


function abs!(var::Variable{T}) where T
    out = Variable{T}(abs(var.value), var.backprop)
    if var.backprop
        function absBackward()
            if need2computeδ!(var)
                var.delta += sign.(var.value) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, absBackward)
    end
    return out
end


function Base.:abs(var::Variable{T}) where T
    out = Variable{T}(abs(var.value), var.backprop)
    if var.backprop
        function absBackward()
            if need2computeδ!(var)
                var.delta += sign.(var.value) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, absBackward)
    end
    return out
end


function Base.:reshape(var::Variable{T}, newsize) where T
    out = Variable{T}( reshape(var.value, newsize), var.backprop )
    if var.backprop
        function reshapeBackward()
            if need2computeδ!(var)
                var.delta += reshape(out.delta, var.shape)
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, reshapeBackward)
    end
    return out
end


function exp2!(x::AbstractArray)
    @. x = exp2(x)
end


function Base.:exp2(x::AbstractArray)
    return exp2.(x)
end


function exp2!(var::Variable{T}) where T
    # exp2 represents y = 2^x
    out = Variable{T}(exp2!(var.value), var.backprop)
    if var.backprop
        TWO = eltype(var)(2.0)
        function exp2Backward()
            if need2computeδ!(var)
                var.delta += log(TWO) .* out.value .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, exp2Backward)
    end
    return out
end


function Base.:exp2(var::Variable{T}) where T
    # EXP2 represents y = 2^x
    out = Variable{T}(exp2(var.value), var.backprop)
    if var.backprop
        TWO = eltype(var)(2.0)
        function exp2Backward()
            if need2computeδ!(var)
                var.delta += log(TWO) .* out.value .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, exp2Backward)
    end
    return out
end


function exp10!(x::AbstractArray)
    @. x = exp10(x)
end


function Base.:exp10(x::AbstractArray)
    return exp10.(x)
end


function exp10!(var::Variable{T}) where T
    # EXP10 represents y = 10^x
    out = Variable{T}(exp10!(var.value), var.backprop)
    if var.backprop
        TEN = eltype(var)(10.0)
        function exp10Backward()
            if need2computeδ!(var)
                var.delta += log(TEN) .* out.value .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, exp10Backward)
    end
    return out
end


function Base.:exp10(var::Variable{T}) where T
    # EXP10 represents y = 10^x
    out = Variable{T}(exp10(var.value), var.backprop)
    if var.backprop
        TEN = eltype(var)(10.0)
        function exp10Backward()
            if need2computeδ!(var)
                var.delta += log(TEN) .* out.value .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, exp10Backward)
    end
    return out
end


function log2!(x::AbstractArray)
    @. x = log2(x)
end


function Base.:log2(x::AbstractArray)
    return log2.(x)
end


function log2!(var::Variable{T}) where T
    out = Variable{T}(log2(var.value), var.backprop)
    if var.backprop
        TWO = eltype(var)(2.0)
        function log2Backward()
            if need2computeδ!(var)
                var.delta += out.delta ./ (log(TWO) .* var.value)
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, log2Backward)
    end
    return out
end


function Base.:log2(var::Variable{T}) where T
    out = Variable{T}(log2(var.value), var.backprop)
    if var.backprop
        TWO = eltype(var)(2.0)
        function log2Backward()
            if need2computeδ!(var)
                var.delta += out.delta ./ (log(TWO) .* var.value)
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, log2Backward)
    end
    return out
end


function log10!(x::AbstractArray)
    @. x = log10(x)
end


function Base.:log10(x::AbstractArray)
    return log10.(x)
end


function log10!(var::Variable{T}) where T
    out = Variable{T}(log10(var.value), var.backprop)
    if var.backprop
        TEN = eltype(var)(10.0)
        function log10Backward()
            if need2computeδ!(var)
                var.delta += out.delta ./ (log(TEN) .* var.value)
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, log10Backward)
    end
    return out
end


function Base.:log10(var::Variable{T}) where T
    out = Variable{T}(log10(var.value), var.backprop)
    if var.backprop
        TEN = eltype(var)(10.0)
        function log10Backward()
            if need2computeδ!(var)
                var.delta += out.delta ./ (log(TEN) .* var.value)
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, log10Backward)
    end
    return out
end


function sec!(var::Variable{T}) where T
    # SEC represents y = sec(x)
    out = Variable{T}(sec(var.value), var.backprop)
    if var.backprop
        function secBackward()
            if need2computeδ!(var)
                var.delta += out.delta .* out.value .* tan.(var.value)
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, secBackward)
    end
    return out
end


function Base.:sec(var::Variable{T}) where T
    # SEC represents y = sec(x)
    out = Variable{T}(sec(var.value), var.backprop)
    if var.backprop
        function secBackward()
            if need2computeδ!(var)
                var.delta += out.delta .* out.value .* tan.(var.value)
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, secBackward)
    end
    return out
end


function sqrt!(var::Variable{T}) where T
    out = Variable{T}(sqrt!(var.value), var.backprop)
    if var.backprop
        TOO = eltype(var)
        TWO = TOO(2.000)
        EPS = TOO(1e-38)
        function sqrtBackward()
            if need2computeδ!(var)
                var.delta += out.delta ./ (TWO .* (out.value .+ EPS))
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, sqrtBackward)
    end
    return out
end


function Base.:sqrt(var::Variable{T}) where T
    out = Variable{T}(sqrt(var.value), var.backprop)
    if var.backprop
        TOO = eltype(var)
        TWO = TOO(2.000)
        EPS = TOO(1e-38)
        function sqrtBackward()
            if need2computeδ!(var)
                var.delta += out.delta ./ (TWO .* (out.value .+ EPS))
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, sqrtBackward)
    end
    return out
end


# -- tan serials --
export tan!
function tan!(var::Variable{T}) where T
    out = Variable{T}(tan!(var.value), var.backprop)
    if var.backprop
        TOO = eltype(var)
        ONE = TOO(1.0)
        TWO = TOO(2.0)
        function tanBackward()
            if need2computeδ!(var)
                var.delta += (ONE .+ out.value.^TWO) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, tanBackward)
    end
    return out
end


function Base.:tan(var::Variable{T}) where T
    out = Variable{T}(tan(var.value), var.backprop)
    if var.backprop
        TOO = eltype(var)
        ONE = TOO(1.0)
        TWO = TOO(2.0)
        function tanBackward()
            if need2computeδ!(var)
                var.delta += (ONE .+ out.value.^TWO) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, tanBackward)
    end
    return out
end


export tand!
function tand!(x::AbstractArray)
    @. x = tand(x)
end


function Base.:tand(x::AbstractArray)
    return tand.(x)
end


function tand!(var::Variable{T}) where T
    out = Variable{T}(tand!(var.value), var.backprop)
    if var.backprop
        TOO = eltype(var)
        ONE = TOO(1.0)
        TWO = TOO(2.0)
        DPI = TOO(pi/180)
        function tandBackward()
            if need2computeδ!(var)
                var.delta += DPI .* (ONE .+ out.value.^TWO) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, tandBackward)
    end
    return out
end


function Base.:tand(var::Variable{T}) where T
    out = Variable{T}(tand(var.value), var.backprop)
    if var.backprop
        TOO = eltype(var)
        ONE = TOO(1.0)
        TWO = TOO(2.0)
        DPI = TOO(pi/180)
        function tandBackward()
            if need2computeδ!(var)
                var.delta += DPI .* (ONE .+ out.value.^TWO) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, tandBackward)
    end
    return out
end


export tanh!
function tanh!(var::Variable{T}) where T
    out = Variable{T}(tanh!(var.value), var.backprop)
    if var.backprop
        TOO = eltype(var)
        ONE = TOO(1.0)
        TWO = TOO(2.0)
        function tanhBackward()
            if need2computeδ!(var)
                var.delta += (ONE .- out.value.^TWO) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, tanhBackward)
    end
    return out
end


function Base.:tanh(var::Variable{T}) where T
    out = Variable{T}(tanh(var.value), var.backprop)
    if var.backprop
        TOO = eltype(var)
        ONE = TOO(1.0)
        TWO = TOO(2.0)
        function tanhBackward()
            if need2computeδ!(var)
                var.delta += (ONE .- out.value.^TWO) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, tanhBackward)
    end
    return out
end


export tanhshrink, tanhshrink!
function tanhshrink!(x::AbstractArray)
    @. x = x - tanh(x)
end


function tanhshrink(x::AbstractArray)
    return  x - tanh(x)
end


function tanhshrink!(var::Variable{T}) where T
    return var - tanh(var)
end


function tanhshrink(var::Variable{T}) where T
    return var - tanh(var)
end


# # -- sin serials --
export sin!
function sin!(var::Variable{T}) where T
    out = Variable{T}(sin(var.value), var.backprop)
    if var.backprop
        function sinBackward()
            if need2computeδ!(var)
                var.delta += cos.(var.value) .* out.delta;
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, sinBackward)
    end
    return out
end


function Base.:sin(var::Variable{T}) where T
    out = Variable{T}(sin(var.value), var.backprop)
    if var.backprop
        function sinBackward()
            if need2computeδ!(var)
                var.delta += cos.(var.value) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, sinBackward)
    end
    return out
end


export sinc!
function sinc!(x::AbstractArray)
    @. x = sinc(x)
end


function Base.:sinc(x::AbstractArray)
    return sinc.(x)
end


function sinc!(var::Variable{T}) where T
    # sinc represents y = sin(pi*x)/(pi*x)
    out = Variable{T}(sinc(var.value), var.backprop)
    if var.backprop
        function sincBackward()
            if need2computeδ!(var)
                var.delta += cosc.(var.value) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, sincBackward)
    end
    return out
end


function Base.:sinc(var::Variable{T}) where T
    # sinc represents y = sin(pi*x)/(pi*x)
    out = Variable{T}(sinc(var.value), var.backprop)
    if var.backprop
        function sincBackward()
            if need2computeδ!(var)
                var.delta += cosc.(var.value) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, sincBackward)
    end
    return out
end


export sind!
function sind!(x::AbstractArray)
    @. x = sind(x)
end


function Base.:sind(x::AbstractArray)
    return sind.(x)
end


function sind!(var::Variable{T}) where T
    out = Variable{T}(sind(var.value), var.backprop)
    if var.backprop
        DPI = eltype(var)(pi/180)
        function sindBackward()
            if need2computeδ!(var)
                var.delta += DPI .* cosd.(var.value) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, sindBackward)
    end
    return out
end


function Base.:sind(var::Variable{T}) where T
    out = Variable{T}(sind(var.value), var.backprop)
    if var.backprop
        DPI = eltype(var)(pi/180)
        function sindBackward()
            if need2computeδ!(var)
                var.delta += DPI .* cosd.(var.value) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, sindBackward)
    end
    return out
end


export sinpi!
function sinpi!(x::AbstractArray)
    @. x = sinpi(x)
end


function Base.:sinpi(x::AbstractArray)
    return sinpi.(x)
end


function sinpi!(var::Variable{T}) where T
    out = Variable{T}(sinpi(var.value), var.backprop)
    if var.backprop
        PI = eltype(var)(pi)
        function sinpiBackward()
            if need2computeδ!(var)
                var.delta += PI .* cospi.(var.value) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, sinpiBackward)
    end
    return out
end


function Base.:sinpi(var::Variable{T}) where T
    out = Variable{T}(sinpi(var.value), var.backprop)
    if var.backprop
        PI = eltype(var)(pi)
        function sinpiBackward()
            if need2computeδ!(var)
                var.delta += PI .* cospi.(var.value) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, sinpiBackward)
    end
    return out
end


export linearsin,linearsin!
function linearsin!(x::AbstractArray)
    @. x = sin(x) + x
end


function linearsin(x::AbstractArray)
    return sin(x) + x
end


function linearsin!(var::Variable{T}) where T
    return sin(var) + var
end


function linearsin(var::Variable{T}) where T
    return sin(var) + var
end


export cos!
function cos!(var::Variable{T}) where T
    out = Variable{T}(cos(var.value), var.backprop)
    if var.backprop
        function cosBackward()
            if need2computeδ!(var)
                var.delta += - sin.(var.value) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, cosBackward)
    end
    return out
end


function Base.:cos(var::Variable{T}) where T
    out = Variable{T}(cos(var.value), var.backprop)
    if var.backprop
        function cosBackward()
            if need2computeδ!(var)
                var.delta += - sin.(var.value) .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, cosBackward)
    end
    return out
end


export inv!
function inv!(var::Variable{T}) where T
    out = Variable{T}(inv!(var.value), var.backprop)
    if var.backprop
        function invBackward()
            if need2computeδ!(var)
                var.delta += - out.delta .* out.value .* out.value;
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, invBackward)
    end
    return out
end


function inv(var::Variable{T}) where T
    out = Variable{T}(inv(var.value), var.backprop)
    if var.backprop
        function invBackward()
            if need2computeδ!(var)
                var.delta += - out.delta .* out.value .* out.value
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, invBackward)
    end
    return out
end
