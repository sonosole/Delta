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


function relu!(x::Variable{T}) where T
    y = Variable{T}(relu!(ᵛ(x)), x.backprop)
    if x.backprop
        ∇ = ᵛ(x) .> 0.0
        function reluBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* ∇
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, reluBackward)
    end
    return y
end


function relu(x::Variable{T}) where T
    y = Variable{T}(relu(ᵛ(x)), x.backprop)
    if x.backprop
        ∇ = ᵛ(x) .> 0.0
        function reluBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* ∇
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, reluBackward)
    end
    return y
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


function relu1!(x::Variable{T}) where T
    y = Variable{T}(relu1!(ᵛ(x)), x.backprop)
    if x.backprop
        ∇ = 0.0 .< ᵛ(x) .< 1.0
        function relu1Backward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* ∇
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, relu1Backward)
    end
    return y
end


function relu1(x::Variable{T}) where T
    y = Variable{T}(relu1(ᵛ(x)), x.backprop)
    if x.backprop
        ∇ = 0.0 .< ᵛ(x) .< 1.0
        function relu1Backward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* ∇
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, relu1Backward)
    end
    return y
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


function relu6!(x::Variable{T}) where T
    y = Variable{T}(relu6!(ᵛ(x)), x.backprop)
    if x.backprop
        ∇ = 0.0 .< ᵛ(x) .< 6.0
        function relu1Backward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* ∇
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, relu1Backward)
    end
    return y
end


function relu6(x::Variable{T}) where T
    y = Variable{T}(relu6(ᵛ(x)), x.backprop)
    if x.backprop
        ∇ = 0.0 .< ᵛ(x) .< 6.0
        function relu1Backward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* ∇
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, relu1Backward)
    end
    return y
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


function line!(x::Variable{T}) where T
    ∇ = -1.0 .< ᵛ(x) .< 1.0
    ᵛ(x) .*= ∇
    y = Variable{T}(ᵛ(x), x.backprop)
    if x.backprop
        function lineBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* ∇
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, lineBackward)
    end
    return y
end


function line(x::Variable{T}) where T
    ∇ = -1.0f0 .< ᵛ(x) .< 1.0f0
    y = Variable{T}(ᵛ(x) .* ∇, x.backprop)
    if x.backprop
        function lineBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* ∇
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, lineBackward)
    end
    return y
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


function hardtanh!(x::Variable{T}) where T
    y = Variable{T}(hardtanh!(ᵛ(x)), x.backprop)
    if x.backprop
        ∇ = abs(ᵛ(x)) .< 1.0f0
        function hardtanhBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* ∇
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, hardtanhBackward)
    end
    return y
end


function hardtanh(x::Variable{T}) where T
    y = Variable{T}(hardtanh(ᵛ(x)), x.backprop)
    if x.backprop
        ∇ = abs(ᵛ(x)) .< 1.0
        function hardtanhBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* ∇
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, hardtanhBackward)
    end
    return y
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


function leakyrelu!(x::Variable{T}) where T
    ZPONE = eltype(x)(0.1)
    tempv = ᵛ(x) .* ZPONE
    ᵛ(x) .= max.(ᵛ(x), tempv)
    y = Variable{T}(ᵛ(x), x.backprop)
    if x.backprop
        mask1 = ᵛ(x) .> tempv
        mask2 = .!mask1
        function leakyreluBackward()
            if need2computeδ!(x)
                δ(x) .+= (mask1 .+ ZPONE .* mask2) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, leakyreluBackward)
    end
    return y
end


function leakyrelu(x::Variable{T}) where T
    ZPONE = eltype(x)(0.1)
    tempv = ᵛ(x) .* ZPONE
    mask1 = ᵛ(x) .> tempv
    mask2 = .!mask1
    y = Variable{T}(max.(tempv, ᵛ(x)), x.backprop)
    if x.backprop
        function leakyreluBackward()
            if need2computeδ!(x)
                δ(x) .+= (mask1 + ZPONE .* mask2) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, leakyreluBackward)
    end
    return y
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


function sigmoid!(x::Variable{T}) where T
    y = Variable{T}(sigmoid!(ᵛ(x)), x.backprop)
    if x.backprop
        ONE = eltype(x)(1)
        function sigmoidBackward()
            if need2computeδ!(x)
                δ(x) .+= ᵛ(y) .* (ONE .- ᵛ(y)) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, sigmoidBackward)
    end
    return y
end


function sigmoid(x::Variable{T}) where T
    y = Variable{T}(sigmoid(ᵛ(x)), x.backprop)
    if x.backprop
        ONE = eltype(x)(1.0)
        function sigmoidBackward()
            if need2computeδ!(x)
                δ(x) .+= ᵛ(y) .* (ONE .- ᵛ(y)) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, sigmoidBackward)
    end
    return y
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


function swish!(x::Variable{T}) where T
    return dotMul(sigmoid(x), x)
end


function swish(x::Variable{T}) where T
    return dotMul(sigmoid(x), x)
end


export softmax

function softmax(x::AbstractArray; dims::Union{Int,NTuple{N,Int}}) where N
    y = exp.(x .- maximum(x, dims=dims))
    SUM = eltype(x)(1.0) ./ sum(y, dims=dims)
    return y .* SUM
end


function softmax(x::Variable{T}; dims::Union{Int,NTuple{N,Int}}) where {T,N}
    y = Variable{T}(softmax(ᵛ(x); dims=dims), x.backprop)
    if x.backprop
        function softmaxBackward()
            if need2computeδ!(x)
                ẏy = δ(y) .* ᵛ(y);
                δ(x) .+= ẏy .- ᵛ(y) .* sum(ẏy, dims=dims);
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, softmaxBackward)
    end
    return y
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


function softplus!(x::Variable{T}) where T
    y = Variable{T}(softplus(ᵛ(x)), x.backprop)
    if x.backprop
        ONE = eltype(x)(1.0)
        function softplusBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ (ONE .+ exp.( - ᵛ(x) ))
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, softplusBackward)
    end
    return y
end


function softplus(x::Variable{T}) where T
    y = Variable{T}(softplus(ᵛ(x)), x.backprop)
    if x.backprop
        ONE = eltype(x)(1.0)
        function softplusBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ (ONE .+ exp.( - ᵛ(x) ))
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, softplusBackward)
    end
    return y
end


export exp!
function exp!(x::Variable{T}) where T
    y = Variable{T}(exp!(ᵛ(x)), x.backprop)
    if x.backprop
        function expBackward()
            if need2computeδ!(x)
                δ(x) .+= ᵛ(y) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, expBackward)
    end
    return y
end


function Base.:exp(x::Variable{T}) where T
    y = Variable{T}(exp(ᵛ(x)), x.backprop)
    if x.backprop
        function expBackward()
            if need2computeδ!(x)
                δ(x) .+= ᵛ(y) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, expBackward)
    end
    return y
end


export log!
function log!(x::Variable{T}) where T
    y = Variable{T}(log(ᵛ(x)), x.backprop)
    if x.backprop
        function logBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ ᵛ(x)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, logBackward)
    end
    return y
end


function Base.:log(x::Variable{T}) where T
    y = Variable{T}(log(ᵛ(x)), x.backprop)
    if x.backprop
        function logBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ ᵛ(x)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, logBackward)
    end
    return y
end


export abs!
function abs!(x::AbstractArray)
    @. x = abs(x)
end


function Base.:abs(x::AbstractArray)
    return abs.(x)
end


function abs!(x::Variable{T}) where T
    y = Variable{T}(abs(ᵛ(x)), x.backprop)
    if x.backprop
        function absBackward()
            if need2computeδ!(x)
                δ(x) .+= sign.(ᵛ(x)) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, absBackward)
    end
    return y
end


function Base.:abs(x::Variable{T}) where T
    y = Variable{T}(abs(ᵛ(x)), x.backprop)
    if x.backprop
        function absBackward()
            if need2computeδ!(x)
                δ(x) .+= sign.(ᵛ(x)) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, absBackward)
    end
    return y
end


function Base.:reshape(x::Variable{T}, newsize) where T
    y = Variable{T}( reshape(ᵛ(x), newsize), x.backprop )
    if x.backprop
        function reshapeBackward()
            if need2computeδ!(x)
                δ(x) .+= reshape(δ(y), x.shape)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, reshapeBackward)
    end
    return y
end


function exp2!(x::AbstractArray)
    @. x = exp2(x)
end


function Base.:exp2(x::AbstractArray)
    return exp2.(x)
end


function exp2!(x::Variable{T}) where T
    # exp2 represents y = 2^x
    y = Variable{T}(exp2!(ᵛ(x)), x.backprop)
    if x.backprop
        TWO = eltype(x)(2)
        function exp2Backward()
            if need2computeδ!(x)
                δ(x) .+= log(TWO) .* ᵛ(y) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, exp2Backward)
    end
    return y
end


function Base.:exp2(x::Variable{T}) where T
    # EXP2 represents y = 2^x
    y = Variable{T}(exp2(ᵛ(x)), x.backprop)
    if x.backprop
        TWO = eltype(x)(2)
        function exp2Backward()
            if need2computeδ!(x)
                δ(x) .+= log(TWO) .* ᵛ(y) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, exp2Backward)
    end
    return y
end


function exp10!(x::AbstractArray)
    @. x = exp10(x)
end


function Base.:exp10(x::AbstractArray)
    return exp10.(x)
end


function exp10!(x::Variable{T}) where T
    # EXP10 represents y = 10^x
    y = Variable{T}(exp10!(ᵛ(x)), x.backprop)
    if x.backprop
        TEN = eltype(x)(10)
        function exp10Backward()
            if need2computeδ!(x)
                δ(x) .+= log(TEN) .* ᵛ(y) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, exp10Backward)
    end
    return y
end


function Base.:exp10(x::Variable{T}) where T
    # EXP10 represents y = 10^x
    y = Variable{T}(exp10(ᵛ(x)), x.backprop)
    if x.backprop
        TEN = eltype(x)(10)
        function exp10Backward()
            if need2computeδ!(x)
                δ(x) .+= log(TEN) .* ᵛ(y) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, exp10Backward)
    end
    return y
end


function log2!(x::AbstractArray)
    @. x = log2(x)
end


function Base.:log2(x::AbstractArray)
    return log2.(x)
end


function log2!(x::Variable{T}) where T
    y = Variable{T}(log2(ᵛ(x)), x.backprop)
    if x.backprop
        TWO = eltype(x)(2)
        function log2Backward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ (log(TWO) .* ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, log2Backward)
    end
    return y
end


function Base.:log2(x::Variable{T}) where T
    y = Variable{T}(log2(ᵛ(x)), x.backprop)
    if x.backprop
        TWO = eltype(x)(2)
        function log2Backward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ (log(TWO) .* ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, log2Backward)
    end
    return y
end


function log10!(x::AbstractArray)
    @. x = log10(x)
end


function Base.:log10(x::AbstractArray)
    return log10.(x)
end


function log10!(x::Variable{T}) where T
    y = Variable{T}(log10(ᵛ(x)), x.backprop)
    if x.backprop
        TEN = eltype(x)(10)
        function log10Backward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ (log(TEN) .* ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, log10Backward)
    end
    return y
end


function Base.:log10(x::Variable{T}) where T
    y = Variable{T}(log10(ᵛ(x)), x.backprop)
    if x.backprop
        TEN = eltype(x)(10)
        function log10Backward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ (log(TEN) .* ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, log10Backward)
    end
    return y
end


function sec!(x::Variable{T}) where T
    # SEC represents y = sec(x)
    y = Variable{T}(sec(ᵛ(x)), x.backprop)
    if x.backprop
        function secBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* ᵛ(y) .* tan.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, secBackward)
    end
    return y
end


function Base.:sec(x::Variable{T}) where T
    # SEC represents y = sec(x)
    y = Variable{T}(sec(ᵛ(x)), x.backprop)
    if x.backprop
        function secBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* ᵛ(y) .* tan.(ᵛ(x))
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, secBackward)
    end
    return y
end


function sqrt!(x::Variable{T}) where T
    y = Variable{T}(sqrt!(ᵛ(x)), x.backprop)
    if x.backprop
        TOO = eltype(x)
        TWO = TOO(2.000)
        EPS = TOO(1e-38)
        function sqrtBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ (TWO .* (ᵛ(y) .+ EPS))
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, sqrtBackward)
    end
    return y
end


function Base.:sqrt(x::Variable{T}) where T
    y = Variable{T}(sqrt(ᵛ(x)), x.backprop)
    if x.backprop
        TOO = eltype(x)
        TWO = TOO(2.000)
        EPS = TOO(1e-38)
        function sqrtBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) ./ (TWO .* (ᵛ(y) .+ EPS))
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, sqrtBackward)
    end
    return y
end


# -- tan serials --
export tan!
function tan!(x::Variable{T}) where T
    y = Variable{T}(tan!(ᵛ(x)), x.backprop)
    if x.backprop
        TOO = eltype(x)
        ONE = TOO(1.0)
        TWO = TOO(2.0)
        function tanBackward()
            if need2computeδ!(x)
                δ(x) .+= (ONE .+ ᵛ(y).^TWO) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, tanBackward)
    end
    return y
end


function Base.:tan(x::Variable{T}) where T
    y = Variable{T}(tan(ᵛ(x)), x.backprop)
    if x.backprop
        TOO = eltype(x)
        ONE = TOO(1.0)
        TWO = TOO(2.0)
        function tanBackward()
            if need2computeδ!(x)
                δ(x) .+= (ONE .+ ᵛ(y).^TWO) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, tanBackward)
    end
    return y
end


export tand!
function tand!(x::AbstractArray)
    @. x = tand(x)
end


function Base.:tand(x::AbstractArray)
    return tand.(x)
end


function tand!(x::Variable{T}) where T
    y = Variable{T}(tand!(ᵛ(x)), x.backprop)
    if x.backprop
        TOO = eltype(x)
        ONE = TOO(1.0)
        TWO = TOO(2.0)
        DPI = TOO(pi/180)
        function tandBackward()
            if need2computeδ!(x)
                δ(x) .+= DPI .* (ONE .+ ᵛ(y).^TWO) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, tandBackward)
    end
    return y
end


function Base.:tand(x::Variable{T}) where T
    y = Variable{T}(tand(ᵛ(x)), x.backprop)
    if x.backprop
        TOO = eltype(x)
        ONE = TOO(1.0)
        TWO = TOO(2.0)
        DPI = TOO(pi/180)
        function tandBackward()
            if need2computeδ!(x)
                δ(x) .+= DPI .* (ONE .+ ᵛ(y).^TWO) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, tandBackward)
    end
    return y
end


export tanh!
function tanh!(x::Variable{T}) where T
    y = Variable{T}(tanh!(ᵛ(x)), x.backprop)
    if x.backprop
        TOO = eltype(x)
        ONE = TOO(1.0)
        TWO = TOO(2.0)
        function tanhBackward()
            if need2computeδ!(x)
                δ(x) .+= (ONE .- ᵛ(y).^TWO) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, tanhBackward)
    end
    return y
end


function Base.:tanh(x::Variable{T}) where T
    y = Variable{T}(tanh(ᵛ(x)), x.backprop)
    if x.backprop
        TOO = eltype(x)
        ONE = TOO(1.0)
        TWO = TOO(2.0)
        function tanhBackward()
            if need2computeδ!(x)
                δ(x) .+= (ONE .- ᵛ(y).^TWO) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, tanhBackward)
    end
    return y
end


export tanhshrink, tanhshrink!
function tanhshrink!(x::AbstractArray)
    @. x = x - tanh(x)
end


function tanhshrink(x::AbstractArray)
    return  x - tanh(x)
end


function tanhshrink!(x::Variable{T}) where T
    return x - tanh(x)
end


function tanhshrink(x::Variable{T}) where T
    return x - tanh(x)
end


# # -- sin serials --
export sin!
function sin!(x::Variable{T}) where T
    y = Variable{T}(sin(ᵛ(x)), x.backprop)
    if x.backprop
        function sinBackward()
            if need2computeδ!(x)
                δ(x) .+= cos.(ᵛ(x)) .* δ(y);
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, sinBackward)
    end
    return y
end


function Base.:sin(x::Variable{T}) where T
    y = Variable{T}(sin(ᵛ(x)), x.backprop)
    if x.backprop
        function sinBackward()
            if need2computeδ!(x)
                δ(x) .+= cos.(ᵛ(x)) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, sinBackward)
    end
    return y
end


export sinc!
function sinc!(x::AbstractArray)
    @. x = sinc(x)
end


function Base.:sinc(x::AbstractArray)
    return sinc.(x)
end


function sinc!(x::Variable{T}) where T
    # sinc represents y = sin(pi*x)/(pi*x)
    y = Variable{T}(sinc(ᵛ(x)), x.backprop)
    if x.backprop
        function sincBackward()
            if need2computeδ!(x)
                δ(x) .+= cosc.(ᵛ(x)) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, sincBackward)
    end
    return y
end


function Base.:sinc(x::Variable{T}) where T
    # sinc represents y = sin(pi*x)/(pi*x)
    y = Variable{T}(sinc(ᵛ(x)), x.backprop)
    if x.backprop
        function sincBackward()
            if need2computeδ!(x)
                δ(x) .+= cosc.(ᵛ(x)) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, sincBackward)
    end
    return y
end


export sind!
function sind!(x::AbstractArray)
    @. x = sind(x)
end


function Base.:sind(x::AbstractArray)
    return sind.(x)
end


function sind!(x::Variable{T}) where T
    y = Variable{T}(sind(ᵛ(x)), x.backprop)
    if x.backprop
        DPI = eltype(x)(pi/180)
        function sindBackward()
            if need2computeδ!(x)
                δ(x) .+= DPI .* cosd.(ᵛ(x)) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, sindBackward)
    end
    return y
end


function Base.:sind(x::Variable{T}) where T
    y = Variable{T}(sind(ᵛ(x)), x.backprop)
    if x.backprop
        DPI = eltype(x)(pi/180)
        function sindBackward()
            if need2computeδ!(x)
                δ(x) .+= DPI .* cosd.(ᵛ(x)) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, sindBackward)
    end
    return y
end


export sinpi!
function sinpi!(x::AbstractArray)
    @. x = sinpi(x)
end


function Base.:sinpi(x::AbstractArray)
    return sinpi.(x)
end


function sinpi!(x::Variable{T}) where T
    y = Variable{T}(sinpi(ᵛ(x)), x.backprop)
    if x.backprop
        PI = eltype(x)(pi)
        function sinpiBackward()
            if need2computeδ!(x)
                δ(x) .+= PI .* cospi.(ᵛ(x)) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, sinpiBackward)
    end
    return y
end


function Base.:sinpi(x::Variable{T}) where T
    y = Variable{T}(sinpi(ᵛ(x)), x.backprop)
    if x.backprop
        PI = eltype(x)(pi)
        function sinpiBackward()
            if need2computeδ!(x)
                δ(x) .+= PI .* cospi.(ᵛ(x)) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, sinpiBackward)
    end
    return y
end


export linearsin,linearsin!
function linearsin!(x::AbstractArray)
    @. x = sin(x) + x
end


function linearsin(x::AbstractArray)
    return sin(x) + x
end


function linearsin!(x::Variable{T}) where T
    return sin(x) + x
end


function linearsin(x::Variable{T}) where T
    return sin(x) + x
end


export cos!
function cos!(x::Variable{T}) where T
    y = Variable{T}(cos(ᵛ(x)), x.backprop)
    if x.backprop
        function cosBackward()
            if need2computeδ!(x)
                δ(x) .+= - sin.(ᵛ(x)) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, cosBackward)
    end
    return y
end


function Base.:cos(x::Variable{T}) where T
    y = Variable{T}(cos(ᵛ(x)), x.backprop)
    if x.backprop
        function cosBackward()
            if need2computeδ!(x)
                δ(x) .+= - sin.(ᵛ(x)) .* δ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, cosBackward)
    end
    return y
end


export inv!
function inv!(x::Variable{T}) where T
    y = Variable{T}(inv!(ᵛ(x)), x.backprop)
    if x.backprop
        function invBackward()
            if need2computeδ!(x)
                δ(x) .+= - δ(y) .* ᵛ(y) .* ᵛ(y);
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, invBackward)
    end
    return y
end


function inv(x::Variable{T}) where T
    y = Variable{T}(inv(ᵛ(x)), x.backprop)
    if x.backprop
        function invBackward()
            if need2computeδ!(x)
                δ(x) .+= - δ(y) .* ᵛ(y) .* ᵛ(y)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, invBackward)
    end
    return y
end
