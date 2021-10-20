abstract type Optimizer end

include("./1-Descent.jl")
include("./2-Momentum.jl")
include("./3-Adam.jl")
include("./4-AdaGrad.jl")
include("./5-RMSProp.jl")
include("./6-AdamW.jl")

export Optimizer
export Descent
export Momentum
export Adam
export AdaGrad
export RMSProp
export AdamW
export decay

export normclip
export LpNormClip
export L2NormClip
export L1NormClip
export L0NormClip
export LPInfNormClip
export LNInfNormClip
export lrarray


function decay(params::Vector{Variable}; ratio=0.999)
    for p in params
        p.value .*= ratio
    end
end


function L2NormClip(x::AbstractArray, clipvalue)
    pnorm = sqrt(sum(x.^2) / length(x))
    scale = clipvalue / pnorm
    if pnorm > clipvalue
        x .*= scale
    end
    return x
end


function L1NormClip(x::AbstractArray, clipvalue)
    pnorm = sum(abs.(x)) / length(x)
    scale = clipvalue / pnorm
    if pnorm > clipvalue
        x .*= scale
    end
    return x
end


function L0NormClip(x::AbstractArray, clipvalue)
    pnorm = sum(x .!= 0.0) / length(x)
    scale = clipvalue / pnorm
    if pnorm > clipvalue
        x .*= scale
    end
    return x
end


function LPInfNormClip(x::AbstractArray, clipvalue)
    pnorm = maximum(abs.(x))
    scale = clipvalue / pnorm
    if pnorm > clipvalue
        x .*= scale
    end
    return x
end


function LNInfNormClip(x::AbstractArray, clipvalue)
    pnorm = minimum(abs.(x))
    scale = clipvalue / pnorm
    if pnorm > clipvalue
        x .*= scale
    end
    return x
end


function LpNormClip(x::AbstractArray, clipvalue; order::Union{Int,String}=2)
    order==2 && return L2NormClip(x, clipvalue)
    order==1 && return L1NormClip(x, clipvalue)
    order==0 && return L0NormClip(x, clipvalue)
    order=="inf"  && return LPInfNormClip(x, clipvalue)
    order=="-inf" && return LNInfNormClip(x, clipvalue)
    pnorm = (sum( abs.(x).^order ) / length(x)) ^ (1/order)
    scale = clipvalue / pnorm
    if pnorm > clipvalue
        x .*= scale
    end
    return x
end


"""
    lrarray(init,final,steps;func="exp") -> Array{Float32,1}
# Arguments
- ` init`: initial learning rate
- `final`: final learning rate
- `steps`: steps to change from `init` to `final`
- ` func`: function used to change the learning rate. (e.g. exp/cos/linear)
"""
function lrarray(init,final,steps;func="exp")
    func=="exp"    && return lrarrayexp(init,final,steps)
    func=="cos"    && return lrarraycos(init,final,steps)
    func=="linear" && return lrarraylinear(init,final,steps)
end


function lrarrayexp(i,f,n)
    if n == 1
        return [i]
    else
        lr = zeros(Float32,n)
        α  = log(f/i)/(n-1)
        for x = 1:n
            lr[x] = i * exp(α * (x-1))
        end
        return lr
    end
end

function lrarraylinear(i,f,n)
    if n == 1
        return [i]
    else
        lr = zeros(Float32,n)
        α  = (f-i)/(n-1)
        for x = 1:n
            lr[x] = α*(x-1) + i
        end
        return lr
    end
end


function lrarraycos(i,f,n)
    if n == 1
        return [i]
    else
        lr = zeros(Float32,n)
        α  = pi/2/(n-1)
        Δ  = (i-f)
        for x = 1:n
            lr[x] = cos(α*(x-1)) * Δ + f
        end
        return lr
    end
end
