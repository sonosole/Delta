abstract type Optimizer end
export Optimizer


# optimizers with L1 and L2 weight decay
include("./1-SGDL1L2.jl")
include("./2-MomentumL1L2.jl")
include("./3-AdamL1L2.jl")
include("./4-AdaGradL1L2.jl")
include("./5-RMSPropL1L2.jl")
export SGD
export Momentum
export Adam
export AdaGrad
export RMSProp

# auto gradient clipping
include("./auto-grad-cliper.jl")
export AutoGradCliper
export AutoGradNormCliper

export decay
export normclip
export LpNormClip
export L2NormClip
export L1NormClip
export L0NormClip
export LPInfNormClip
export LNInfNormClip
export setNanInfZero
export lrarray


"""
    clip!(::Vector{XVariable}, kind='u'; L1decay=0.0, L2decay=0.0, clipvalue=1.0)

Limit the amplitude of parameters.
"""
function clip!(xparams::Vector{XVariable}, kind='u'; L1decay=0.0, L2decay=0.0, clipvalue=1.0)
    @assert clipvalue>0 "clipvalue is positive, but got $clipvalue"
    if !(kind=='u' || kind=='b' || kind=='w')
        @error "type of XVariable not among u/w/b, but got $kind"
    end

    λ₁ = -L1decay
    λ₂ = -L2decay
    for (c, θ) in xparams
        if c == kind
            𝒗 = ᵛ(θ)
            i = abs.(𝒗) .> clipvalue
            if λ₁==0 && λ₂==0                     # Hard truncation
                @. 𝒗[i] = clipvalue * sign(𝒗[i])
            elseif λ₁==0 && λ₂!=0                 # Soft truncation (L2)
                @. 𝒗[i] += λ₂ * 𝒗[i]
            elseif λ₁!=0 && λ₂==0                 # Gradual truncation (L1)
                @. 𝒗[i] += λ₁ * sign(𝒗[i])
            else  # λ₁!=0 && λ₂!=0
                @. 𝒗[i] += λ₁ * sign(𝒗[i]) + λ₂ * 𝒗[i]
            end
        end
    end
end


function decay(params::Vector{Variable}; ratio=0.999)
    for p in params
        p.value .*= ratio
    end
end


"""
    setNanInfZero(x)
```julia
x = randn(1,4)
x[1] = Inf;
x[2] =-Inf;
x[3] = NaN;
```
```
julia> x
1×4 Array{Float64,2}:
 Inf  -Inf  NaN  0.602655

julia> x = setNanInfZero!(x)
1×4 Array{Float64,2}:
 0.0  0.0  0.0  0.602655
```
 """
function setNanInfZero(x)
    x[ isnan.(x) .⊻ isinf.(x) ] .= 0.0
    return x
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
