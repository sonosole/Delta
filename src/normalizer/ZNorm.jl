"""
# Summary
    mutable struct ZNorm <: Normalizer
# Fields
    γ        :: VarOrNil                        # scaling params
    β        :: VarOrNil                        # shifting params
    μ        :: Union{AbstractArray,Nothing}    # running average
    σ        :: Union{AbstractArray,Nothing}    # running variance otherwise standard deviation
    views    :: NTuple                          # views to collect elements for mean and var
    training :: Bool                            # if traning or not
    epsilion :: AbstractFloat                   # prevent dividing by zero, 1e-10 for default
    momentum :: AbstractFloat                   # smoothing const, or called historical inertia coefficient

Applies mean and scaling normalization over a N-dimensional input, like BatchNorm LayerNorm and InstanceNorm.

"""
mutable struct ZNorm <: Normalizer
    γ::VarOrNil                        # scaling params
    β::VarOrNil                        # shifting params
    μ::Union{AbstractArray,Nothing}    # running average
    σ::Union{AbstractArray,Nothing}    # running variance otherwise standard deviation
    views::NTuple                      # views to collect elements for mean and var
    training::Bool                     # if traning or not
    epsilion::AbstractFloat            # prevent dividing by zero, 1e-10 for default
    momentum::AbstractFloat            # inertia coefficient
    function ZNorm(;ndims::Int,        # how many dimentions the input data has
                   keptdims::Union{Tuple,Int},     # must be unique and sorted and positive
                   keptsize::Union{Tuple,Int},     # must be positive
                   epsilion::AbstractFloat=1e-10,  # stability const
                   momentum::AbstractFloat=0.900,  # smoothing const or historical inertia
                   type::Type=Array{Float32})

        shape, views = ShapeAndViews(ndims, keptdims, keptsize);
        γ = Variable{type}( Ones(type, shape), true, true, true);
        β = Variable{type}(Zeros(type, shape), true, true, true);
        μ = Zeros(type, shape);
        σ =  Ones(type, shape);
        new(γ, β, μ, σ, views, true, epsilion, momentum)
    end
    function ZNorm(views, training, epsilion, momentum)
        new(nothing, nothing, nothing, nothing, views, training, epsilion, momentum)
    end
end


function clone(this::ZNorm; type::Type=Array{Float32})
    cloned = ZNorm(this.views, this.training, this.epsilion, this.momentum)
    cloned.γ = clone(this.γ, type=type)
    cloned.β = clone(this.β, type=type)
    cloned.μ = type(this.μ)
    cloned.σ = type(this.σ)
    return cloned
end

function Base.show(io::IO, m::ZNorm)
    SIZE = size(m.β.value)
    TYPE = typeof(m.β.value)
    print(io, "ZNorm(statistic vars' size=$SIZE; type=$TYPE)")
end


function paramsof(m::ZNorm)
    params = Vector{Variable}(undef,2)
    params[1] = m.γ
    params[2] = m.β
    return params
end

function xparamsof(m::ZNorm)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', m.γ)
    xparams[2] = ('b', m.β)
    return xparams
end

function nparamsof(model::ZNorm)
    return 4*length(model.β)
end

function bytesof(model::ZNorm, unit::String="MB")
    n = nparamsof(model) * sizeof(eltype(model.β))
    return blocksize(n, uppercase(unit))
end

function forward(b::ZNorm, x::Variable{T}) where T
    γ = b.γ
    β = b.β
    ϵ = b.epsilion
    ρ = b.momentum
    v = b.views
    μ = mean(x.value, dims=v)
    σ =  std(x.value, dims=v, mean=μ, corrected=false)
    𝐗 = (x.value .- μ) ./ (σ .+ ϵ)
    y = Variable{T}(𝐗 .* γ.value .+ β.value, x.backprop)

    if y.backprop
        Σ = σ .* σ
        @. b.μ = ρ * b.μ + (1 - ρ) * μ    # running mean
        @. b.σ = ρ * b.σ + (1 - ρ) * Σ    # running var
        function ZNormBackward()
            if need2computeδ!(x)
                n     = length(σ)/length(x)
                σ¯¹   = 1 ./ (σ .+ ϵ)
                σ¯³   = (σ¯¹).^3
                Γ     = γ.value
                Δ     = x.value .- μ
                ∂𝐋∂𝐗  = y.delta .* Γ
                Δ∂𝐋∂𝐗 = Δ .* ∂𝐋∂𝐗
                SumΔ∂𝐋∂𝐗  = sum(Δ∂𝐋∂𝐗, dims=v)

                x.delta .+= σ¯¹ .* ∂𝐋∂𝐗
                x.delta .-= σ¯³ .* n   .* SumΔ∂𝐋∂𝐗 .* Δ
                x.delta .+= σ¯³ .* n^2 .* SumΔ∂𝐋∂𝐗 .* sum(Δ, dims=v) .- σ¯¹ .* n .* sum(∂𝐋∂𝐗, dims=v)

                if need2computeδ!(γ) γ.delta .+= sum(y.delta .* 𝐗, dims=v) end
                if need2computeδ!(β) β.delta .+= sum(y.delta,      dims=v) end
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, ZNormBackward)
    end
    return y
end


function predict(b::ZNorm, x::AbstractArray)
    ϵ = b.epsilion
    γ = b.γ.value
    β = b.β.value
    μ = b.μ
    σ = b.σ
    return @. (x - μ) / sqrt(σ + ϵ) * γ + β
end


function BatchNorm1d(nchannels::Int;
            epsilion::AbstractFloat=1e-10,
            momentum::AbstractFloat=0.900,
            type::Type=Array{Float32})
    return ZNorm(ndims=3,
                 keptdims=1,
                 keptsize=nchannels,
                 epsilion=epsilion,
                 momentum=momentum,
                 type=type)
end


function BatchNorm0d(nchannels::Int;
            epsilion::AbstractFloat=1e-10,
            momentum::AbstractFloat=0.900,
            type::Type=Array{Float32})
    return ZNorm(ndims=2,
                 keptdims=1,
                 keptsize=nchannels,
                 epsilion=epsilion,
                 momentum=momentum,
                 type=type)
end
