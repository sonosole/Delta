"""
# Summary
    mutable struct MeanNorm <: Normalizer
# Fields
    β        :: VarOrNil                        # shifting params
    μ        :: Union{AbstractArray,Nothing}    # running average
    views    :: NTuple                          # views to get the statistical mean
    training :: Bool                            # if trainning then true
    momentum :: AbstractFloat                   # smoothing const for moving average

Applies mean normalization over a N-dimensional input

"""
mutable struct MeanNorm <: Normalizer
    β::VarOrNil                        # shifting params
    μ::Union{AbstractArray,Nothing}    # running average
    views::Union{NTuple,Nothing}
    training::Union{Bool,Nothing}
    momentum::Union{AbstractFloat,Nothing}
    function MeanNorm(;ndims::Int,
                       keptdims::Union{Tuple,Int},    # must be unique and sorted and positive
                       keptsize::Union{Tuple,Int},    # must be positive
                       momentum::AbstractFloat=0.10,  # smoothing const
                       type::Type=Array{Float32})

        @assert typeof(keptsize)==typeof(keptdims) "keptsize & keptdims shall be the same type"
        @assert ndims >= maximum(keptdims) "ndims >= maximum(keptdims) shall be met"
        @assert ndims > length(keptdims) "this is no elements for statistical analysis"

        if typeof(keptdims) <: Int
            if keptdims == 0
                if keptsize!=1
                    @warn "keptsize should be 1 here, but got $keptsize"
                end
                shape = ntuple(i -> i==keptdims ? keptsize : 1, ndims);
                views = ntuple(i -> i, ndims);
            else
                shape = ntuple(i -> i==keptdims ? keptsize : 1, ndims);
                views = ntuple(i -> i>=keptdims ? i+1 : i, ndims-1);
            end
        else
            array = [i for i in keptsize]
            shape = ntuple(i -> i in keptdims ? popfirst!(array) : 1, ndims);
            views = deleteat!(ntuple(i -> i, ndims), keptdims)
        end

        β = Variable{type}(Zeros(type, shape), true, true, true);
        μ = Zeros(type, shape);
        T = eltype(type);
        new(β, μ, views, true, T(momentum))
    end
    function MeanNorm()
        new(nothing, nothing, nothing, nothing, nothing)
    end
end

function clone(this::MeanNorm; type::Type=Array{Float32})
    cloned = MeanNorm()
    cloned.β =  clone(this.β, type=type)
    cloned.μ =   type(this.μ)
    cloned.views    = this.views
    cloned.training = this.training
    cloned.momentum = this.momentum
    return cloned
end

function Base.show(io::IO, m::MeanNorm)
    SIZE = size(m.β.value)
    TYPE = typeof(m.β.value)
    print(io, "MeanNorm(β=$SIZE; type=$TYPE)")
end

function xparamsof(m::MeanNorm)
    xparams = Vector{XVariable}(undef,1)
    xparams[1] = ('b', m.β)
    return xparams
end

function nparamsof(model::MeanNorm)
    return 2*length(model.β)
end

function bytesof(model::MeanNorm, unit::String="MB")
    n = nparamsof(model)
    u = lowercase(unit)
    if u == "kb" return n * sizeof(eltype(model[1].w)) / 1024 end
    if u == "mb" return n * sizeof(eltype(model[1].w)) / 1048576 end
    if u == "gb" return n * sizeof(eltype(model[1].w)) / 1073741824 end
    if u == "tb" return n * sizeof(eltype(model[1].w)) / 1099511627776 end
end


function forward(M::MeanNorm, x::Variable{T}) where T
    β = M.β         # shifting params
    μ = M.μ         # statistical mean
    ρ = M.momentum  # smoothing const
    v = M.views
    n = length(β) / length(x)
    μₓ = sum(x.value, dims=v) .* n
    y  = Variable{T}(x.value .- μₓ .+ β.value, x.backprop)

    if x.backprop
        @. μ = (1 - ρ) * μ + ρ * μₓ    # running mean
        function MeanNormBackward()
            if need2computeδ!(x)
                x.delta += y.delta .- sum(y.delta, dims=v) .* n
            end
            if need2computeδ!(β)
                β.delta += sum(y.delta, dims=v)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, MeanNormBackward)
    end
    return y
end


function predict(M::MeanNorm, x::AbstractArray)
    β = M.β.value   # learned shifting param
    μ = M.μ         # statistical mean param
    return x .- μ .+ β
end
