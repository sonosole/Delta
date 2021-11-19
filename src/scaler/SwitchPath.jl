"""
# Summary SwitchPath
    mutable struct SwitchPath <: Scaler
# Fields
    slope::AbstractFloat
    scale::VarOrNil
Applies x -> x/(1+exp(-slope*scale) over a N-dimensional input x,
where slope is user defined, scale is a scalar Variable.
"""
mutable struct SwitchPath <: Scaler
    slope::AbstractFloat
    scale::VarOrNil
    function SwitchPath(slope::AbstractFloat; ndims::Int, type::Type=Array{Float32})
        @assert ndims >= 1 "ndims >= 1 shall be met, but got ndims=$ndims"
        shape = ntuple(i->1, ndims)
        scale = Variable{type}(Zeros(type, shape) .+ eltype(type)(7/slope), true, true, true);
        new(slope, scale)
    end
    function SwitchPath()
        new(1.0f0, nothing)
    end
end


function clone(this::SwitchPath; type::Type=Array{Float32})
    cloned = SwitchPath()
    cloned.slope = this.slope
    cloned.scale = clone(this.scale, type=type)
    return cloned
end

function Base.show(io::IO, m::SwitchPath)
    print(io, "SwitchPath(slope=$slope, scale=$(m.scale.value[1]); type=$TYPE)")
end

function paramsof(m::SwitchPath)
    params = Vector{Variable}(undef,1)
    params[1] = m.scale
    return params
end

function xparamsof(m::SwitchPath)
    xparams = Vector{XVariable}(undef,1)
    xparams[1] = ('w', m.scale)
    return xparams
end

function nparamsof(m::SwitchPath)
    return 1
end

function bytesof(m::SwitchPath, unit::String="MB")
    return blocksize(sizeof(m.scale), uppercase(unit))
end


function forward(m::SwitchPath, x::Variable{T}) where T
    k = m.slope  # f(x) = σ(k*x)
    a = m.scale  # scale param
    G = 1 / (1 + exp(-k * ᵛ(a)))
    y = Variable{T}(ᵛ(x) .* G, x.backprop)

    if x.backprop
        function SwitchPathBackward()
            if need2computeδ!(x) δ(x) .+=     δ(y) .* G                      end
            if need2computeδ!(a) δ(a) .+= sum(δ(y) .* k .* (1 .- G) .* ᵛ(y)) end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, SwitchPathBackward)
    end
    return y
end


function predict(m::SwitchPath, x::AbstractArray)
    k = m.slope
    a = m.scale
    C = 1 / (1 + exp(-k * ᵛ(a)))
    return ᵛ(x) .* C
end
