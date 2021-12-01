function wnorm(x, smooth=0.0; by="min")
    by=="allone"  && return allone(x)
    by=="prior"   && return prior(x,smooth)
    by=="inprior" && return inprior(x,smooth)
end

function allone(x)
    return ones(eltype(x),length(x))
end


"""
    prior(x, smooth=0.0) -> z
    y = (x .+ smooth*maximum(x))
    z = y ./ sum(y)
# Example
    julia> x'
    1×3 LinearAlgebra.Adjoint{Int64,Array{Int64,1}}:
    1  100  1000
    julia> prior(x,0.5)'
    1×3 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
    0.192618  0.230681  0.576701
"""
function prior(x, smooth=0.0)
    n = maximum(x) * smooth
    y = x .+ n
    s = sum(y)
    return y ./ sum(y)
end


"""
    invprior(x, smooth=0.0) -> z
    y = 1 ./ (x .+ smooth*maximum(x))
    z = y ./ sum(y)
# Example
    julia> x'
    1×3 LinearAlgebra.Adjoint{Int64,Array{Int64,1}}:
     1  100  1000

    julia> invprior(x)'
    1×3 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
     0.461042  0.38497  0.153988
"""
function invprior(x, smooth=0.0)
    n = maximum(x) * smooth
    y = 1 ./ (x .+ n)
    s = sum(y)
    return y ./ sum(y)
end
