import Base.sin
import Base.cos
import Base.exp
import Base.log
import Base.sec
import Base.sqrt
import Base.tan
import Base.tanh
import Base.inv


function sin!(x::T) where T <: AbstractArray
    @. x = sin(x)
end


function Base.:sin(x::T) where T <: AbstractArray
    return sin.(x)
end


function cos!(x::T) where T <: AbstractArray
    @. x = cos(x)
end


function Base.:cos(x::Array)
    return cos.(x)
end


function sec!(x::T) where T <: AbstractArray
    @. x = sec(x)
end


function Base.:sec(x::T) where T <: AbstractArray
    return sec.(x)
end


function tan!(x::T) where T <: AbstractArray
    @. x = tan(x)
end


function Base.:tan(x::T) where T <: AbstractArray
    return tan.(x)
end


function tanh!(x::T) where T <: AbstractArray
    @. x = tanh(x)
end


function Base.:tanh(x::T) where T <: AbstractArray
    return tanh.(x)
end


function exp!(x::T) where T <: AbstractArray
    @. x = exp(x)
end


function Base.:exp(x::T) where T <: AbstractArray
    return exp.(x)
end


function log!(x::T) where T <: AbstractArray
    @. x = log(x)
end


function Base.:log(x::T) where T <: AbstractArray
    return log.(x)
end


function sqrt!(x::T) where T <: AbstractArray
    @. x = sqrt(x)
end


function Base.:sqrt(x::T) where T <: AbstractArray
    return sqrt.(x)
end


function inv!(x::T) where T <: AbstractArray
    @. x = inv(x)
end


function inv(x::T) where T <: AbstractArray
    return inv.(x)
end
