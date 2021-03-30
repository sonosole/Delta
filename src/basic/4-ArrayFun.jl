begin

XArray = eval(:Array)

import Base.sin
import Base.cos
import Base.exp
import Base.log
import Base.sec
import Base.sqrt
import Base.tan
import Base.tanh
import Base.inv


function sin!(x::XArray)
    @. x = sin(x)
end


function Base.:sin(x::XArray)
    return sin.(x)
end


function cos!(x::XArray)
    @. x = cos(x)
end


function Base.:cos(x::Array)
    return cos.(x)
end


function sec!(x::XArray)
    @. x = sec(x)
end


function Base.:sec(x::XArray)
    return sec.(x)
end


function tan!(x::XArray)
    @. x = tan(x)
end


function Base.:tan(x::XArray)
    return tan.(x)
end


function tanh!(x::XArray)
    @. x = tanh(x)
end


function Base.:tanh(x::XArray)
    return tanh.(x)
end


function exp!(x::XArray)
    @. x = exp(x)
end


function Base.:exp(x::XArray)
    return exp.(x)
end


function log!(x::XArray)
    @. x = log(x)
end


function Base.:log(x::XArray)
    return log.(x)
end


function sqrt!(x::XArray)
    @. x = sqrt(x)
end


function Base.:sqrt(x::XArray)
    return sqrt.(x)
end


function inv!(x::XArray)
    @. x = inv(x)
end


function inv(x::XArray)
    return inv.(x)
end


end
