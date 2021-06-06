import Base.sin
import Base.cos
import Base.exp
import Base.log
import Base.sec
import Base.sqrt
import Base.tan
import Base.tanh
import Base.inv

begin

Xarray = Array


function sin!(x::Xarray)
    @. x = sin(x)
end


function Base.:sin(x::Xarray)
    return sin.(x)
end


function cos!(x::Xarray)
    @. x = cos(x)
end


function Base.:cos(x::Xarray)
    return cos.(x)
end


function sec!(x::Xarray)
    @. x = sec(x)
end


function Base.:sec(x::Xarray)
    return sec.(x)
end


function tan!(x::Xarray)
    @. x = tan(x)
end


function Base.:tan(x::Xarray)
    return tan.(x)
end


function tanh!(x::Xarray)
    @. x = tanh(x)
end


function Base.:tanh(x::Xarray)
    return tanh.(x)
end


function exp!(x::Xarray)
    @. x = exp(x)
end


function Base.:exp(x::Xarray)
    return exp.(x)
end


function log!(x::Xarray)
    @. x = log(x)
end


function Base.:log(x::Xarray)
    return log.(x)
end


function sqrt!(x::Xarray)
    @. x = sqrt(x)
end


function Base.:sqrt(x::Xarray)
    return sqrt.(x)
end


function inv!(x::Xarray)
    @. x = inv(x)
end


function inv(x::Xarray)
    return inv.(x)
end


end # begin
