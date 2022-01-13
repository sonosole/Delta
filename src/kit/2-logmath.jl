export LogZero
export LogSum2Exp
export LogSum3Exp
export LogSumExp


LogZero(T::DataType) = - floatmax(T)

"""
    LogSum2Exp(a::Real, b::Real) -> max(a,b) + log(1.0 + exp(-abs(a-b)))

LogSum2Exp(log(a), log(b)) isequal to log(a + b)

```julia
julia> LogSum2Exp(Float32(1.2),Float64(3.3))
3.4155195283818967

julia> LogSum2Exp(log(1.0), log(2.0)) ≈ log(1.0 + 2.0)
true
```
"""
function LogSum2Exp(a::Real, b::Real)
    isinf(a) && return b
    isinf(b) && return a
    if a < b
        a, b = b, a
    end
    return (a + log(1.0 + exp(b-a)))
end


function LogSum3Exp(a::Real, b::Real, c::Real)
    return LogSum2Exp(LogSum2Exp(a,b),c)
end


function LogSumExp(a)
    tmp = LogZero(eltype(a))
    for i = 1:length(a)
        tmp = LogSum2Exp(tmp, a[i])
    end
    return tmp
end
