export LogZero
export LogSum2Exp
export LogSum3Exp
export LogSumExp


LogZero(T::DataType) = - floatmax(T)


function LogSum2Exp(a::T1, b::T2) where {T1<:Real, T2<:Real}
    Log0 = min(LogZero(T1), LogZero(T2))
    if a <= Log0
        a = Log0
    end
    if b <= Log0
        b = Log0
    end
    return (max(a,b) + log(1.0 + exp(-abs(a-b))))
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
