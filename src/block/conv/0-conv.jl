include("./1-conv1d.jl")

# 每个块使用线程数量
global CuThreads = 512

# 线程块数
function CuBlocks(n::Int)
    return div(n + CuThreads - 1, CuThreads)
end

# 设置线程数量
function SetCudaThreads(n::Int)
    @assert n>=1 "minimum number of threads is 1, but got $n"
    global CuThreads = n
    return nothing
end
