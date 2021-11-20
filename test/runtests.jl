using Test
using Delta


@testset "checking gradient" begin
    include("./checkgrad/0-softmax.jl")
    include("./checkgrad/1-pool.jl")
    include("./checkgrad/2-linear.jl")
    include("./checkgrad/3-mlp.jl")
    include("./checkgrad/4-chain.jl")
    include("./checkgrad/5-conv1d.jl")
    include("./checkgrad/6-addingProblem.jl")
    include("./checkgrad/7-scaler.jl")
end
