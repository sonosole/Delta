Base.__precompile__(true)

module Delta

include("./kit/0-kit.jl")
include("./basic/0-basic.jl")
include("./block/0-block.jl")
include("./loss/0-loss.jl")
include("./optimizer/optimizer.jl")
include("./optimizer/update.jl")
include("./optimizer/regularize.jl")
include("./normalizer/normalizers.jl")

end  # moduleDelta
