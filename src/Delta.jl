Base.__precompile__(true)

module Delta

using Statistics: mean, std
import Statistics.mean

include("./kit/0-kit.jl")
include("./basic/0-basic.jl")
include("./block/0-block.jl")
include("./loss/0-loss.jl")
include("./optimizer/optimizer.jl")
include("./optimizer/update.jl")
include("./optimizer/regularize.jl")
include("./normalizer/normalizers.jl")
include("./scaler/Scalers.jl")

end  # moduleDelta
