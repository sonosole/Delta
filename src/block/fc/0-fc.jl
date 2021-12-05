export linear
export dense
export MLP
export maxout
export affine
export ResDense
export SelfLoopResNet
export SelfLoopCumulativeResNet
export MeanNormResDense

include("./1-linear.jl")
include("./2-dense.jl")
include("./3-maxout.jl")
include("./4-affine.jl")
include("./5-Res0d.jl")
include("./6-SelfLoopResNet.jl")
include("./7-SelfLoopCumulativeResNet.jl")
include("./8-MeanNormResDense.jl")
