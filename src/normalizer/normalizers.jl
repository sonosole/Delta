abstract type Normalizer end

export Normalizer

export MeanNorm
include("./MeanNorm.jl")

export ZNorm
export BatchNorm0d, BatchNorm1d
include("./ZNorm.jl")

export wnorm
include("./weight-normer.jl")
