abstract type Normalizer end

export Normalizer

export MeanNorm
include("./MeanNorm.jl")

export ZNorm, BatchNorm1d, BatchNorm0d
include("./ZNorm.jl")
