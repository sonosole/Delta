abstract type Normalizer end

export Normalizer

export MeanNorm
include("./MeanNorm.jl")

using Statistics
export ZNorm, BatchNorm1d, BatchNorm0d
include("./ZNorm.jl")
