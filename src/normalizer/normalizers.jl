abstract type Normalizer end

export Normalizer

export MeanNorm
include("./MeanNorm.jl")

using Statistics
export ZNorm
include("./ZNorm.jl")
