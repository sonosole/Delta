"""
    abstract type Block includes basic network struct like:
    1. Dense, MLP
    2. rnn irnn indrnn rin lstm indlstm
    3. RNN IRNN INDRNN RIN LSTM INDLSTM
    4. PlainConv1d

"""
abstract type Block end
export Block
export bytesof
export gradsof
export paramsof
export xparamsof
export nparamsof
export weightsof
export unbiasedof

include("./1-chain.jl")
include("./2-residual.jl")
include("./3-dropout.jl")
include("./4-macro.jl")

include("./conv/0-conv.jl")
include(  "./fc/0-fc.jl")
include( "./rnn/0-rnns.jl")
