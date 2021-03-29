"""
    abstract type Block includes basic network struct like:
    1. dense, MLP
    2. rnn irnn indrnn rin lstm indlstm
    3. RNN IRNN INDRNN RIN LSTM INDLSTM
    4. conv_1xd

"""
abstract type Block end

include("./conv/0-conv.jl")
include(  "./fc/0-fc.jl")
include( "./rnn/0-rnns.jl")
include("./1-chain.jl")
include("./2-residual.jl")
include("./3-dropout.jl")
