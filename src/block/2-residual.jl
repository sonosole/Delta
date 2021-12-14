export Residual


mutable struct Residual
    blocks::Vector
    function Residual(sequence::Vector)
        blocknum = length(sequence)
        blocks = Vector(undef,blocknum)
        for i = 1:blocknum
            blocks[i] = sequence[i]
        end
        new(blocks)
    end
    function Residual(sequence...)
        blocknum = length(sequence)
        blocks = Vector(undef,blocknum)
        for i = 1:blocknum
            blocks[i] = sequence[i]
        end
        new(blocks)
    end
end


function paramsof(m::Residual)
    params = Vector{Variable}(undef,0)
    for i = 1:length(m.blocks)
        append!(params, paramsof(m.blocks[i]))
    end
    return params
end


function xparamsof(m::Residual)
    params = Vector{XVariable}(undef,0)
    for i = 1:length(m.blocks)
        append!(params, xparamsof(m.blocks[i]))
    end
    return params
end


function forward(r::Residual, input::Variable)
    x = forward(r.blocks[1], input)
    for i = 2:length(r.blocks)
        x = forward(r.blocks[i], x)
    end
    @assert size(x)==size(input) "the input's dims should be the same as output's dims"
    return x + input
end


function predict(r::Residual, input)
    x = predict(r.blocks[1], input)
    for i = 2:length(r.blocks)
        x = predict(r.blocks[i], x)
    end
    return x + input
end



function nparamsof(r::Residual)
    c = 0
    for i = 1:length(r.blocks)
        c += nparamsof(r.blocks[i])
    end
    return c
end
