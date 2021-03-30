export residual


mutable struct residual
    blocks::Vector
    function residual(sequence::Vector)
        blocknum = length(sequence)
        blocks = Vector(undef,blocknum)
        for i = 1:blocknum
            blocks[i] = sequence[i]
        end
        new(blocks)
    end
    function residual(sequence...)
        blocknum = length(sequence)
        blocks = Vector(undef,blocknum)
        for i = 1:blocknum
            blocks[i] = sequence[i]
        end
        new(blocks)
    end
end


function paramsof(m::residual)
    params = Vector{Variable}(undef,0)
    for i = 1:length(m.blocks)
        append!(params, paramsof(m.blocks[i]))
    end
    return params
end


function forward(r::residual, input::Variable)
    x = forward(r.blocks[1], input)
    for i = 2:length(r.blocks)
        x = forward(r.blocks[i], x)
    end
    @assert size(x)==size(input) "the input's dims should be the same as output's dims"
    return x + input
end


function predict(r::residual, input)
    x = predict(r.blocks[1], input)
    for i = 2:length(r.blocks)
        x = predict(r.blocks[i], x)
    end
    return x + input
end



function nparamsof(r::residual)
    c = 0
    for i = 1:length(r.blocks)
        c += nparamsof(r.blocks[i])
    end
    return c
end
