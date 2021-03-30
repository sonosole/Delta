export clip

function clip(x, clipval)
    x = (abs(x) > clipval) ? clipval * sign(x) : x
end


function update(var::Variable, lr)
    # update single Variable
    @. var.value -= lr * var.delta
end


function update(vars::Vector{Variable}, lr)
    # update multi Variables
    for var in vars
        update(var, lr)
    end
end


function zerograds(parameters)
    for v in parameters
        v.delta .= 0.0
    end
end
