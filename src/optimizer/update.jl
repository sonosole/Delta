export clip

function clip(x, clipval)
    x = (abs(x) > clipval) ? clipval * sign(x) : x
end


function update!(var::Variable, lr)
    # update single Variable
    @. var.value -= lr * setNanInfZero(var.delta)
end


function update!(vars::Vector{Variable}, lr)
    # update multi Variables
    for var in vars
        update!(var, lr)
    end
end


function zerograds!(parameters)
    for var in parameters
        if var.delta !== nothing
            var.delta .= 0.0
        end
    end
end


function zerograds!(O::Optimizer)
    for xparam in O.xparams
        c , θ = xparam
        if θ.delta !== nothing
            θ.delta .= 0.0
        end
    end
end
