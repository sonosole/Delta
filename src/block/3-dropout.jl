export dropout


mutable struct dropout <: Block
    p # dropout probibility
    dropout(   ) = new(0.1)
    dropout(pro) = new(pro)
end


function paramsof(m::dropout)
    return nothing
end


function xparamsof(m::dropout)
    return nothing
end


function forward(d::dropout, var::Variable{T}) where T
    # 对网络激活节点进行灭活
    # 属于in-place操作,但是输入输出共享节点值引用
    type = eltype(var)
    prob = type(d.p)
    one  = type(1.0)
    RandMask = (rand(type, var.shape) .< (one - prob)) .* (one/(one - prob))
    var.value .*= RandMask
    out = Variable{T}(var.value, var.backprop)
    if var.backprop
        function dropoutBackward()
            if need2computeδ!(var)
                var.delta += RandMask .* out.delta
            end
            ifNotKeepδThenFreeδ!(out);
        end
        push!(graph.backward, dropoutBackward)
    end
    return out
end


function predict(d::dropout, input)
    return input
end


function nparamsof(m::dropout)
    return 0
end
