export Dropout


mutable struct Dropout <: Block
    p # Dropout probibility
    Dropout(   ) = new(0.1)
    Dropout(pro) = new(pro)
end


function paramsof(m::Dropout)
    return nothing
end


function xparamsof(m::Dropout)
    return nothing
end


function forward(d::Dropout, x::Variable{T}) where T
    # 对网络激活节点进行灭活
    # 属于in-place操作,但是输入输出共享节点值引用
    type = eltype(x)
    𝟙 = type(1.0)
    𝕡 = type(d.p)
    𝕄 = (rand(type, x.shape) .< (𝟙 - 𝕡)) .* (𝟙/(𝟙 - 𝕡)) # mask
    x.value .*= 𝕄
    y = Variable{T}(ᵛ(x), x.backprop)
    if x.backprop
        function dropoutBackward()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* 𝕄
            end
            ifNotKeepδThenFreeδ!(y);
        end
        push!(graph.backward, dropoutBackward)
    end
    return y
end


function predict(d::Dropout, input)
    return input
end


function nparamsof(m::Dropout)
    return 0
end
