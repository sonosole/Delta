mutable struct indlstm <: Block
    # input control gate params
    wi::Variable
    ui::Variable
    bi::Variable
    # forget control gate params
    wf::Variable
    uf::Variable
    bf::Variable
    # out control gate params
    wo::Variable
    uo::Variable
    bo::Variable
    # new cell info params
    wc::Variable
    uc::Variable
    bc::Variable
    h  # hidden variable
    c  #   cell variable
    function indlstm(inputSize::Int, hiddenSize::Int; type::Type=Array{Float32})
        T  = eltype(type)
        A  = T(1E-1)

        wi = randn(T, hiddenSize, inputSize) .* sqrt( T(2/inputSize) )
        ui = randn(T, hiddenSize, 1) .* A
        bi = 3ones(T, hiddenSize, 1)

        wf = randn(T, hiddenSize, inputSize) .* sqrt( T(2/inputSize) )
        uf = randn(T, hiddenSize, 1) .* A
        bf =-3ones(T, hiddenSize, 1)

        wo = randn(T, hiddenSize, inputSize) .* sqrt( T(2/inputSize) )
        uo = randn(T, hiddenSize, 1) .* A
        bo = 3ones(T, hiddenSize, 1)

        wc = randn(T, hiddenSize, inputSize) .* sqrt( T(2/inputSize) )
        uc = randn(T, hiddenSize, 1) .* A
        bc = zeros(T, hiddenSize, 1)

        new(Variable{type}(wi,true,true,true), Variable{type}(ui,true,true,true), Variable{type}(bi,true,true,true),
            Variable{type}(wf,true,true,true), Variable{type}(uf,true,true,true), Variable{type}(bf,true,true,true),
            Variable{type}(wo,true,true,true), Variable{type}(uo,true,true,true), Variable{type}(bo,true,true,true),
            Variable{type}(wc,true,true,true), Variable{type}(uc,true,true,true), Variable{type}(bc,true,true,true), nothing, nothing)
    end
end


mutable struct INDLSTM <: Block
    layers::Vector{indlstm}
    function INDLSTM(topology::Vector{Int}; type::Type=Array{Float32})
        n = length(topology) - 1
        layers = Vector{indlstm}(undef, n)
        for i = 1:n
            layers[i] = indlstm(topology[i], topology[i+1]; type=type)
        end
        new(layers)
    end
end


Base.getindex(m::INDLSTM,     k...) =  m.layers[k...]
Base.setindex!(m::INDLSTM, v, k...) = (m.layers[k...] = v)
Base.length(m::INDLSTM)       = length(m.layers)
Base.lastindex(m::INDLSTM)    = length(m.layers)
Base.firstindex(m::INDLSTM)   = 1
Base.iterate(m::INDLSTM, i=firstindex(m)) = i>length(m) ? nothing : (m[i], i+1)


function Base.show(io::IO, m::indlstm)
    SIZE = size(m.wi)
    TYPE = typeof(m.wi.value)
    print(io, "indlstm($(SIZE[2]), $(SIZE[1]); type=$TYPE)")
end


function Base.show(io::IO, model::INDLSTM)
    for m in model
        show(io, m)
    end
end


function resethidden(model::indlstm)
    model.h = nothing
    model.c = nothing
end


function resethidden(model::INDLSTM)
    for m in model
        resethidden(m)
    end
end


function forward(model::indlstm, x::Variable{T}) where T
    wi = model.wi
    ui = model.ui
    bi = model.bi

    wf = model.wf
    uf = model.uf
    bf = model.bf

    wo = model.wo
    uo = model.uo
    bo = model.bo

    wc = model.wc
    uc = model.uc
    bc = model.bc

    h = model.h ≠ nothing ? model.h : Variable{T}(Zeros(T, size(wi,1), size(x,2)))
    c = model.c ≠ nothing ? model.c : Variable{T}(Zeros(T, size(wc,1), size(x,2)))

    z = tanh(    matAddVec(wc * x + matMulVec(h,uc), bc) )
    i = sigmoid( matAddVec(wi * x + matMulVec(h,ui), bi) )
    f = sigmoid( matAddVec(wf * x + matMulVec(h,uf), bf) )
    o = sigmoid( matAddVec(wo * x + matMulVec(h,uo), bo) )
    c = dotMul(f, c) + dotMul(i, z)
    h = dotMul(o, tanh(c))

    model.c = c
    model.h = h

    return h
end


function forward(model::INDLSTM, x::Variable)
    for m in model
        x = forward(m, x)
    end
    return x
end


function predict(model::indlstm, x::T) where T
    wi = model.wi.value
    ui = model.ui.value
    bi = model.bi.value

    wf = model.wf.value
    uf = model.uf.value
    bf = model.bf.value

    wo = model.wo.value
    uo = model.uo.value
    bo = model.bo.value

    wc = model.wc.value
    uc = model.uc.value
    bc = model.bc.value

    h = model.h ≠ nothing ? model.h : Zeros(T, size(wi,1), size(x,2))
    c = model.c ≠ nothing ? model.c : Zeros(T, size(wc,1), size(x,2))

    z = tanh(    wc * x + h .* uc .+ bc )
    i = sigmoid( wi * x + h .* ui .+ bi )
    f = sigmoid( wf * x + h .* uf .+ bf )
    o = sigmoid( wo * x + h .* uo .+ bo )
    c = f .* c + i .* z
    h = o .* tanh(c)

    model.c = c
    model.h = h

    return h
end


function predict(model::INDLSTM, x)
    for m in model
        x = predict(m, x)
    end
    return x
end


"""
    unbiasedof(m::indlstm)

unbiased weights of indlstm block
"""
function unbiasedof(m::indlstm)
    weights = Vector(undef, 4)
    weights[1] = m.wi.value
    weights[2] = m.wf.value
    weights[3] = m.wo.value
    weights[4] = m.wc.value
    return weights
end


function weightsof(m::indlstm)
    weights = Vector{Variable}(undef,12)
    weights[1] = m.wi.value
    weights[2] = m.ui.value
    weights[3] = m.bi.value

    weights[4] = m.wf.value
    weights[5] = m.uf.value
    weights[6] = m.bf.value

    weights[7] = m.wo.value
    weights[8] = m.uo.value
    weights[9] = m.bo.value

    weights[10] = m.wc.value
    weights[11] = m.uc.value
    weights[12] = m.bc.value
    return weights
end


"""
    unbiasedof(model::INDLSTM)

unbiased weights of INDLSTM block
"""
function unbiasedof(model::INDLSTM)
    weights = Vector(undef, 0)
    for m in model
        append!(weights, unbiasedof(m))
    end
    return weights
end


function weightsof(model::INDLSTM)
    weights = Vector(undef,0)
    for m in model
        append!(weights, weightsof(m))
    end
    return weights
end


function gradsof(m::indlstm)
    grads = Vector{Variable}(undef,12)
    grads[1] = m.wi.delta
    grads[2] = m.ui.delta
    grads[3] = m.bi.delta

    grads[4] = m.wf.delta
    grads[5] = m.uf.delta
    grads[6] = m.bf.delta

    grads[7] = m.wo.delta
    grads[8] = m.uo.delta
    grads[9] = m.bo.delta

    grads[10] = m.wc.delta
    grads[11] = m.uc.delta
    grads[12] = m.bc.delta
    return grads
end


function gradsof(model::INDLSTM)
    grads = Vector(undef,0)
    for m in model
        append!(grads, gradsof(m))
    end
    return grads
end


function zerograds!(m::indlstm)
    for v in gradsof(m)
        v .= zero(v)
    end
end


function zerograds!(m::INDLSTM)
    for v in gradsof(m)
        v .= zero(v)
    end
end


function paramsof(m::indlstm)
    params = Vector{Variable}(undef,12)
    params[1] = m.wi
    params[2] = m.ui
    params[3] = m.bi

    params[4] = m.wf
    params[5] = m.uf
    params[6] = m.bf

    params[7] = m.wo
    params[8] = m.uo
    params[9] = m.bo

    params[10] = m.wc
    params[11] = m.uc
    params[12] = m.bc
    return params
end


function paramsof(model::INDLSTM)
    params = Vector{Variable}(undef,0)
    for m in model
        append!(params, paramsof(m))
    end
    return params
end


function nparamsof(m::indlstm)
    lw = length(m.wi)
    lu = length(m.ui)
    lb = length(m.bi)
    return (lw+lu+lb)*4
end


function nparamsof(model::INDLSTM)
    num = 0
    for m in model
        num += nparamsof(m)
    end
    return num
end



function to(type::Type, m::indlstm)
    m.wi = to(type, m.wi)
    m.ui = to(type, m.ui)
    m.bi = to(type, m.bi)

    m.wf = to(type, m.wf)
    m.uf = to(type, m.uf)
    m.bf = to(type, m.bf)

    m.wo = to(type, m.wo)
    m.uo = to(type, m.uo)
    m.bo = to(type, m.bo)

    m.wc = to(type, m.wc)
    m.uc = to(type, m.uc)
    m.bc = to(type, m.bc)
    return m
end


function to!(type::Type, m::indlstm)
    m = to(type, m)
    return nothing
end


function to(type::Type, m::INDLSTM)
    for layer in m
        layer = to(type, layer)
    end
    return m
end


function to!(type::Type, m::INDLSTM)
    for layer in m
        to!(type, layer)
    end
end
