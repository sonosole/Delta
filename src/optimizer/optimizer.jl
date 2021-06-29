export Optimizer
export Descent
export Momentum
export Adam
export decay
export normclip


abstract type Optimizer end


mutable struct Descent <: Optimizer
    lr::Union{Float32,Float64}
    decay::Union{Float32,Float64}
    name::String
    function Descent(;learnRate=1e-4,decay=1.0)
        new(learnRate, decay, "Descent")
    end
end

# pretty printing
function Base.show(io::IO, x::Descent)
    print("Descent(learnRate=$(x.lr), decay=$(x.decay))")
end


function update(d::Descent, params::Vector{Variable})
    lrate = d.lr
    d.lr *= d.decay
    update(params, lrate)
end


mutable struct Momentum <: Optimizer
    v::Vector
    lr::Union{Float32,Float64}
    p::Union{Float32,Float64}
    decay::Union{Float32,Float64}
    name::String
    function Momentum(params::Vector{Variable};
                      learnRate=1e-4,
                      inertia=0.9,
                      decay=1.0)
        num = length(params)
        vel = Vector(undef,num)
        for i = 1:num
           vel[i] = Zeros(typeof(params[i].value), params[i].shape)
        end
        new(vel, learnRate, inertia, decay, "Momentum")
    end
end


function Base.show(io::IO, x::Momentum)
    print("Momentum(learnRate=$(x.lr), inertia=$(x.p), decay=$(x.decay))")
end


function update(m::Momentum, params::Vector{Variable}; clipvalue=1e4)
    vel = m.v
    lr  = m.lr
    p   = m.p
    m.lr *= m.decay
    for i = 1:length(params)
        @. vel[i] = p * vel[i] + clip(params[i].delta, clipvalue)
        @. params[i].value -= lr * vel[i]
    end
end


mutable struct Adam <: Optimizer
    w1::Vector
    w2::Vector
    lr::Union{Float32,Float64}
    b1::Union{Float32,Float64}
    b2::Union{Float32,Float64}
    ϵ::Union{Float32,Float64}
    t::UInt
    b1t::Union{Float32,Float64}
    b2t::Union{Float32,Float64}
    decay::Union{Float32,Float64}
    name::String
    function Adam(params::Vector{Variable}; learnRate=1e-4, b1=0.9, b2=0.996, epsilon=1e-8, decay=1.0)
        num = length(params)
        w1  = Vector(undef,num)
        w2  = Vector(undef,num)
        for i = 1:num
            w1[i] = Zeros(typeof(params[i].value), params[i].shape)
            w2[i] = Zeros(typeof(params[i].value), params[i].shape)
        end
        new(w1,w2,learnRate, b1, b2, epsilon, 0, 1.0, 1.0, decay, "Adam")
    end
end


function Base.show(io::IO, x::Adam)
    print("Adam(learnRate=$(x.lr), β₁=$(x.b1), β₂=$(x.b2), ϵ=$(x.ϵ) decay=$(x.decay))");
end


function update(a::Adam, params::Vector{Variable}; clipvalue=1.0)
    w₁ = a.w1
    w₂ = a.w2
    lr = a.lr
    b₁ = a.b1
    b₂ = a.b2
    ϵ  = a.ϵ

    a.t   += 1
    a.b1t *= b₁
    a.b2t *= b₂
    a.lr  *= a.decay
    b₁ᵗ = a.b1t
    b₂ᵗ = a.b2t

    for i = 1:length(params)
        μ = sqrt(1-b₂ᵗ) / (1-b₁ᵗ) * lr
        ∇ = clip.(params[i].delta, clipvalue)
        @. w₁[i] = b₁ * w₁[i] + (1-b₁) * ∇
        @. w₂[i] = b₂ * w₂[i] + (1-b₂) * ∇ * ∇
        @. params[i].value -= μ * w₁[i] / sqrt(w₂[i] + ϵ)
    end
end


function decay(params::Vector{Variable}; ratio=0.999)
    for p in params
        p.value .*= ratio
    end
end


function normclip(gradient, clipvalue)
    L2NormVal  = sqrt( sum(gradient .^ 2) / length(gradient) )
    Normalizer = clipvalue / L2NormVal
    if L2NormVal > clipvalue
        gradient .*= Normalizer
    end
    return gradient
end
