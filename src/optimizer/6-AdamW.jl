mutable struct AdamW <: Optimizer
    w1::Vector
    w2::Vector
    lr::AbstractFloat
    b1::AbstractFloat
    b2::AbstractFloat
    ϵ::AbstractFloat
    t::UInt
    b1t::AbstractFloat
    b2t::AbstractFloat
    lrdecay::AbstractFloat
    wtdecay::AbstractFloat
    name::String
    function AdamW(params::Vector{Variable}; lr=1e-3, b1=0.9, b2=0.999, epsilon=1e-8, lrdecay=1.0, wtdecay=0.01)
        num = length(params)
        w1  = Vector(undef,num)
        w2  = Vector(undef,num)
        for i = 1:num
            w1[i] = Zeros(typeof(params[i].value), params[i].shape)
            w2[i] = Zeros(typeof(params[i].value), params[i].shape)
        end
        new(w1,w2,lr, b1, b2, epsilon, 0, 1.0, 1.0, lrdecay, wtdecay, "AdamW")
    end
end


function Base.show(io::IO, A::AdamW)
    print("AdamW(lr=$(A.lr), β₁=$(A.b1), β₂=$(A.b2), ϵ=$(A.ϵ) lrdecay=$(A.lrdecay))");
end


function update!(a::AdamW, params::Vector{Variable}; clipfn::Function=LPInfNormClip, clipvalue=10.0)
    w₁ = a.w1
    w₂ = a.w2
    lr = a.lr
    b₁ = a.b1
    b₂ = a.b2
    ϵ  = a.ϵ
    λ  = a.wtdecay
    a.t   += 1
    a.b1t *= b₁
    a.b2t *= b₂
    a.lr  *= a.lrdecay
    b₁ᵗ = a.b1t
    b₂ᵗ = a.b2t
    # biased or momery weights should not use weight decay strategies
    for i = 1:length(params)
        μ = - sqrt(1-b₂ᵗ) / (1-b₁ᵗ) * lr
        ∇ = clipfn(setNanInfZero(params[i].delta), clipvalue)
        @. w₁[i] = b₁ * w₁[i] + (1-b₁) * ∇
        @. w₂[i] = b₂ * w₂[i] + (1-b₂) * ∇ * ∇
        @. params[i].value += μ * (w₁[i] / sqrt(w₂[i] + ϵ) + λ * params[i].value)
    end
end
