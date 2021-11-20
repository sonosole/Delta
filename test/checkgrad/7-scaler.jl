clear()
@testset "check ScalePath op's gradient" begin
    # [1] prepare input data and its label
    input = randn(19,17,7);
    label = randn(19,17,7);
    x = Variable(input, type=Array{Float64});
    l = Variable(label, type=Array{Float64});

    # [2] prepare a learnabel scalar multiplier
    k = ScalePath(1.0, ndims=3, type=Array{Float64});

    # [3] forward and get gradient by backpropagation
    y  = forward(k, x);
    C₁ = mseCost(y, l);
    backward();
    ∇ = Delta.δ(k.scale)[1];  # backpropagation gradient


    # [4] with a samll change of weight, forward again
    Δ = 1e-6;
    Delta.ᵛ(k.scale)[1] += Δ;

    y  = forward(k, x);
    C₂ = mseCost(y, l);
    backward();

    # [5] check if the auto-grad is true or not
    ϵ    = eps(Float64)
    ∂C∂k = (C₂ - C₁)/Δ;            # numerical gradient
    err  = (∂C∂k - ∇)/(∇+ϵ);       # relative error
    err  = abs(err) * 100;         # relative error in %
    err  = err < 1e-1 ? 0.0 : err; # zero-error if under 0.1%
    @test err < 1.0
end
