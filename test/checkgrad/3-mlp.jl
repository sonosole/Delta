@testset "check gradient for MLP and Dense blocks" begin
    clear()
    # [1] prepare input data and its label
    T = Array{Float64}
    x = randn(256, 62)
    l = rand(64, 62)
    l = l ./ sum(l,dims=1)
    m = MLP([256, 128,120,100,80,60, 64], [relu,relu,relu!,leakyrelu,relu,relu]; type=T)

    # [2] forward and backward propagation
    x = Variable(x; type=T);
    l = Variable(l; type=T);

    outs = forward(m, x)
    LOSS1 = mse(outs, l)
    COST1 = cost(LOSS1)
    backward()

    # [3] with a samll change of a weight
    GRAD = m[1].w.delta[1]
    DELTA = 1e-6
    m[1].w.value[1] += DELTA

    # [4] forward and backward propagation
    outs = forward(m, x)
    LOSS2 = mse(outs, l)
    COST2 = cost(LOSS2)
    backward()

    # [5] check if the auto-grad is true or not
    dLdW = (COST2 - COST1)/DELTA;   # numerical gradient
    err  = abs((dLdW-GRAD)/(GRAD+eps(Float64)))*100;  # relative error in %
    err  = err < 1e-1 ? 0.0 : err;
    @test err<1.0
end
