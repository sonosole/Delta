@testset "check gradient for Chain" begin
    clear()

    # [1] prepare input data and its label
    TYPE = Array{Float64}
    x = randn(256, 62)
    l = rand(64, 62)
    l = l ./ sum(l, dims=1)
    x = Variable(x; type=TYPE)
    l = Variable(l; type=TYPE)

    blocks = [
        Dense(256,128,tanh; type=TYPE),
        Dense(128,128,cos;  type=TYPE),
        Dense(128,128,sin;  type=TYPE),
        Dense(128,128,cos!; type=TYPE),
        Dense(128,128,sin!; type=TYPE),
        Dense(128, 64,relu; type=TYPE),
        Maxout(64, 64; k=3, type=TYPE),
        residual(
            Dense(64,32,sin;type=TYPE),
            Linear(32,64,   type=TYPE)
        )
    ]

    model = Chain(blocks)

    # [2] forward and backward propagation
    outs = softmax(forward(model, x); dims=1)
    COST1 = crossEntropyCost(outs, l)
    backward()

    # [3] with a samll change of a weight
    DELTA = 1e-5
    model[1].w.value[1] += DELTA
    GRAD = blocks[1].w.delta[1]

    # [4] forward and backward propagation
    outs = softmax(forward(model, x); dims=1)
    COST2 = crossEntropyCost(outs, l)
    backward()

    # [5] check if the auto-grad is true or not
    dLdW = (COST2 - COST1)/DELTA
    err  = abs((dLdW-GRAD)/(GRAD==0.0 ? 1.0 : GRAD))*100
    err  = err < 1e-3 ? 0.0 : err
    @test err < 5.0
end
