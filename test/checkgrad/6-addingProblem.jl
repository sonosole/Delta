@testset "check gradient for IndRNN block in adding problem" begin
    function addingProblemData(T::Int)
        #= 两个长度为 T 的序列组成输入：
        第一个序列在 (0,1)范围内均匀采样，
        第二个序列的前两个为 1，其余都为 0.
        例如：
        x1 = 0.90  0.93  0.90  0.56  0.46
        x2 = 1.0   1.0   0.0   0.0   0.0
        而输出为 sum(x1 .* x2) = 0.90 + 0.93
        =#
        @assert (T>1) "The sequence length should lager than 1"
        x1 = rand(1,T)./3
        x2 = zeros(1,T)
        x2[1] = 1.0
        x2[2] = 1.0
        y = sum(x1 .* x2)
        return [x1;x2],[y]
    end

    TYPE = Array{Float64};
    clear()

    # [0] prepare model
    model = Chain(
    IndRNN(2,128,relu; type=TYPE),
    IndRNN(128,64,cos; type=TYPE),
    IndRNN(64,64,cos;  type=TYPE),
    Dense(64,1,relu;   type=TYPE)
    )

    # [1] prepare input data and its label
    T = 15
    x, s = addingProblemData(T)

    # [2] forward and backward propagation
    resethidden(model)
    for t = 1:T-1
        tmp = forward(model, Variable( reshape(x[:,t], 2,1); type=TYPE) )
        zeroDelta(tmp)
    end
    y = forward(model, Variable( reshape(x[:,T], 2,1); type=TYPE) )
    COST1 = mseCost(y, Variable( reshape(s,1,1); type=TYPE) )
    backward()

    # [3] with a samll change of a weight
    GRAD = model[1].w.delta[1]
    DELTA = 1e-4
    model[1].w.value[1] += DELTA

    # [4] forward and backward propagation again
    resethidden(model)
    for t = 1:T-1
        tmp = forward(model, Variable( reshape(x[:,t], 2,1); type=TYPE) )
        zeroDelta(tmp)
    end
    y = forward(model, Variable( reshape(x[:,T], 2,1); type=TYPE) )
    COST2 = mseCost(y, Variable( reshape(s,1,1); type=TYPE) )
    backward()

    # [5] check if the auto-grad is true or not
    dLdW = (COST2 - COST1)/DELTA
    @test abs(dLdW-GRAD)<1e-4
end
