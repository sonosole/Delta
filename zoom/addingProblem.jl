
clear()
model = Chain(
indrnn(2,128,relu),
indrnn(128,64,relu),
Dense(64,1,relu)
)

params = paramsof(model)
optimi = Adam(params;lr=1e-4)

B = 32
T = 100
x, s = addingProblemData(B,T)

epochs = 1000
lossval = zeros(epochs,1)
for e=1:epochs
    resethidden(model)
    for t = 1:T-1
        tmp = forward(model, Variable( reshape(x[:,t,:], 2,B) ) )
        zeroDelta(tmp)
    end
    y = forward(model, Variable( reshape(x[:,T,:], 2,B) ) )
    COST = mseCost(y, Variable( reshape(s,1,B) ) )
    backward()
    update!(optimi, params; clipvalue=1e4)
    zerograds!(params)
    lossval[e] = COST/B
end

lineplot(vec(lossval))


function addingProblemData(N::Int,T::Int)
    #= 两个长度为 T 的序列组成输入：
    第一个序列在 (0,1)范围内均匀采样，
    第二个序列的前两个为 1，其余都为 0.
    例如：
    x1 = 0.90  0.93  0.90  0.56  0.46
    x2 = 1.0   1.0   0.0   0.0   0.0
    而输出为 sum(x1 .* x2) = 0.90 + 0.93
    =#
    @assert (T>1) "The sequence length should lager than 1"
    x1 =  rand(1,T,N)./3
    x2 = zeros(1,T,N)
    x2[1,1,:] .= 1.0
    x2[1,2,:] .= 1.0
    y = sum(x1 .* x2, dims=2)
    return [x1;x2],y
end
