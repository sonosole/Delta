# Delta

这个项目实现了一些可微分的基础操作，借助这些基础操作，我们可以轻松地搭建任意复杂的神经网络模型，用户只需要定义好网络结构、优化器、损失函数，就可以轻松训练自己的网络。

## 示例代码一
测试 ReLU 激活函数
```julia
m = [ 1. -2.;
     -3.  4.;
      5. -6.]

# 包裹精度为 float64 的 CPU 数组为可微变量
x = Variable(m; keepsgrad=true, type=Array{Float64})

# 对可微变量进行激活操作
y = relu(x)

# 手动指定损失值对激活变量的梯度
y.delta = [-0.1 0.2;
            0.3 0.4;
            0.5 0.6]

# 反向传播
backward()
print(x.delta)
#[-0.1 0.0;
#  0.0 0.4;
#  0.5 0.0]
```

## 示例代码二
使用搭建好的一个多层感知机（多层前馈全连接网络）

```julia
# prepare some data and labels
input = ...
label = ...

# define model and its parameters
topology = [2, 128, 2]
operator = [relu, relu]
mlpmodel = MLP(topology, operator)
params   = paramsof(mlpmodel)

# train the model
for i=1:epoch
    outs = forward(mlpmodel, Variable(input))
    prob = softmax(outs, dims=1)
    COST = crossEntropyCost(prob, Variable(label))
    backward()
    update(params, learnRate)
    zerograds(params)
    println("loss: ", COST)
end
```


## 技术基础 Basic Knowledge Used
+ 全微分
+ 链式法则

## 目前可微分的基本操作 Differentiable Operators

|OPS| Meaning              | Example                                             |
| - | -------------------- | --------------------------------------------------- |
| + |  MatOrVec + Constant | Variable(rand(M,N)) + 7.0                           |
| + |  Constant + MatOrVec | 7.0 + Variable(rand(M,N))                           |
| + |  MatOrVec + MatOrVec | Variable(rand(M,N)) + Variable(rand(M,N))           |
| - |  MatOrVec - Constant | Variable(rand(M,N)) - 7.0                           |
| - |  Constant - MatOrVec | 7.0 - Variable(rand((M,N))                          |
| - |  MatOrVec - MatOrVec | Variable(rand(M,N)) - Variable(rand((M,N))          |
| * |  MatOrVec * Constant | Variable(rand((M,N)) * 7.0                          |
| * |  Constant * MatOrVec | 7.0 * Variable(rand((M,N))                          |
| * |  MatOrVec * MatOrVec | Variable(rand(M,N)) * Variable(rand(N,K))           |
| ^ | MatOrVec ^ N         | Variable(rand((M,N)) ^ 7.0                          |
| dotAdd    | .+           | dotAdd(Variable(rand((M,N)), Variable(rand((M,N)))  |
| dotMul    | .\*          | dotMul(Variable(rand((M,N)), Variable(rand((M,N)))  |
| matAddVec | mat .+ Vec   | matAddVec(rand(M,N)), Variable(rand(N,1))           |
| matMulVec | mat .* Vec   | matMulVec(rand(M,N)), Variable(rand(N,1))           |

+ tan/tand/tanh + sin/sinc/sind/sinpi + log/log2/log10 + exp/exp2/exp10 + cos + swish + relu + P1Relu + leakyrelu + sigmoid + softmax + sqrt + inv + maxout
+ 线性聚合linearpool + 指数聚合exppool + 均值聚合meanpool + 最大值聚合maxpool

## 基础损失函数 Basic Loss Functions
+ mse + crossEntropy + binaryCrossEntropy

## 基础网络结构 Basic Networks
+ dense 层
+ 多层感知机 MLP
+ rnn 层 + RNN(堆叠rnn)
+ rin 层 + RIN(堆叠rin)
+ indrnn 层 + INDRNN(堆叠indrnn)
+ 门控循环结构,lstm + LSTM(堆叠lstm)
+ 残差块 residual
+ 线性映射层 linear
+ dropout 层
+ conv1d

## TODO List
+ 基础网络架构  GRU
+ batchNorm
+ 2D卷积操作
+ CTC on GPU
