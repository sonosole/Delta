# Delta

**Delta** is a neural network modeling framework that can easily build any complex neural network models using many differentiable operators.

## Example
Build a multi-layer neural network (e.g. MLP)

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
    update!(params, learnRate)
    zerograds!(params)
    println("loss: ", COST)
end
```
## Cases

### Automatic Speech Recognition (ASR)
- [Training **ASR** Model on part of thchs30 dataset](https://github.com/sonosole/ASR-TH30-Demo)


## Differentiable Operators

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
| ^ |  MatOrVec ^ N        | Variable(rand((M,N)) ^ 7.0                          |
| .+|  dotAdd              | Variable(rand(M,N,K) .+ Variable(rand(M,1,1)        |
| .-|  dotMinus            | Variable(rand(M,N,K) .- Variable(rand(1,M,1)        |
| .*|  dotMul              | Variable(rand(M,N) .* Variable(rand(M)              |
| ./|  dotDiv              | Variable(rand(M,1) ./ Variable(rand(M,1)            |
| matAddVec | mat .+ Vec   | matAddVec(rand(M,N)), Variable(rand(N,1))           |
| matMulVec | mat .* Vec   | matMulVec(rand(M,N)), Variable(rand(N,1))           |

+ **Activation functions :** tan/tand/tanh + sin/sinc/sind/sinpi + log/log2/log10 + exp/exp2/exp10 + cos + swish + relu/P1Relu/leakyrelu/relu1/relu6 + min2max + sigmoid + softmax + sqrt + inv + Maxout
+ **Aggregation functions :** linearpool + exppool + mean + maximum + minimum + sum + maxmin + minmax

## Loss Functions
+ maeLoss (L1Loss)
+ mseLoss (L2Loss)
+ LpLoss
+ crossEntropy
+ binaryCrossEntropy
+ CTC
+ TCS

## Basic Blocks
+ Linear
+ Dense
+ MLP (stacked Dense)
+ RNN + RNNs(stacked RNN)
+ lstm + LSTM(stacked lstm)
+ rin  + RIN(stacked rin)
+ IndRNN + IndRNNs(stacked IndRNN)
+ Residual
+ dropout
+ PlainConv1d
+ Res0d + Res0dWithBN
+ SelfLoopResNet + SelfLoopCumulativeResNet
+ MeanNormResDense

## Optimizers
+ SGD, SGDL1, SGDL2, SGDL1L2
+ Momentum, MomentumL1, MomentumL2, MomentumL1L2
+ Adam, AdamL1, AdamL2, AdamL1L2
+ AdaGrad, AdaGradL1, AdaGradL2, AdaGradL1L2
+ RMSProp, RMSPropL1, RMSPropL2, RMSPropL1L2

## Normilizers
+ ZNorm
+ BatchNorm0d
+ BatchNorm1d
+ MeanNorm

## Scalers
+ ScaleChannels
+ ScalePath
+ SwitchPath
