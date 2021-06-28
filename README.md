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
    update(params, learnRate)
    zerograds(params)
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
| ^ | MatOrVec ^ N         | Variable(rand((M,N)) ^ 7.0                          |
| dotAdd    | .+           | dotAdd(Variable(rand((M,N)), Variable(rand((M,N)))  |
| dotMul    | .\*          | dotMul(Variable(rand((M,N)), Variable(rand((M,N)))  |
| matAddVec | mat .+ Vec   | matAddVec(rand(M,N)), Variable(rand(N,1))           |
| matMulVec | mat .* Vec   | matMulVec(rand(M,N)), Variable(rand(N,1))           |

+ **Activation functions:** tan/tand/tanh + sin/sinc/sind/sinpi + log/log2/log10 + exp/exp2/exp10 + cos + swish + relu + P1Relu + leakyrelu + sigmoid + softmax + sqrt + inv + maxout
+ **Aggregation functions :** linearpool + exppool + meanpool + maxpool

## Loss Functions
+ mse
+ crossEntropy
+ binaryCrossEntropy
+ CTC

## Basic Blocks
+ linear
+ dense
+ MLP
+ rnn + RNN(stacked rnn)
+ rin  + RIN(stacked rin)
+ indrnn + INDRNN(stacked indrnn)
+ lstm + LSTM(stacked lstm)
+ residual
+ dropout
+ conv1d
