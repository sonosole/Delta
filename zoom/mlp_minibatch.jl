#=
in this example, we just put a batch of the data into the
network at once over and over again to optimizes the
parameters of network.
=#

using Gadfly
using Random

# first class data and label
input1 = 3*randn(2,2000);
label1 = zeros(2,2000);
label1[1,:] .= 1.0;

# second class data and label
t = 0:pi/1000:2*pi;
input2 = zeros(2,length(t));
label2 = zeros(2,length(t));
label2[2,:] .= 1.0;
for i = 1:length(t)
    r = randn() + 16.0;
    input2[1,i] = r * cos(t[i]);
    input2[2,i] = r * sin(t[i]);
end

input = hcat(input1,input2)
label = hcat(label1,label2)


batch = 2048
epoch = 1000
lrate = 1e-4
topology = [2, 32,32,32, 2]
operator = [relu, relu, relu, relu]
indexs   = collect(1:batch)


mlpmodel = MLP(topology, operator)
paramter = paramsof(mlpmodel)
lossval  = zeros(epoch,1)
tic = time()
for i = 1:epoch
    ids  = shuffle!(indexs)
    outs = forward(mlpmodel, Variable(input[:,ids[1:batch]]))
    outs = softmax(outs; dims=1)
    COST = crossEntropyCost(outs, Variable(label[:,ids[1:batch]]))
    backward()
    update!(paramter,lrate)
    zerograds!(paramter)
    lossval[i] = COST
end
toc = time()
println("\n------- sgd optimizer loss: ", lossval[end])
l1 = layer(y=log.(lossval), Geom.line,Theme(default_color=colorant"cyan"))


mlpmodel  = MLP(topology, operator)
paramter  = paramsof(mlpmodel)
optimizer = Momentum(paramter,lr=lrate,inertia=0.9)
lossval   = zeros(epoch,1)
tic = time()
for i = 1:epoch
    ids  = shuffle!(indexs)
    outs = forward(mlpmodel, Variable(input[:,ids[1:batch]]))
    outs = softmax(outs; dims=1)
    COST = crossEntropyCost(outs, Variable(label[:,ids[1:batch]]))
    backward()
    update!(optimizer,paramter)
    zerograds!(paramter)
    lossval[i] = COST
end
toc = time()
println("-- Momentum optimizer loss: ", lossval[end])
l2 = layer(y=log.(lossval), Geom.line,Theme(default_color=colorant"yellow"))

mlpmodel  = MLP(topology, operator)
paramter  = paramsof(mlpmodel)
optimizer = Adam(paramter,lr=1e-3)
lossval   = zeros(epoch,1)
tic = time()
for i = 1:epoch
    ids  = shuffle!(indexs)
    outs = forward(mlpmodel, Variable(input[:,ids[1:batch]]))
    outs = softmax(outs; dims=1)
    COST = crossEntropyCost(outs, Variable(label[:,ids[1:batch]]))
    backward()
    update!(optimizer,paramter)
    zerograds!(paramter)
    lossval[i] = COST
end
toc = time()
println("------ Adma optimizer loss: ", lossval[end])
l3 = layer(y=log.(lossval), Geom.line,Theme(default_color=colorant"red"))

plot(l1,l2,l3)



mlpmodel  = Chain(
    maxout(2,32,k=3),
    residual(Dense(32,32),Dense(32,32),Linear(32,32)),
    residual(Dense(32,32),Dense(32,32),Linear(32,32)),
    residual(Dense(32,32),Dense(32,32),Linear(32,32)),
    Dense(32,2)
)

epoch = 70
parameter = paramsof(mlpmodel)
lossval  = zeros(epoch,1)
for i = 1:epoch
    ids  = shuffle!(indexs)
    outs = forward(mlpmodel, Variable(input[:,ids[1:batch]]))
    outs = softmax(outs; dims=1)
    COST = crossEntropyCost(outs, Variable(label[:,ids[1:batch]]))
    backward()
    update!(parameter,lrate)
    zerograds!(parameter)
    lossval[i] = COST
end
println(" sgd optimizer loss: ", lossval[end])
