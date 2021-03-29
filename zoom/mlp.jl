#=
in this example, we just put all the data into the
network at once over and over again to optimizes the
parameters of network.
=#

using UnicodePlots

# first class data and lable
input1 = 3*randn(2,2000);
label1 = zeros(2,2000);
label1[1,:] .= 1.0;

# second class data and lable
t = 0:pi/1000:2*pi;
input2 = zeros(2,length(t));
label2 = zeros(2,length(t));
label2[2,:] .= 1.0;
for i = 1:length(t)
    r = randn() + 17.0;
    input2[1,i] = r * cos(t[i]);
    input2[2,i] = r * sin(t[i]);
end

# make model's input and label
input = hcat(input1,input2)
label = hcat(label1,label2)
x = Variable(input)
l = Variable(label)
N = size(x,2)

dst = densityplot(input2[1,:],input2[2,:],color=:red);
densityplot!(dst,input1[1,:],input1[2,:],color=:yellow)


epoch = 100
lrate = 1e-5

# parameters for constructing a Multi-Layer-Perceptron
topology  = [2, 32,32,32, 2]
operator  = [relu, relu, relu, relu]

# train a model with sgd
mlpmodel  = MLP(topology, operator)
parameter = paramsof(mlpmodel)
lossval1  = zeros(epoch,1)
tic = time()
for i = 1:epoch
    outs = forward(mlpmodel, x)
    COST = crossEntropyCost(softmax(outs;dims=1), l)
    backward()
    update(parameter, lrate)
    zerograds(parameter)
    lossval1[i] = COST/N
end
toc = time()
println("\n======== SGD =========")
println(" time: ", toc-tic," sec")
println(" loss: ", lossval1[end])


# train a model with Momentum optimizer
mlpmodel  = MLP(topology, operator)
parameter = paramsof(mlpmodel)
optimizer = Momentum(parameter;learnRate=lrate,inertia=0.9)
lossval2  = zeros(epoch,1)

tic = time()
for i = 1:epoch
    outs = forward(mlpmodel, x)
    COST = crossEntropyCost(softmax(outs;dims=1), l)
    backward()
    update(optimizer,parameter)
    zerograds(parameter)
    lossval2[i] = COST/N
end
toc = time()
println("\n======== Momentum =========")
println(" time: ", toc-tic," sec")
println(" loss: ", lossval2[end])


# train a model with Adam optimizer
mlpmodel  = MLP(topology, operator)
parameter = paramsof(mlpmodel)
optimizer = Adam(parameter;learnRate=1e-3,b1=0.9,b2=0.9)
lossval3  = zeros(epoch,1)
tic = time()
for i = 1:epoch
    outs = forward(mlpmodel, x)
    COST = crossEntropyCost(softmax(outs;dims=1), l)
    backward()
    update(optimizer,parameter)
    zerograds(parameter)
    lossval3[i] = COST/N
end
toc = time()
println("\n======== Adam =========")
println(" time: ", toc-tic," sec")
println(" loss: ", lossval3[end])


p1 = lineplot(vec(log.(lossval1)),color =:red,name ="sgd")
p2 = lineplot(vec(log.(lossval2)),color =:yellow,name ="Momentum")
p3 = lineplot(vec(log.(lossval3)),color =:green,name ="Adam")

plt = lineplot(vec(log.(lossval1)),color =:red,name ="sgd")
lineplot!(plt,vec(log.(lossval2)),color =:yellow,name ="Momentum")
lineplot!(plt,vec(log.(lossval3)),color =:green,name ="Adam")



# train a model with Adam optimizer with Dropout
clear()
mlpmodel = Chain(
    dense(2,32),
    dropout(0.1),
    dense(32,32),
    dropout(0.1),
    dense(32,32),
    dropout(0.1),
    dense(32,2)
)
parameter = paramsof(mlpmodel)
optimizer = Adam(parameter;learnRate=1e-3,b1=0.9,b2=0.999)
lossval4  = zeros(epoch,1)
tic = time()
for i = 1:epoch
    outs = forward(mlpmodel, x)
    COST = crossEntropyCost(softmax(outs;dims=1), l)
    backward()
    update(optimizer, parameter)
    zerograds(parameter)
    lossval4[i] = COST/N
end
toc = time()
println("\n======== Adam with Dropout =========")
println(" time: ", toc-tic," sec")
println(" loss: ", lossval4[end])
p4 = lineplot(vec(log.(lossval4)),color =:white,name ="Adam with Dropout")
