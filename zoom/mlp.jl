#=
in this example, we just put all the data into the
network at once over and over again to optimizes the
parameters of network.
=#

# first class data and lable
using Plots
input1 = 3*randn(2,1024);
label1 = zeros(2,1024);
label1[1,:] .= 1.0;

# second class data and lable
t = 0:pi/1000:2*pi;
input2 = zeros(2,length(t));
label2 = zeros(2,length(t));
label2[2,:] .= 1.0;
for i = 1:length(t)
    r = randn() + 18.0;
    input2[1,i] = r * cos(t[i]);
    input2[2,i] = r * sin(t[i]);
end

# make model's input and label
input = hcat(input1,input2)
label = hcat(label1,label2)
x = Variable(input);
l = Variable(label);
N = size(x,2)

scatter(input2[1,:],input2[2,:],color=:blue)
scatter!(input1[1,:],input1[2,:],color=:red)


function runme(epoch, lr, opt)
    mlpmodel  = Chain(dense(2,32),
                      dense(32,32),
                      dense(32,32),
                      linear(32,2))

    params  = xparamsof(mlpmodel)
    optim   = opt(params;lr=lr)
    lossval = zeros(epoch,1)

    for i = 1:epoch
        outs = forward(mlpmodel, x)
        COST = crossEntropyCost(softmax(outs;dims=1), l)
        backward()
        update!(optim)
        zerograds!(optim)
        lossval[i] = COST/N
    end
    return lossval
end

plot(vec(log.(runme(400, 1e-4, SGD))), color=:red,label="SGD");
plot!(vec(log.(runme(400, 1e-4, Momentum))), color=:orange,label="Momentum");
plot!(vec(log.(runme(400, 1e-4, Adam))), color=:green,label="Adam");
plot!(vec(log.(runme(400, 1e-3, RMSProp))), color=:blue,label="RMSProp");
plot!(vec(log.(runme(400, 1e-3, AdaGrad))), color=:cyan,label="AdaGrad")

L1 = 1.0
function runmeL1(epoch, lr, opt, L1)
    mlpmodel  = Chain(dense(2,32),
                      dense(32,32),
                      dense(32,32),
                      linear(32,2))

    params  = xparamsof(mlpmodel)
    optim   = opt(params;lr=lr,L1decay=L1)
    lossval = zeros(epoch,1)

    for i = 1:epoch
        outs = forward(mlpmodel, x)
        COST = crossEntropyCost(softmax(outs;dims=1), l)
        backward()
        update!(optim)
        zerograds!(optim)
        lossval[i] = COST/N
    end
    return lossval
end
plot(vec(log.(runmeL1(400, 1e-4, SGDL1, L1))), color=:red,label="SGDL1")
plot!(vec(log.(runmeL1(400, 1e-4, MomentumL1, L1))), color=:orange,label="MomentumL1")
plot!(vec(log.(runmeL1(400, 1e-4, AdamL1, L1))), color=:green,label="AdamL1")
plot!(vec(log.(runmeL1(400, 1e-3, RMSPropL1, L1))), color=:blue,label="RMSPropL1")
plot!(vec(log.(runmeL1(400, 1e-3, AdaGradL1, L1))), color=:cyan,label="AdaGradL1")

L2 = 1.0
function runmeL2(epoch, lr, opt, L2)
    mlpmodel  = Chain(dense(2,32),
                      dense(32,32),
                      dense(32,32),
                      linear(32,2))

    params  = xparamsof(mlpmodel)
    optim   = opt(params;lr=lr,L2decay=L2)
    lossval = zeros(epoch,1)

    for i = 1:epoch
        outs = forward(mlpmodel, x)
        COST = crossEntropyCost(softmax(outs;dims=1), l)
        backward()
        update!(optim)
        zerograds!(optim)
        lossval[i] = COST/N
    end
    return lossval
end
plot(vec(log.(runmeL2(400, 1e-4, SGDL2, L2))), color=:red,label="SGDL2")
plot!(vec(log.(runmeL2(400, 1e-4, MomentumL2, L2))), color=:orange,label="MomentumL2")
plot!(vec(log.(runmeL2(400, 1e-4, AdamL2, L2))), color=:green,label="AdamL2")
plot!(vec(log.(runmeL2(400, 1e-3, RMSPropL2, L2))), color=:blue,label="RMSPropL2")
plot!(vec(log.(runmeL2(400, 1e-3, AdaGradL2, L2))), color=:cyan,label="AdaGradL2")

L1 = 1.0
L2 = 1.0
function runmeL1L2(epoch, lr, opt, L1, L2)
    mlpmodel  = Chain(dense(2,32),
                      dense(32,32),
                      dense(32,32),
                      linear(32,2))

    params  = xparamsof(mlpmodel)
    optim   = opt(params;lr=lr,L1decay=L1,L2decay=L2)
    lossval = zeros(epoch,1)

    for i = 1:epoch
        outs = forward(mlpmodel, x)
        COST = crossEntropyCost(softmax(outs;dims=1), l)
        backward()
        update!(optim)
        zerograds!(optim)
        lossval[i] = COST/N
    end
    return lossval
end
plot(vec(log.(runmeL1L2(400, 1e-4, SGDL1L2, L1, L2))),color=:red,label="SGDL1L2")
plot!(vec(log.(runmeL1L2(400, 1e-4, MomentumL1L2, L1, L2))),color=:orange,label="MomentumL1L2")
plot!(vec(log.(runmeL1L2(400, 1e-4, AdamL1L2, L1, L2))),color=:green,label="AdamL1L2")
plot!(vec(log.(runmeL1L2(400, 1e-5, RMSPropL1L2, L1, L2))),color=:blue,label="RMSPropL1L2")
plot!(vec(log.(runmeL1L2(400, 1e-5, AdaGradL1L2, L1, L2))),color=:cyan,label="AdaGradL1L2")
