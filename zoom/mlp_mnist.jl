using Random
using MLDatasets


# download mnist train and test dataset
Xtrain, Ytrain = MNIST.traindata()
Xtest,  Ytest  = MNIST.testdata()

# construct a simple mlp network
topology = [784, 256, 256, 256, 128, 10]
operator = [relu, relu, relu, relu, relu]
mlpmodel = MLP(topology, operator)

# or equivalently construct a mlp like this:
mlpmodel = Chain(
    dense(784,256,relu),
    dense(256,256,relu),
    dense(256,256,relu),
    dense(256,128,relu),
    dense(128, 10,relu)
)

# training function
function train(Xtrain, Ytrain, Xtest, Ytest, mlpmodel)
    parameter = paramsof(mlpmodel)
    optimizer = Adam(parameter;lr=1e-3,b1=0.9,b2=0.996)
    epoch     = 2
    batchsize = 32
    filesnums = size(Xtrain,3)
    batchnums = div(filesnums,batchsize)
    lossvalue = zeros(batchnums,1)
    idx   = collect(1:filesnums)
    loss  = zeros(epoch,1)
    accc  = zeros(epoch,1)
    for e = 1:epoch
        shuffle!(idx)
        for b = 1:batchnums
            shift = batchsize*(b-1)
            input = zeros(784,0)
            label = zeros(10,0)
            for i = 1:batchsize
                feats = convert.(Float64, reshape(Xtrain[:,:,idx[i+shift]],784,1))
                numbs = tolabel(Ytrain[idx[i+shift]])
                input = hcat(input,feats)
                label = hcat(label,numbs)
            end
            outs = forward(mlpmodel, Variable(input))
            prob = softmax(outs; dims=1)
            Cost = crossEntropyCost(prob, Variable(label))
            backward()
            if b%8==0
                update(optimizer,parameter;clipvalue=50)
                zerograds(parameter)
            end
            lossvalue[b] = Cost / batchsize
        end
        accc[e] = test(Xtest, Ytest, mlpmodel)
        loss[e] = sum(lossvalue) / batchnums
        println("epoch[$(e)]","loss: ",loss[e] , " accuracy: ",accc[e])
    end
    return loss,accc
end


# testing function
function test(Xtest, Ytest, mlpmodel)
    acc = 0.0
    num = size(Xtest,3)
    for i = 1:num
        feats = convert.(Float64, reshape(Xtest[:,:,i],784,1))
        label = convert(Int,Ytest[i]) + 1
        result = argmax(softmax(predict(mlpmodel, feats),dims=1))[1]
        acc += (label==result) ? 1.0 : 0.0
    end
    return acc/num
end


function tolabel(x)
    y = zeros(10,1)
    y[x+1] = 1.0
    return y
end


loss, acc = train(Xtrain, Ytrain, Xtest, Ytest, mlpmodel)
