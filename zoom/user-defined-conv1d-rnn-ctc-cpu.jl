using Delta
import Delta.paramsof
import Delta.forward
import Delta.predict

# 1. define model struct
mutable struct Model
    blocks::Vector
    function Model(featdims::Int)
        c1 = conv1d(featdims,256,8,stride=3)
        c2 = conv1d(256,256,3,stride=2)
        c3 = conv1d(256,512,3,stride=1)
        c4 = conv1d(512,512,3,stride=1)
        c5 = conv1d(512,512,3,stride=1)
        r1 = indrnn(512,512)
        r2 = indrnn(512,512)
        f1 = linear(512,140)
        new(push!([], c1,c2,c3,c4,c5, Chain(r1,r2,f1)))
    end
end

# 2. define how to extrac model's params
function paramsof(m::Model)
    params = Vector{Variable}(undef,0)
    for b in m.blocks
        paramsof(params, b)
    end
    return params
end

# 3. define model's forward calculation
function forward(ASRModel::Model, input::Variable)
    x1 = forward(ASRModel.blocks[1], input)
    x1 = relu(x1)
    x2 = forward(ASRModel.blocks[2], x1)
    x2 = relu(x2)
    x3 = forward(ASRModel.blocks[3], x2)
    x3 = relu(x3)
    x4 = forward(ASRModel.blocks[4], x3)
    x4 = relu(x4)
    x5 = forward(ASRModel.blocks[5], x4)
    x5 = relu(x5)
    return PackedSeqForward(ASRModel.blocks[6], x5)
end


# 4. experimental constants
featdims  = 64
timesteps = 124
batchsize = 16

# 5. pseudo inputs & labels
x = Variable(randn(featdims,timesteps,batchsize));
seqlabels = Vector(undef,batchsize)
for i=1:batchsize
    seqlabels[i] = [2, 3, 4, 5]
end

# 6. instance an model object
#    extrac its parameters
#    choose an optimizer
asr = Model(featdims);
asrparams = paramsof(asr);
optim = Momentum(asrparams);


# 7. train this model
for e = 1:20
    y = forward(asr, x)
    loglikely = CRNN_Batch_CTCLoss_With_Softmax(y, seqlabels);
    println(loglikely);
    backward()
    update(optim, asrparams, clipvalue=10.0)
    zerograds(asrparams)
end


# 8. define predict calculation
function predict(ASRModel::Model, input::AbstractArray)
    x1 = predict(ASRModel.blocks[1], input)
    x1 = relu(x1)
    x2 = predict(ASRModel.blocks[2], x1)
    x2 = relu(x2)
    x3 = predict(ASRModel.blocks[3], x2)
    x3 = relu(x3)
    x4 = predict(ASRModel.blocks[4], x3)
    x4 = relu(x4)
    x5 = predict(ASRModel.blocks[5], x4)
    x5 = relu(x5)
    return PackedSeqPredict(ASRModel.blocks[6], x5)
end

# 9. predict
y = predict(asr, randn(featdims,timesteps,batchsize));
println(CTCGreedySearch(y[:,:,1]))
