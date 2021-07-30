using Delta
import Delta.paramsof
import Delta.forward
import Delta.predict
import Base.length
import Base.getindex


# 1. define model struct and its length and indexing method
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
        new([c1,c2,c3,c4,c5, Chain(r1,r2,f1)])
    end
end


Base.length(m::Model) = length(m.blocks)
Base.getindex(m::Model, k...) = m.blocks[k...]


# 2. define how to extrac model's params
function paramsof(m::Model)
    params = Vector{Variable}(undef,0)
    for i = 1:length(m)
        append!(params, paramsof(m[i]))
    end
    return params
end

# 3. define model's forward calculation
function forward(m::Model, input::Variable)
    x1 = forward(m[1], input)
    x1 = relu(x1)
    x2 = forward(m[2], x1)
    x2 = relu(x2)
    x3 = forward(m[3], x2)
    x3 = relu(x3)
    x4 = forward(m[4], x3)
    x4 = relu(x4)
    x5 = forward(m[5], x4)
    x5 = relu(x5)
    return PackedSeqForward(m[6], x5)
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

# 6. instantiate an model object
#    extrac its parameters
#    choose an optimizer
asr = Model(featdims);
asrparams = paramsof(asr);
optim = Momentum(asrparams);
epoch = 37

# 7. train this model
for e = 1:epoch
    y = forward(asr, x)
    loglikely = CRNN_Batch_CTC_With_Softmax(y, seqlabels);
    println("iter $e loss = $loglikely");
    backward()
    update!(optim, asrparams, clipvalue=20.0)
    zerograds!(asrparams)
end

# 8. define predict calculation
function predict(m::Model, input::AbstractArray)
    x1 = predict(m[1], input)
    x1 = relu(x1)
    x2 = predict(m[2], x1)
    x2 = relu(x2)
    x3 = predict(m[3], x2)
    x3 = relu(x3)
    x4 = predict(m[4], x3)
    x4 = relu(x4)
    x5 = predict(m[5], x4)
    x5 = relu(x5)
    return PackedSeqPredict(m[6], x5)
end

# 9. predict
y = predict(asr, randn(featdims,timesteps,batchsize));
r = CTCGreedySearch(softmax(y[:,:,10],dims=1));
println("decoding result: ", r);



# for more humanlity iteration style
Base.lastindex(m::Model)  = length(m)
Base.firstindex(m::Model) = 1
Base.iterate(m::Model, i=firstindex(m)) = i>length(m) ? nothing : (m[i], i+1)

# so we could iterate like `for x in X`
function Delta.nparamsof(model::Model)
    nparams = 0
    for m in model
        nparams += nparamsof(m)
    end
    return nparams
end

# of course we could also calculate how many bytes it uses
function bytesof(model::Model, unit::String="MB")
    n = nparamsof(model)
    u = lowercase(unit)
    if u == "kb" return n * sizeof(eltype(model[1].w)) / 1024 end
    if u == "mb" return n * sizeof(eltype(model[1].w)) / 1048576 end
    if u == "gb" return n * sizeof(eltype(model[1].w)) / 1073741824 end
    if u == "tb" return n * sizeof(eltype(model[1].w)) / 1099511627776 end
end
