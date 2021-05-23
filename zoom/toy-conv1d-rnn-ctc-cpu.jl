using Delta

featdims  = 64
timesteps = 124
batchsize = 16

# 1. make some pseudo inputs
x = Variable(randn(featdims,timesteps,batchsize));

# 2. make some pseudo seqlabels
seqlabels = Vector(undef,batchsize)
for i=1:batchsize
    seqlabels[i] = [2, 3, 4, 5]
end

# 3. build conv1d layers
c1 = conv1d(featdims,256,8,stride=3)
c2 = conv1d(256,256,3,stride=2)
c3 = conv1d(256,512,3,stride=1)
c4 = conv1d(512,512,3,stride=1)
c5 = conv1d(512,512,3,stride=1)
chain1 = Chain(c1,c2,c3,c4,c5);

# 4. build a structure with recurrent operators
r1 = indrnn(512,512)
r2 = indrnn(512,512)
f1 = linear(512,1400)
chain2 = Chain(r1,r2,f1);

p1 = paramsof(chain1);
p2 = paramsof(chain2);

optim1 = Momentum(p1);
optim2 = Momentum(p2);

for i=1:20
    # for non-recurrent layers, we computes all timesteps at once
    y1 = forward(chain1,x);
    # for recurrent layers, we computes one timestep at once
    steps = size(y1,2)
    y2 = Vector{Variable}(undef,steps)
    resethidden(chain2)
    for t=1:steps
        y2[t] = forward(chain2, y1[:,t,:])
    end
    # for sequential type loss, we re-arrange time slices
    y3 = unionRNNSteps(y2)
    # computes ctc loss
    LogLikely = CRNN_Batch_CTCLoss_With_Softmax(y3, seqlabels)
    println(LogLikely)
    # backward gradients and update params
    backward()
    update(optim1,p1,clipvalue=10.0)
    update(optim2,p2,clipvalue=10.0)
    # zero gradients
    zerograds(p1)
    zerograds(p2)
end
