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

# 3. build PlainConv1d layers
c1 = PlainConv1d(featdims,256,8,stride=3)
c2 = PlainConv1d(256,256,3,stride=2)
c3 = PlainConv1d(256,512,3,stride=1)
c4 = PlainConv1d(512,512,3,stride=1)
c5 = PlainConv1d(512,512,3,stride=1)
chain1 = Chain(c1,c2,c3,c4,c5);

# 4. build a structure with recurrent operators
r1 = indrnn(512,512)
r2 = indrnn(512,512)
f1 = linear(512,1400)
chain2 = Chain(r1,r2,f1);

p1 = xparamsof(chain1);
p2 = xparamsof(chain2);

optim1 = Momentum(p1);
optim2 = Momentum(p2);

for i=1:20
    # for non-recurrent layers, we computes all timesteps at once
    y1 = forward(chain1, x);
    # sequential forward propagation
    y2 = PackedSeqForward(chain2, y1);
    # computes ctc loss
    logLikely = CRNN_Batch_CTC_With_Softmax(y2, seqlabels);
    println(logLikely);
    # backward gradients and update! params
    backward();
    update!(optim1,p1,clipvalue=10.0);
    update!(optim2,p2,clipvalue=10.0);
    # zero gradients
    zerograds!(p1);
    zerograds!(p2);
end
