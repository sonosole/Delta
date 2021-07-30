using UnicodePlots

# a simple example for DNN_CTCLoss_With_Softmax usage
# by the way, it's for a single input case
S = 5
T = 100
E = 1000
x = 0.5*rand(S,T);
P = Variable(x,keepsgrad=true);
L = zeros(1, E)
for i = 1:E
    L[i] = DNN_CTC_With_Softmax(P, [2 3 4 5 2 3 4 5])
    backward()
    update!(P, 1E-3)
end

lineplot(vec(L))


# a simple example for DNN_Batch_CTCLoss_With_Softmax usage
# by the way, it's for a batch of inputs case
E = 100
S = 5
T1 = 10
T2 = 20
x1 = 0.5*rand(S,T1);  # first input
x2 = 0.3*rand(S,T2);  # second input
seq1 = [2 3 4];       # first input's label
seq2 = [3 2];         # second input's label

x   = [x1 x2];        # a batch of inputs
seq = [seq1 seq2];    # a batch of labels

P = Variable(x, keepsgrad=true);
L = zeros(1, E)
for i = 1:E
    L[i] = DNN_Batch_CTC_With_Softmax(P, seq, [T1 T2], [length(seq1) length(seq2)])
    backward()
    update!(P, 1E-3)
end

lineplot(vec(L))
