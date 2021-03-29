# The precision of forward-differiation is accurate at Float64 but not at Float32/16.
# Anyway the backward-differentiation is accurate at Float64/32/16.
# Just check at Float64/32/16 while keeping the same random seed.
# For the below example the gradients for CPU/GPU on my device are
# Float64 is 100.724249419
# Float32 is 100.72425
# Float16 is 100.75
# ochannels = 1

@testset "check gradient for conv1d block" begin
    clear()
    # using Random
    # Random.seed!(1234);
    TYPE = Array{Float64};

    # [0] prepare model
    ichannels = 1;
    ochannels = 2;
    c1 = conv1d(ichannels,4,3; type=TYPE);
    c2 = conv1d(4,ochannels,2; type=TYPE);

    # [1] prepare input data and its label
    timeSteps = 128;
    batchsize = 32;
    x = Variable(rand(ichannels,timeSteps,batchsize); type=TYPE);
    l = Variable(rand(ochannels,      125,batchsize); type=TYPE);

    # [2] forward and backward propagation
    o1 = forward(c1,  x);
    o2 = forward(c2, o1);
    COST1 = mseCost(o2, l);
    backward();

    # [3] with a samll change of a weight
    GRAD = c1.w.delta[1];
    DELTA = 1e-6;
    c1.w.value[1] += DELTA;

    # [4] forward and backward propagation
    o1 = forward(c1,  x);
    o2 = forward(c2, o1);
    COST2 = mseCost(o2, l);
    backward();

    # [5] check if the auto-grad is true or not
    dLdW = (COST2 - COST1)/DELTA;   # numerical gradient
    err  = abs((dLdW-GRAD)/(GRAD+eps(Float64)))*100;  # relative error in %
    err  = err < 1e-1 ? 0.0 : err;
    @test err < 0.1
end
