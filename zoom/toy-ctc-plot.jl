using Gadfly
    
S = 10
T = 30
x = 0.01*rand(S,T)
P = softmax(x, dims=1)

tic = time()
r, loglikely = CTC(P,[3 4 5])
toc = time()
println("loglikely = ",loglikely/T," (timeSteps averaged)")
println("ctc_time  = ",(toc-tic)*1000," ms")

Gadfly.plot(
layer(y=r[1,:],Geom.line,Theme(default_color=colorant"red")),
layer(y=r[2,:],Geom.line,Theme(default_color=colorant"yellow")),
layer(y=r[3,:],Geom.line,Theme(default_color=colorant"blue")),
layer(y=r[4,:],Geom.line,Theme(default_color=colorant"green")),
layer(y=r[5,:],Geom.line,Theme(default_color=colorant"orange"))
)