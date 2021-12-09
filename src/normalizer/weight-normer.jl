"""
    wnorm(x, smooth=0.0; by="invprior")

`x` is each category's number of samples. `by` option provides two methods:\n
+ `prior`: let biger be more biger, e.g. [1,4] -> [0.25,1.0]\n
+ `invprior`: let smaller be biger, e.g. [1,4] -> [1.0,0.25]\n
to return each category's weight. `help?> prior` and `help?> invprior` for details.
As `p(X|S) = P(S|X)P(X)/P(S) ∝  P(S|X)/P(S)`, `P(S)` can be considered as weight
when we're dealing with imbalanced category samples with a neural network. `P(S)`
can be estimated by the proportion of each category. i.e. ⤦\n
           (# samples from S)
    P(S) = ────────────────── ∝  (# samples from S)
            (# all samples)
"""
function wnorm(x, smooth=0.0; by="invprior")
    @assert smooth>=0.0 "smooth >= 0 , but got $smooth"
    by=="invprior" && return invprior(x, smooth)
    by=="prior"    && return prior(x, smooth)
end


"""
    prior(x, smooth=0.0) -> z

let biger be more biger! ⤦

    y = (x .+ smooth*maximum(x))
    z = y ./ sum(y)
    z = z ./ maximum(z)
# Example
    julia> x = [1,100,1000];
    julia> prior(x,0.5)'
    0.192618  0.230681  0.576701
"""
function prior(x, smooth=0.0)
    n = maximum(x) * smooth
    y = x .+ n
    s = sum(y)
    z = y ./ sum(y)
    return z ./ maximum(z)
end


"""
    invprior(x, smooth=0.0) -> z

let smaller be biger! ⤦

    y = 1 ./ (x .+ smooth*maximum(x))
    z = y ./ sum(y)
    z = z ./ maximum(z)
# Example
    julia> x = [1,100,1000];
    julia> invprior(x,0.5)'
     1.0  0.835  0.334
"""
function invprior(x, smooth=0.0)
    n = maximum(x) * smooth
    y = 1 ./ (x .+ n)
    s = sum(y)
    z = y ./ sum(y)
    return z ./ maximum(z)
end
