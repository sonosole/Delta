mutable struct SGD <: Optimizer
    lr::AbstractFloat
    lrdecay::AbstractFloat
    name::String
    function SGD(;lr=1e-4, lrdecay=1.0)
        new(lr, lrdecay, "SGD")
    end
end

# pretty printing
function Base.show(io::IO, S::SGD)
    print("SGD(lr=$(S.lr), lrdecay=$(S.lrdecay))")
end


function update!(s::SGD, params::Vector{Variable})
    lrate = s.lr
    s.lr *= s.decay
    update!(params, lrate)
end
