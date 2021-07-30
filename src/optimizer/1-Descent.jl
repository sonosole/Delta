mutable struct Descent <: Optimizer
    lr::AbstractFloat
    decay::AbstractFloat
    name::String
    function Descent(;lr=1e-4,decay=1.0)
        new(lr, decay, "Descent")
    end
end

# pretty printing
function Base.show(io::IO, D::Descent)
    print("Descent(lr=$(D.lr), decay=$(D.decay))")
end


function update!(d::Descent, params::Vector{Variable})
    lrate = d.lr
    d.lr *= d.decay
    update!(params, lrate)
end
