export uniform, eye
export randdiagonal
export randndiagonal


"""
    uniform(dtype::Type, shape::Tuple; from=0.0, to=1.0)

returns a N×M matrix with random values

# Exmaple
    julia> uniform(Float16, (2,4), from=0.2, to=1)
    2×4 Array{Float16,2}:
    0.2695  0.5615  0.4766  0.364
    0.8506  0.3726  0.5605  0.8994
"""
function uniform(dtype::Type, shape::Tuple; from=0.0, to=1.0)
    INIT = dtype(from)
    LAST = dtype(to)
    if from==dtype(0.0) && to==dtype(1.0)
        return rand(dtype, shape)
    else
        return rand(dtype, shape) .* (LAST - INIT) .+ INIT
    end
end

uniform(shape::Tuple; from=0.0, to=1.0) = uniform(Float32, shape; from=from, to=to)


"""
    randdiagonal(dtype::Type, N::Int; from=0.0, to=1.0)

returns a diagonal N×N matrix with random values for the diagonal elements

# Exmaple
    julia> randdiagonal(Float16, 4, from=2.0, to=-1.0)
    4×4 Array{Float16,2}:
     1.739   0.0      0.0     0.0
     0.0    -0.8555   0.0     0.0
     0.0     0.0     -0.9707  0.0
     0.0     0.0      0.0     1.224
"""
function randdiagonal(dtype::Type, N::Int; from=0.0, to=1.0)
    INIT = dtype(from)
    LAST = dtype(to)
    a = zeros(dtype, N, N)
    for i = 1:N
        a[i,i] = rand(dtype) * (LAST - INIT) + INIT
    end
    return a
end

randdiagonal(N::Int; from=0.0, to=1.0) = randdiagonal(Float32, N; from=from, to=to)


"""
    randndiagonal(dtype::Type, N::Int; mean=0.0, std=1.0)

returns a diagonal N×N matrix with random values for the diagonal elements

# Exmaple
    julia> randndiagonal(Float16, 4, mean=0.5, std=7)
    4×4 Array{Float16,2}:
     -3.887  0.0   0.0     0.0
      0.0    0.68  0.0     0.0
      0.0    0.0   2.793   0.0
      0.0    0.0   0.0    -3.695
"""
function randndiagonal(dtype::Type, N::Int; mean=0.0, std=1.0)
    μ = dtype(mean)
    σ = dtype(std)
    a = zeros(dtype, N, N)
    for i = 1:N
        a[i,i] = (randn(dtype) - μ) * σ
    end
    return a
end

randndiagonal(N::Int; mean=0.0, std=1.0) = randndiagonal(Float32, N; mean=mean, std=std)


"""
    eye(dtype::Type, N::Int)

give a N×N identity matrix

# Exmaple
    julia> eye(Float64, 3)
    3×3 Array{Float64,2}:
     1.0  0.0  0.0
     0.0  1.0  0.0
     0.0  0.0  1.0
"""
function eye(dtype::Type, N::Int)
    𝟙 = dtype(1.0f0)
    a = zeros(dtype, N, N)
    for i = 1:N
        a[i,i] = 𝟙
    end
    return a
end

eye(N::Int) = eye(Float32, N)
