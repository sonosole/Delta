export timeslotmat
export adjustLossWeights
export reduce3d

"""
    timeslotmat(matrix::AbstractMatrix, timestamp::AbstractVector; dim=2, slotvalue=1.0)

mark `matrix` with `timestamp`, all elements of `timestamp` is ∈ (0,1), standing for time ratio

# Example
    julia> timeslotmat(reshape(1:24,2,12), [0.2, 0.8], dim=2, slotvalue=0)
    2×12 Array{Int64,2}:
    1  3  0  7   9  11  13  15  17  0  21  23
    2  4  0  8  10  12  14  16  18  0  22  24
          ↑                         ↑
          20% position              80% position
"""
function timeslotmat(matrix::AbstractMatrix{S1}, timestamp::AbstractVector{S2}; dim=2, slotvalue=1.0) where {S1,S2}
    x = copy(matrix)
    T = size(x, dim)               # time dimention
    I = ceil.(Int, T .* timestamp) # map time in (0,1) to integer [1,T]
    if dim==2
        for t in I
            x[:,t] .= slotvalue
        end
    elseif dim==1
        for t in I
            x[t,:] .= slotvalue
        end
    else
        @error "dim is 1 or 2, but got $dim"
    end
    return x
end


"""
    adjustLossWeights(x...) -> w

`x` is the loss values of multiple losses at training step i, so the weights for
each loss function at training step (i+1) is `w` , `w` is from:

    yᵢ = 1 / xᵢ
    wᵢ = yᵢ / ∑ yᵢ
"""
function adjustLossWeights(x...)
    n = length(x)
    w = zeros(n)
    for i = 1:n
        w[i] = 1 / x[i]
    end
    return w ./ sum(w)
end


function reduce3d(Δ::Array{T}, loglikely::Vector{T}, seqlabels::Vector, reduction::String) where T
    featdims, timesteps, batchsize = size(Δ)
    # 标签长度归一化 ⤦
    if isequal(reduction, "seqlen")
        Threads.@threads for b = 1:batchsize
            seqlen   = length(seqlabels[b]) * batchsize
            seqlen⁻¹ = 1 / ifelse(seqlen≠0, seqlen, batchsize)
            Δ[:,:,b]    .*= seqlen⁻¹
            loglikely[b] *= seqlen⁻¹
        end
    # 时间长度归一化 ⤦
    elseif isequal(reduction, "timesteps")
        timesteps⁻¹ = 1 / (timesteps * batchsize)
        Δ         .*= timesteps⁻¹
        loglikely .*= timesteps⁻¹
    # 网格归一化 ⤦
    elseif isequal(reduction, "trellis")
        Threads.@threads for b = 1:batchsize
            volume   = length(seqlabels[b]) * timesteps * batchsize
            volume⁻¹ = 1 / ifelse(volume≠0, volume, timesteps * batchsize)
            Δ[:,:,b]    .*= volume⁻¹
            loglikely[b] *= volume⁻¹
        end
    # 只是 batchsize 归一化 ⤦
    elseif isequal(reduction, "normal")
        batchsize⁻¹ = 1 / batchsize
        Δ         .*= batchsize⁻¹
        loglikely .*= batchsize⁻¹
    # 无归一化 ⤦
    elseif isequal(reduction, "nil")
        return nothing
    else
        @warn "reduction is one of seqlen/timesteps/trellis/normal/nil, but got $reduction"
    end
    return nothing
end
