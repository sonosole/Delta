export CRNN_BoundedCTC_With_Softmax


function CRNN_BoundedCTC_With_Softmax(x::Variable{Array{T}},
                                     seqlabels::Vector;
                                     blank::Int=1,
                                     bound::Int=2,
                                     reduction::String="seqlen",
                                     weight::Float64=1.0) where T
    featdims, timesteps, batchsize = size(x)
    loglikely = zeros(T, batchsize)
    p = softmax(ᵛ(x); dims=1)
    r = zero(ᵛ(x))

    Threads.@threads for b = 1:batchsize
        r[:,:,b], loglikely[b] = BoundedCTC(p[:,:,b], seqlabels[b], blank=blank, bound=bound)
    end

    Δ = p - r
    reduce3d(Δ, loglikely, seqlabels, reduction)

    if x.backprop
        function CRNN_BoundedCTC_With_Softmax_Backward()
            if need2computeδ!(x)
                if weight==1.0
                    δ(x) .+= Δ
                else
                    δ(x) .+= Δ .* weight
                end
            end
        end
        push!(graph.backward, CRNN_BoundedCTC_With_Softmax_Backward)
    end
    return sum(loglikely)
end
