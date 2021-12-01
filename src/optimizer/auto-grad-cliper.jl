# element-wise clipping
function AutoGradCliper(xparams::Vector{XVariable}; lr=1e-3, clipvalue=0.01, eps=1e-3)
    g = clipvalue / lr
    for k = 1:length(xparams)
        c , θ = xparams[k]
        w = ᵛ(θ)
        ∇ = δ(θ)
        a = max(abs.(w), eps) .* g
        i = abs.(∇) .> a
        @. ∇[i] = a[i] * sign(∇[i])
    end
end

# group wise clipping
function AutoGradNormCliper(xparams::Vector{XVariable}; lr=1e-3, clipvalue=0.01, eps=1e-3, debug=false)
    g = clipvalue / lr
    n = length(xparams)
    for k = 1:n
        c , θ = xparams[k]
        w = ᵛ(θ)
        ∇ = δ(θ)
        NormW = sum(abs.(w))      # l₁ norm of param
        if NormW/length(w) > eps  # 平均幅度不应该太低
            NormW = g * NormW     # 与放大系数合并
            Norm∇ = sum(abs.(∇))  # l₁ norm of gradient
            if Norm∇ > NormW
                ∇ .*= NormW/Norm∇
                debug ? (@info "AutoGradNormCliper[$(k)th] $(lr*Norm∇/(NormW/g)) > $clipvalue, $c gradient too steep") : nothing
            end
        end
    end
end
