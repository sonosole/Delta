export @index
export @basic
export @extend

macro index(xstruct, blocks)
    quote
        NerualStruct = $(esc(xstruct))
        Base.length(s::NerualStruct)     = length(s.$blocks)
        Base.lastindex(s::NerualStruct)  = length(s.$blocks)
        Base.firstindex(s::NerualStruct) = 1
        Base.getindex(s::NerualStruct, k...)     =  s.$blocks[k...]
        Base.setindex!(s::NerualStruct, v, k...) = (s.$blocks[k...] = v)
        Base.iterate(s::NerualStruct, i=firstindex(s)) = i>length(s) ? nothing : (s[i], i+1)
    end
end

macro basic(xstruct, blocks)
    quote
        NerualStruct = $(esc(xstruct))

        function Base.show(io::IO, s::NerualStruct)
            print(io, "$NerualStruct(\n")
            join(io, s.$blocks, "\n")
            print(io, "\n)")
        end

        function Delta.paramsof(m::NerualStruct)
            params = Vector{Variable}(undef,0)
            for i = 1:length(m)
                append!(params, paramsof(m[i]))
            end
            return params
        end

        function Delta.xparamsof(m::NerualStruct)
            xparams = Vector{XVariable}(undef,0)
            for i = 1:length(m)
                append!(xparams, xparamsof(m[i]))
            end
            return xparams
        end

        function Delta.nparamsof(m::NerualStruct)
            nparams = 0
            for i = 1:length(m)
                nparams += nparamsof(m[i])
            end
            return nparams
        end

        function Delta.bytesof(model::NerualStruct, unit::String="MB")
            n = nparamsof(model)
            u = uppercase(unit)
            if u == "KB" return n * sizeof(eltype(model[1].w)) / 1024 end
            if u == "MB" return n * sizeof(eltype(model[1].w)) / 1048576 end
            if u == "GB" return n * sizeof(eltype(model[1].w)) / 1073741824 end
            if u == "TB" return n * sizeof(eltype(model[1].w)) / 1099511627776 end
        end
    end
end

macro extend(xstruct, blocks)
    quote
        NerualStruct = $(esc(xstruct))
        Base.length(s::NerualStruct)     = length(s.$blocks)
        Base.lastindex(s::NerualStruct)  = length(s.$blocks)
        Base.firstindex(s::NerualStruct) = 1
        Base.getindex(s::NerualStruct, k...)     =  s.$blocks[k...]
        Base.setindex!(s::NerualStruct, v, k...) = (s.$blocks[k...] = v)
        Base.iterate(s::NerualStruct, i=firstindex(s)) = i>length(s) ? nothing : (s[i], i+1)

        function Base.show(io::IO, s::NerualStruct)
            print(io, "$NerualStruct(\n")
            join(io, s.$blocks, "\n")
            print(io, "\n)")
        end

        function Delta.paramsof(m::NerualStruct)
            params = Vector{Variable}(undef,0)
            for i = 1:length(m)
                append!(params, paramsof(m[i]))
            end
            return params
        end

        function Delta.xparamsof(m::NerualStruct)
            xparams = Vector{XVariable}(undef,0)
            for i = 1:length(m)
                append!(xparams, xparamsof(m[i]))
            end
            return xparams
        end

        function Delta.nparamsof(m::NerualStruct)
            nparams = 0
            for i = 1:length(m)
                nparams += nparamsof(m[i])
            end
            return nparams
        end

        function Delta.bytesof(model::NerualStruct, unit::String="MB")
            n = nparamsof(model)
            u = uppercase(unit)
            if u == "KB" return n * sizeof(eltype(model[1].w)) / 1024 end
            if u == "MB" return n * sizeof(eltype(model[1].w)) / 1048576 end
            if u == "GB" return n * sizeof(eltype(model[1].w)) / 1073741824 end
            if u == "TB" return n * sizeof(eltype(model[1].w)) / 1099511627776 end
        end
    end
end
