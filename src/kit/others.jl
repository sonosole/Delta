function blocksize(n::Int, u::String)
    if u == "B"  return n / 1 end
    if u == "KB" return n / 1024 end
    if u == "MB" return n / 1048576 end
    if u == "GB" return n / 1073741824 end
    if u == "TB" return n / 1099511627776 end
end


"""
    deleteat!(atuple::Tuple, iters::Union{Tuple,Array,Vector,Int}) -> Tuple
delete elements from a tuple at specified positions `iters` and return it
# Examples
    julia> deleteat!((0,2,4,6,8), (1,3,5))
    (2, 6)
"""
function Base.deleteat!(atuple::Tuple, iters::Union{Tuple,Array,Vector,Int})
    this = []; append!(this, atuple);
    if typeof(iters) <: Int
        rest = deleteat!(this, iters)
    else
        keys = []; append!(keys,iters);
        rest = deleteat!(this, keys);
    end
    return ntuple(i -> rest[i], length(rest))
end


function ShapeAndViews(ndims::Int,                    # total dims
                       keptdims::Union{Tuple,Int},    # must be unique and sorted and positive
                       keptsize::Union{Tuple,Int})    # must be positive

    @assert typeof(keptsize)==typeof(keptdims) "keptsize & keptdims shall be the same type"
    @assert ndims >= maximum(keptdims) "ndims >= maximum(keptdims) shall be met"
    @assert ndims > length(keptdims) "this is no elements for statistical analysis"
    @assert ndims > 0 "ndims > 0, but got ndims=$ndims"

    if typeof(keptdims) <: Int
        if keptdims == 0
            if keptsize!=1
                @warn "keptsize should be 1 here, but got $keptsize"
            end
            shape = ntuple(i -> i==keptdims ? keptsize : 1, ndims);
            views = ntuple(i -> i, ndims);
        else
            shape = ntuple(i -> i==keptdims ? keptsize : 1, ndims);
            views = ntuple(i -> i>=keptdims ? i+1 : i, ndims-1);
        end
    else
        array = [i for i in keptsize]
        shape = ntuple(i -> i in keptdims ? popfirst!(array) : 1, ndims);
        views = deleteat!(ntuple(i -> i, ndims), keptdims)
    end
    return shape, views
end
