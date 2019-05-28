using CUDAnative
using CuArrays
using CUDAdrv

CuArrays.allowscalar(false)

function fill_diag!(A::HilbertOperator{T,Ar}, val) where {T<:AbstractFloat,Ar<:CuArray}
    x, y = operator_iter(A)
    if x <= 128
        @cuda blocks=y threads=x kernel_fill_diag!(A.data, T(val), 1)
    else
        repeats=Int(x/128)
        @cuda blocks=y threads=128 kernel_fill_diag!(A.data, T(val), repeats)
    end
    return A
end

function kernel_fill_diag!(A, val::T, repeats) where {T}
    i = threadIdx().x
    j = blockIdx().x
    for k = 1:repeats
        for n = 1:blockDim().x*repeats
            A[i, n, j] = T(0)
        end
        A[i, i, j] = val
        i += blockDim().x
    end

    return nothing
end

struct CuSparseMatrixCSC{Tv,Ti<:Integer}
    m::Int                  # Number of rows
    n::Int                  # Number of columns
    colptr::CuArray{Ti,1}      # Column i is in colptr[i]:(colptr[i+1]-1)
    rowval::CuArray{Ti,1}      # Row indices of stored values
    nzval::CuArray{Tv,1}     # Stored values, typically nonzeros
    occupied_cols::CuArray{Ti,1}  #list of columns with non zero stored values

    function CuSparseMatrixCSC(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
        occupied = Vector{Ti}()
        for n = 1:A.n
            A.colptr[n] == A.colptr[n+1] || push!(occupied, n)
        end
        new{Tv,Ti}(A.m, A.n, CuArray(A.colptr), CuArray(A.rowval), CuArray(A.nzval), CuArray(occupied))
    end
end

function propagate!(spec::CuArray, ρ0, detector, prop_generator)
    for (U, position) in prop_generator
        detect!(spec, U, ρ0, detector, detector.occupied_cols, position, prop_generator.temps[1])
    end

    return spec, Uloop
end

function detect!(spec::CuArray, Uloop::HilbertOperator{T1,<:CuArray{T}}, ρ0::CuSparseMatrixCSC,
    detector::CuSparseMatrixCSC, unique_cols, loop,temp) where {T,T1}

    x,num = operator_iter(Uloop)
    @cuda blocks=(length(unique_cols), num) threads=x shmem=sizeof(T)*x kernel_detect!(Uloop.data, ρ0.colptr, ρ0.rowval,
        ρ0.nzval, detector.occupied_cols, detector.colptr, detector.rowval, detector.nzval, spec, loop)
    return spec
end

function kernel_detect!(Uloop, ρ0colptr, ρ0rowval, ρ0nzval::CuDeviceArray{T,1}, rows, dcolptr, drowval, dnzval, results,
    loop) where {T}

    temporary = @cuDynamicSharedMem(T, blockDim().x)
    row = rows[blockIdx().x]
    crystallite = blockIdx().y
    col = threadIdx().x

    accumulated = T(0)
    temporary[col] = Uloop[row, col, crystallite]
    sync_threads()

    row_val = T(0)
    for k = ρ0colptr[col]:(ρ0colptr[col+1]-1)
        row_val += temporary[ρ0rowval[k]]*ρ0nzval[k]
    end

    # calculate elements in Uloop*ρ0*Uloop'.*transpose(d)
    # each thread calculates the results for a single element in the row of Uloop*ρ0
    for m = dcolptr[row]:(dcolptr[row+1]-1)
        accumulated += row_val*CUDAnative.conj(Uloop[drowval[m], col, crystallite])*dnzval[m]
    end

    accumulated = sum_block(accumulated, temporary)

    # write to global memory
    if col == 1
        results[blockIdx().x, blockIdx().y, loop] += accumulated
    end

    return nothing
end

@inline function sum_block(val::T, temporary)::T where {T}
    # reduce across warps
    wid  = div(threadIdx().x-UInt32(1), CUDAnative.warpsize()) + UInt32(1)
    lane = rem(threadIdx().x-UInt32(1), CUDAnative.warpsize()) + UInt32(1)
    val=sum_warp(val)

    if lane == 1
        temporary[wid] = val
    end
    sync_threads()

    # reduce across full block
    if wid == 1
        val = (threadIdx().x <= fld1(blockDim().x,32)) ? temporary[threadIdx().x] : T(0)
        val = sum_warp(val)
    end
    return val
end

@inline function sum_warp(val::T)::T where {T}
    val += CUDAnative.shfl_down(val, UInt32(16))
    val += CUDAnative.shfl_down(val, UInt32(8))
    val += CUDAnative.shfl_down(val, UInt32(4))
    val += CUDAnative.shfl_down(val, UInt32(2))
    val += CUDAnative.shfl_down(val, UInt32(1))
    return val
end

function eig_max_bound(A::CuArray{T}) where {T}
    x = size(A, 1)
    z = size(A, 3)
    results = CuArray{T}(undef, z)
    @cuda blocks=z threads=x shmem=sizeof(T)*z kernel_eig_max_bound(A, results)
    return maximum(Array(results))
end

function kernel_eig_max_bound(A::CuDeviceArray{T}, results) where {T}
    shared = @cuDynamicSharedMem(T, fld1(blockDim().x, 32))

    val = T(0)
    for n = 1:blockDim().x
        val += CUDAnative.abs(A[threadIdx().x, n, blockIdx().x])
    end

    val = max_block(val, shared)

    if threadIdx().x == 1
        results[blockIdx().x] = val
    end

    return nothing
end

@inline function max_block(val::T, shared)::T where {T}
    # compare across warps
    wid  = div(threadIdx().x-UInt32(1), CUDAnative.warpsize()) + UInt32(1)
    lane = rem(threadIdx().x-UInt32(1), CUDAnative.warpsize()) + UInt32(1)
    val = max_warp(val)

    if lane == 1
        shared[wid] = val
    end
    sync_threads()

    # compare across full block
    if wid == 1
        val = (threadIdx().x <= fld1(blockDim().x,32)) ? shared[threadIdx().x] : T(0)
        val = max_warp(val)
    end
    return val
end

@inline function max_warp(val::T) where {T}
    val = CUDAnative.max(val, CUDAnative.shfl_down(val, UInt32(16)))
    val = CUDAnative.max(val, CUDAnative.shfl_down(val, UInt32(8)))
    val = CUDAnative.max(val, CUDAnative.shfl_down(val, UInt32(4)))
    val = CUDAnative.max(val, CUDAnative.shfl_down(val, UInt32(2)))
    val = CUDAnative.max(val, CUDAnative.shfl_down(val, UInt32(1)))
    return val
end

function threshold(A::CuArray{T}, thresh::T) where {T}
    x = size(A, 1)
    z = size(A, 3)
    results = CuArray{Bool}(undef, z)
    @cuda blocks=z threads=x shmem=sizeof(Bool) kernel_threshold(A, thresh, results)
    return all(Array(results))
end

function kernel_threshold(A::CuDeviceArray{T}, thresh::T, results) where {T}
    shared = @cuDynamicSharedMem(Bool,1)

    if threadIdx().x == 1
        shared[1] = true
    end
    sync_threads()
    for n = 1:blockDim().x
        if ! shared[1]
            break
        end
        if ! (CUDAnative.abs(A[threadIdx().x, n, blockIdx().x]) <= thresh)
            shared[1] = false
        end
    end

    sync_threads()
    if threadIdx().x == 1
        results[blockIdx().x] = shared[1]
    end

    return nothing
end

# Fix performance issue in CUDAdrv 2.0.0
function Base.unsafe_convert(::Type{PtrOrCuPtr{T}}, val) where {T}
    ptr = if applicable(Base.unsafe_convert, Ptr{T}, val)
        Base.unsafe_convert(Ptr{T}, val)
    elseif applicable(Base.unsafe_convert, CuPtr{T}, val)
        Base.unsafe_convert(CuPtr{T}, val)
    else
        throw(ArgumentError("cannot convert to either a CPU or GPU pointer"))
    end
    return Base.bitcast(PtrOrCuPtr{T}, ptr)
end

gemm_batch!(transA::Char, transB::Char, alpha::T, A::CuArray{T,3}, B::CuArray{T,3}, beta::T,
    C::CuArray{T,3}) where {T<:BlasFloat} = CuArrays.CUBLAS.gemm_strided_batched!(transA, transB, alpha, A, B, beta, C)

# This isn't an actual BLAS operation, because the arrays are different types, but we use this in expm_cheby
# The fallback uses getindex, which is horribly slow for CuArrays so define a replacement with a CUDA kernel
function axpy!(a::Number, X::CuArray{T}, Y::CuArray{Complex{T}}) where {T}
    if length(X) != length(Y)
        throw(DimensionMismatch("X has length $(length(X)), Y has length $(length(Y))"))
    end
    if a isa Real
        a2 = T(a)
    else
        a2 = Complex{T}(a)
    end

    x = size(X, 1)
    y = size(X, 2)
    z = size(X, 3)
    @cuda blocks=(y,z) threads=x kernel_generic_axpy(a2, X, Y)
    Y
end

function kernel_generic_axpy(a, X, Y)
    Y[threadIdx().x, blockIdx().x, blockIdx().y] += a*X[threadIdx().x, blockIdx().x, blockIdx().y]
    return nothing
end
