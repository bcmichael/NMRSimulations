import LinearAlgebra.axpy!,LinearAlgebra.BLAS.gemm!

abstract type HilbertOperator{T<:AbstractFloat,A<:AbstractArray} end

struct Propagator{T<:AbstractFloat,A<:AbstractArray} <: HilbertOperator{T,A}
    data::A

    function Propagator{T,A}(x) where{T<:AbstractFloat,A<:AbstractArray}
        get_number_type(x)<:Complex || error("Propagator must be complex")
        new{T,A}(x)
    end
end

Propagator(x::A) where {A} = Propagator{get_precision(x),A}(x)

struct Hamiltonian{T<:AbstractFloat,A<:AbstractArray} <: HilbertOperator{T,A}
    data::A
end

Hamiltonian(x::A) where {A} = Hamiltonian{get_precision(x),A}(x)

Hamiltonian(s::SphericalTensor{A}) where {T,A<:AbstractArray{T,2}} =
    SphericalTensor([Hamiltonian(getfield(s, name)) for name in fieldnames(SphericalTensor)]...)

Hamiltonian(s::SphericalTensor{A}, ::Type{Ar}) where {T,A<:AbstractArray{T,2},Ar<:AbstractArray} =
    SphericalTensor([Hamiltonian(Ar(getfield(s, name))) for name in fieldnames(SphericalTensor)]...)

function Hamiltonian(x::Vector{<:SphericalTensor{A}}, ::Type{Ar}) where {A<:AbstractArray,Ar<:AbstractArray}
    fields = Vector{Ar}()
    for name in fieldnames(SphericalTensor)
        combined = cat([getfield(x[n],name) for n = 1:length(x)]..., dims=3)
        push!(fields, Ar(combined))
    end
    SphericalTensor(Hamiltonian.(fields)...)
end

Hamiltonian(x::Vector{<:SphericalTensor{A}}) where {A<:AbstractArray} = Hamiltonian(x,array_wrapper_type(A))

operator_iter(A::HilbertOperator) = size(A.data, 1), size(A.data, 3)

function mul!(C::H, A::H, B::H, transA::Char, transB::Char, alpha::Number, beta::Number) where
    {T<:BLAS.BlasFloat,Ar<:AbstractArray{T,3},T1,H<:HilbertOperator{T1,Ar}}

    gemm_batch!(transA, transB, T(alpha), A.data, B.data, T(beta), C.data)
    C
end

function mul!(C::H, A::H, B::H, transA::Char, transB::Char, alpha::Number, beta::Number) where
    {T<:BLAS.BlasFloat,Ar<:AbstractArray{T,2},T1,H<:HilbertOperator{T1,Ar}}

    gemm!(transA, transB, T(alpha), A.data, B.data, T(beta), C.data)
    C
end

mul!(C::H, A::H, B::H) where {H<:HilbertOperator} = mul!(C, A, B, 'N', 'N', 1, 0)
mul!(C::H, A::H, B::H, transA::Char, transB::Char) where {H<:HilbertOperator} = mul!(C, A, B, transA, transB, 1, 0)
mul!(C::H, A::H, B::H, alpha::Number, beta::Number) where {H<:HilbertOperator} = mul!(C, A, B, 'N', 'N', alpha, beta)

function mul!(C::H, A::SparseMatrixCSC{T,Ti}, B::H, transA::Char, transB::Char, alpha::Number, beta::Number) where
    {Ti,T<:BLAS.BlasFloat,Ar<:AbstractArray{T,3},T1,H<:HilbertOperator{T1,Ar}}

    transB == 'N' || throw(ArgumentError("'T' and 'C' operations for B are not implemented"))
    cscmm!(transA, T(alpha), A, dropdims(B.data, dims=3), T(beta), dropdims(C.data, dims=3))
    C
end

mul!(C::H, A::SparseMatrixCSC, B::H) where {H<:HilbertOperator} = mul!(C, A, B, 'N', 'N', 1, 0)
mul!(C::H, A::SparseMatrixCSC, B::H, transA::Char, transB::Char) where {H<:HilbertOperator} =
    mul!(C, A, B, transA, transB, 1, 0)
mul!(C::H, A::SparseMatrixCSC, B::H, alpha::Number, beta::Number) where {H<:HilbertOperator} =
    mul!(C, A, B, 'N', 'N', alpha, beta)

copy(x::A) where {A<:HilbertOperator} = A(copy(x.data))

copyto!(dest::A,src::A) where {A<:HilbertOperator} = copyto!(dest.data,src.data)

similar(x::A) where {A<:HilbertOperator} = A(similar(x.data))
similar(x::Hamiltonian{T,A}, ::Type{Propagator}) where {T,A} = Propagator(similar(x.data, Complex{T}))

function fill_diag!(A::HilbertOperator, val)
    fill!(A.data, 0)
    x, num = operator_iter(A)
    for j = 1:num
        @simd for n = 1:x
            @inbounds A.data[n, n, j] = val
        end
    end
    return A
end

function real_add(A::Hamiltonian{T,Ar}, B::AbstractMatrix) where {T,Ar}
    out = similar(A.data,T)
    out .= real.(A.data).+real.(B)
    return Hamiltonian(out)
end

function scaledn!(X::HilbertOperator,s::Number)
    X.data ./= s
    X
end

function phase_rotate!(A::HilbertOperator, B::HilbertOperator, phase, parameters)
    At = typeof(A.data)
    rotator = array_wrapper_type(At)(phase_rotator(phase, parameters.xyz))
    A.data .= B.data.*rotator
    return A
end

axpy!(a::Number, X::HilbertOperator, Y::HilbertOperator) = axpy!(a, X.data, Y.data)

array_wrapper_type(::Type{A}) where {A} = A.name.wrapper

"""
    pow!(out, X, p, temp)

Raise 'X' to the power of positive integer 'p' and store the result in 'out'.
'out', 'X', and 'temp' should all be the same type of 'HilbertOperator'. This is
essentially a copy of Base.power_by_squaring that avoids making so many
intermediate allocations. Note that the contents of both 'X' and 'temp' will
also change.
"""
function pow!(out::H, x::H, p::Integer, temp::H) where H<:HilbertOperator
    if p == 1
        copyto!(out, x)
    elseif p == 0
        fill_diag!(out, 1)
    elseif p == 2
        mul!(out, x, x)
    elseif p < 0
        throw(DomainError)
    else
        t = Base.trailing_zeros(p) + 1
        p >>= t
        while (t -= 1) > 0
            mul!(temp, x, x)
            x, temp = temp, x
        end
        copyto!(out, x)
        while p > 0
            t = trailing_zeros(p) + 1
            p >>= t
            while (t -= 1) >= 0
                mul!(temp, x, x)
                x, temp = temp, x
            end
            mul!(temp, out, x)
            put, temp = temp, out
        end
    end
    return out, temp
end
