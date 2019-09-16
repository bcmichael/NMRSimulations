"""
    propagate!(spec, ρ0, detector, prop_generator)

Iterate through 'prop_generator' and calculate the signal using the resulting
propagators, the 'detector' operator, and the initial density operator 'ρ0'.
"""
function propagate!(spec, ρ0, detector, prop_generator)
    unique_cols = occupied_columns(detector)
    for (U, position) in prop_generator
        detect!(spec, U, ρ0, detector, unique_cols, position, prop_generator.temps[1])
    end
    return spec
end

function occupied_columns(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    occupied = Vector{Ti}()
    for n = 1:A.n
        A.colptr[n] == A.colptr[n+1] || push!(occupied, n)
    end
    return occupied
end

function detect!(spec, Uloop, ρ0::SparseMatrixCSC, detector::SparseMatrixCSC, unique_cols, j, temp)
    x, num = operator_iter(Uloop)
    A_mul_B_rows!(temp, Uloop, ρ0, unique_cols)
    for k in unique_cols
        for m = detector.colptr[k]:(detector.colptr[k+1]-1)
            for n = 1:num
                spec[n,j] += A_mul_Bc_single_element(temp.data, Uloop.data, k, detector.rowval[m], n)*detector.nzval[m]
            end
        end
    end
    return spec
end

"""
    A_mul_Bc_single_element(A, B, row, col, n)

Calculate the single element at coordinate ['row','col'] of the result of
A[:,:,n]*B[:,:,n]'.
"""
function A_mul_Bc_single_element(A::At, B::At, row, col, n) where {T,At<:Union{Array{T,2},Array{T,3}}}
    size(A) == size(B) || throw(DimensionMismatch)
    x = size(A, 1)
    y = size(A, 2)
    z = size(A, 3)
    x == y || throw(DimensionMismatch)
    row <= x || throw(DomainError)
    col <= x || throw(DomainError)
    n <= z || throw(DomainError)

    out = zero(T)
    @inbounds for m = 1:x
        out += A[row, m, n]*conj(B[col, m, n])
    end
    return out
end

"""
    A_mul_B_rows!(C, A, B, rows)

Mutate the specified 'rows' in 'C' to match the corrseponding locations in the
product of 'A' and 'B' without calculating the entire matrix product.
"""
function A_mul_B_rows!(C::HilbertOperator, A::HilbertOperator, B::SparseMatrixCSC, rows)
    x, num = operator_iter(A)
    operator_iter(C) == (x, num) || throw(DimensionMismatch)
    (x,x) == (B.n, B.m) || throw(DimensionMismatch)

    Cd = C.data # using these aliases instead of getting them in the loop is faster for some reason
    Ad = A.data
    rowval = B.rowval
    nzval = B.nzval
    @inbounds for n = 1:num, multivec_row in rows, col = 1:B.n
        Cd[multivec_row, col, n] = 0
        for k = B.colptr[col]:(B.colptr[col+1]-1)
            Cd[multivec_row, col, n] += Ad[multivec_row, rowval[k], n] * nzval[k]
        end
    end
    C
end

"""
    γ_average!(spec, sequence, Hinternal, ρ0, detector, prop_generator, prop_cache, parameters)

Run the simulation of a single crystallite over all γ angles and add the results
to 'spec'.
"""
function γ_average!(spec, sequence::Sequence{T,N}, Hinternal::SphericalTensor{<:Hamiltonian{<:AbstractArray{Complex{T}}}},
        ρ0, detector, prop_generator, prop_cache, parameters::SimulationParameters{M,T}) where {T,N,A,M}

    nγ = parameters.nγ
    γ_steps = parameters.γ_steps

    build_propagators!(prop_cache, Hinternal, parameters)

    for n = 1:nγ
        fill_generator!(prop_generator, prop_cache, (n-1)*γ_steps, parameters)
        propagate!(spec, ρ0, detector, prop_generator)
    end
    return spec
end

function prepare_structures(parameters::SimulationParameters{M,T,A}, sequence::Sequence{T,N}, dims) where {M,T,A,N}
    for n = 1:2
        push!(parameters.temps, Propagator(A(undef, dims)))
    end
    prop_generator = build_generator(sequence, parameters)
    allocate_propagators!(prop_generator, parameters)
    prop_cache = build_prop_cache(prop_generator, dims, parameters)
    return (parameters, prop_generator, prop_cache)
end

"""
    powder_average(sequence, Hint, ρ0, detector, crystallites, parameters)

Run the simulation for each crystallite and return a weighted sum of the
results.
"""
function powder_average(sequence::Sequence{T}, Hint::SphericalTensor, ρ0, detector, crystallites::Crystallites{T},
    parameters::SimulationParameters{CPUSingleMode,T,A}) where {T,A}

    nγ = parameters.nγ

    H = [Hamiltonian(euler_rotation(Hint, crystal_angles)) for crystal_angles in crystallites.angles]

    parameters, prop_generator, prop_cache = prepare_structures(parameters, sequence, size(H[1].s00.data))
    spec = zeros(Complex{T}, 1, prop_generator.size...)
    spec_crystallite = zeros(Complex{T}, 1, prop_generator.size...)

    for n = 1:length(crystallites)
        fill!(spec_crystallite, 0)
        spec .+= crystallites.weights[n].*γ_average!(spec_crystallite, sequence, H[n], ρ0, detector, prop_generator,
            prop_cache, parameters)
    end
    return dropdims(spec, dims=1)./nγ
end

function powder_average(sequence::Sequence{T}, Hint::SphericalTensor, ρ0, detector, crystallites::Crystallites{T},
    parameters::SimulationParameters{GPUSingleMode,T,A}) where {T,A}

    nγ = parameters.nγ

    H = [Hamiltonian(euler_rotation(Hint, crystal_angles), CuArray) for crystal_angles in crystallites.angles]

    parameters, prop_generator, prop_cache = prepare_structures(parameters, sequence, size(H[1].s00.data))
    spec = zeros(Complex{T}, 1, prop_generator.size...)
    spec_crystallite = CuArray(zeros(Complex{T}, length(occupied_columns(detector)), 1, prop_generator.size...))

    for n = 1:length(crystallites)
        fill!(spec_crystallite, 0)
        spec3 = Array(γ_average!(spec_crystallite, sequence, H[n], CuSparseMatrixCSC(ρ0), CuSparseMatrixCSC(detector),
            prop_generator, prop_cache, parameters))
        spec2 = dropdims(sum(spec3, dims=1), dims=1)
        spec .+= crystallites.weights[n].*spec2
    end
    return dropdims(spec, dims=1)./nγ
end

function powder_average(sequence::Sequence{T}, Hint::SphericalTensor, ρ0, detector, crystallites::Crystallites{T},
    parameters::SimulationParameters{GPUBatchedMode,T,A}) where {T,A}

    nγ = parameters.nγ

    H = Hamiltonian([euler_rotation(Hint, crystal_angles) for crystal_angles in crystallites.angles], CuArray)

    parameters, prop_generator, prop_cache = prepare_structures(parameters, sequence, size(H.s00.data))

    spec_d = CuArray(zeros(Complex{T}, length(occupied_columns(detector)), length(crystallites), prop_generator.size...))

    spec3 = Array(γ_average!(spec_d, sequence, H, CuSparseMatrixCSC(ρ0), CuSparseMatrixCSC(detector), prop_generator,
        prop_cache, parameters))
    spec2 = dropdims(sum(spec3, dims=1), dims=1)
    spec2 .*= crystallites.weights
    spec = sum(spec2, dims=1)
    return dropdims(spec, dims=1)./nγ
end

const DStructures{A,T,N} = Tuple{SimulationParameters{CPUMultiProcess,T,A},
                           PropagationGenerator{A,T,N},
                           SimCache{T,N,A}} where {A,T,N}

function powder_average(sequence::Sequence{T,N}, Hint::SphericalTensor, ρ0, detector, crystallites::Crystallites{T},
    parameters::SimulationParameters{CPUMultiProcess,T,A}) where {T,A,N}

    nγ = parameters.nγ

    dims = size(Hint.s00)

    # allocate the necessary data structures on each process
    structures = ddata(T=DStructures{A,T,N}, init=I->prepare_structures(parameters, sequence, dims))

    spec = @distributed (+) for n = 1:length(crystallites)
        parameters, prop_generator, prop_cache = structures[:L]
        spec_crystallite = zeros(Complex{T}, 1, prop_generator.size...)
        H = Hamiltonian(euler_rotation(Hint, crystallites.angles[n]))
        crystallites.weights[n].*γ_average!(spec_crystallite, sequence, H, ρ0, detector, prop_generator, prop_cache,
            parameters)
    end
    return dropdims(spec, dims=1)./nγ
end
