import SpecialFunctions: besselj0, besselj1, besselj

"""
    expm_cheby!(U, H, dt, temps)

Modify 'U' to hold a propagator generated from a Hamiltonian ('H') and a time
interval ('dt') using a Chebyshev expansion. Expects 'temps' to be an indexable
collection of two arrays similar to the contents of 'H'.
"""
function expm_cheby!(U, H::Hamiltonian{T,A}, dt, temps) where {T,A}
    nmax = 25
    thresh = T(1E-10)
    bound = eig_max_bound(H.data)
    x = scaledn!(H,bound)
    y = T(-2*dt*pi*bound)
    fill_diag!(U, besselj0(y))
    axpy!(2*im*besselj1(y), x, U)

    t1, t2 = temps
    fill_diag!(t1, 1)
    copyto!(t2, x)

    for n = 3:nmax
        mul!(t1, x, t2, 2, -1)
        j = 2*im^(n-1)*besselj(Cint(n-1), y) # Cast to Cint for type stability when using Float32 as of v1.0.0
        axpy!(j, t1, U)
        if threshold(t1.data, T(thresh/abs(j)))
            break
        end
        t1, t2 = t2, t1
    end
    return U
end

"""
    threshold(A, thresh)

Return true if the absolute value of all elements in A are greater than thresh.
Otherwise return false.
"""
function threshold(A, thresh)
    for n in A
        if abs(n)>thresh
            return false
        end
    end
    return true
end

"""
    eig_max_bound(A)

Calculate an upper bound for the eigenvalues of 'A' based on Gershgorin's
    theorem.
"""
function eig_max_bound(A)
    x, y = size(A)
    out = zero(get_precision(A))
    @inbounds for j = 1:y
        current = zero(get_precision(A))
        for k = 1:x
            current += abs(A[k,j])
        end
        out = max(out, current)
    end
    return out
end

"""
    propagate!(spec, Uloop, ρ0, detector, prop_generator, parameters)

Iterate through 'prop_generator' and calculate the signal using the resulting
propagators, the 'detector' operator, and the initial density operator 'ρ0'.
"""
function propagate!(spec, Uloop, ρ0, detector, prop_generator, parameters)
    unique_cols = occupied_columns(detector)
    state = 1
    for n = 1:size(spec, 2)
        Uloop, state = next!(prop_generator, Uloop, state, parameters.temps)
        spec = detect!(spec, Uloop, ρ0, detector, unique_cols, n, parameters.temps[1])
    end

    return spec, Uloop
end

function find_pulses!(prop_cache, pulse::Pulse, start, parameters) where {T,N}
    period_steps = parameters.period_steps
    step_size = parameters.step_size
    pulse_cache = prop_cache.pulses

    steps = Int(pulse.t/step_size)
    rf = pulse.γB1
    if ! haskey(pulse_cache, rf)
        add_rf!(pulse_cache, rf)
    end

    timing = (mod1(start, period_steps), steps)
    if ! haskey(pulse_cache[rf].timings, timing)
        add_timing!(pulse_cache[rf], timing)
    end

    push!(pulse_cache[rf].timings[timing].phases, pulse.phase)
    return steps
end

function find_pulses!(prop_cache, block::Block, start, parameters)
    period_steps = parameters.period_steps
    block_cache = prop_cache.blocks

    start_period = mod1(start, period_steps)
    cache_key = (block, start_period)
    if haskey(block_cache, cache_key)
        return steps(block_cache, cache_key)
    end

    step_total = 0
    if block.repeats == 1
        for element in block.pulses
            step_total += find_pulses!(prop_cache, element, start+step_total, parameters)
        end
    else
        single_repeat = Block(block.pulses, 1)
        for j = 1:block.repeats
            step_total += find_pulses!(prop_cache, single_repeat, start+step_total, parameters)
        end
    end
    add_block!(block_cache, cache_key, step_total)
    return step_total
end

function find_pulses!(prop_cache, prop_generator::PropagationGenerator, parameters)
    for chunk in prop_generator.loops
        find_pulses!(prop_cache, chunk, parameters)
    end
    find_pulses!(prop_cache, prop_generator.nonloop, parameters)
    return prop_cache
end

function find_pulses!(prop_cache, chunk::PropagationChunk, parameters)
    for elements in (chunk.initial_elements, chunk.incrementor_elements)
        for index in eachindex(elements)
            if isassigned(elements, index)
                find_pulses!(prop_cache, elements[index][1], elements[index][2], parameters)
            end
        end
    end
    return prop_cache
end

function build_block_props!(prop_cache, parameters)
    for collection in prop_cache.blocks.ranks
        for key in keys(collection.steps)
            build_block_prop!(prop_cache, key, parameters)
        end
    end
    return prop_cache
end

function build_block_prop!(prop_cache, key, parameters)
    U, _ = prop_cache.blocks[key]
    block, start = key
    step_total = 0
    if block.repeats == 1
        U = build_nonrepeat_block!(U, block, start, prop_cache, parameters)
    else
        U = build_repeat_block!(U, block, start, prop_cache, parameters)
    end
    prop_cache.blocks[key] = U
    return prop_cache
end

function build_nonrepeat_block!(U, block, start, prop_cache, parameters)
    Uelement, step_total = fetch_propagator(block.pulses[1], start, prop_cache, parameters)
    copyto!(U, Uelement)
    for element in block.pulses[2:end]
        Uelement, steps = fetch_propagator(element, start+step_total, prop_cache, parameters)
        mul!(parameters.temps[1], Uelement,U)
        step_total += steps
        U, parameters.temps[1] = parameters.temps[1], U
    end
    return U
end

function build_repeat_block!(U, block, start, prop_cache, parameters)
    single_repeat = Block(block.pulses, 1)
    Uelement, step_total = fetch_propagator(single_repeat, start, prop_cache, parameters)
    copyto!(U, Uelement)
    for j = 2:block.repeats
        Uelement, steps = fetch_propagator(single_repeat, start+step_total, prop_cache, parameters)
        mul!(parameters.temps[1], Uelement,U)
        step_total += steps
        U, parameters.temps[1] = parameters.temps[1], U
    end
    return U
end

function occupied_columns(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    occupied = Vector{Ti}()
    for n = 1:A.n
        A.colptr[n] == A.colptr[n+1] || push!(occupied, n)
    end
    return occupied
end

function combine_propagators!(rf, step_propagators, parameters)
    γ_steps = parameters.γ_steps

    for (combination, combinations) in rf.combinations
        combine_propagators(combinations, step_propagators, combination, parameters)
    end
    return rf
end

function combine_propagators(combinations, propagators, combination, parameters)
    period_steps = parameters.period_steps
    nγ = parameters.nγ
    γ_steps = parameters.γ_steps
    temp = parameters.temps[1]

    if combination[2] == 0
        for n = 1:nγ
            U = combinations[n]
            position = (n-1)*γ_steps+combination[1]
            copyto!(U, propagators[mod1(position, period_steps)])
            for m = position+1:position+γ_steps-1
                mul!(temp, propagators[mod1(m, period_steps)], U)
                U, temp = temp, U
            end
            combinations[n] = U
        end
    else
        for n = 1:nγ
            U1 = combinations[2*n-1]
            position = (n-1)*γ_steps+combination[1]
            copyto!(U1, propagators[mod1(position, period_steps)])

            for m = position+1:position+combination[2]-1
                mul!(temp,propagators[mod1(m, period_steps)],U1)
                U1, temp = temp, U1
            end
            combinations[2*n-1] = U1

            U2 = combinations[2*n]
            copyto!(U2, propagators[mod1(position+combination[2], period_steps)])
            for m = position+combination[2]+1:position+γ_steps-1
                mul!(temp, propagators[mod1(m, period_steps)], U2)
                U2, temp = temp, U2
            end
            combinations[2*n] = U2
        end
    end
    parameters.temps[1] = temp
    return combinations
end

function build_pulse_props!(pulse_cache, parameters)
    for rf in values(pulse_cache)
        for (timing, timing_cache) in rf.timings
            timing_cache.unphased[1] = build_propagator!(timing_cache.unphased[1], rf, timing, parameters)
        end
    end
    generate_phased_propagators!(pulse_cache, parameters)
    return pulse_cache
end

function build_propagator!(U, rf, timing, parameters)
    nγ = parameters.nγ
    γ_steps = parameters.γ_steps
    temp = parameters.temps[1]

    combinations = rf.combinations[(mod1(timing[1], γ_steps), mod(timing[2], γ_steps))]

    if mod(timing[2], γ_steps) == 0
        start = fld1(timing[1], γ_steps)
        steps = div(timing[2], γ_steps)

        cycles, remain = divrem(steps, nγ)
        cycle_size = nγ
    else
        start = fld1(timing[1], γ_steps)*2-1
        steps = div(timing[2], γ_steps)*2+1

        cycles, remain = divrem(steps, 2*nγ)
        cycle_size = 2*nγ
    end

    if remain != 0
        copyto!(U, combinations[mod1(start, cycle_size)])
        for n = start+1:start+remain-1
            mul!(temp, combinations[mod1(n, cycle_size)], U)
            U, temp = temp, U
        end
    else
        fill_diag!(U,1)
    end

    if cycles>0
        remainder = copy(U)
        for n = start+remain:start+cycle_size-1
            mul!(temp, combinations[mod1(n, cycle_size)], U)
            U,temp = temp, U
        end
        U, temp = pow!(similar(U), U, cycles, temp)
        mul!(temp, remainder, U)
        U, temp = temp, U
    end
    parameters.temps[1] = temp
    return U
end

function generate_phased_propagators!(pulse_cache, parameters)
    for rf in values(pulse_cache)
        for timing in values(rf.timings)
            unphased = timing.unphased[1]
            for phase in timing.phases
                if all(phase .== 0)
                    copyto!(timing.phased[phase], unphased)
                else
                    phase_rotate!(timing.phased[phase], unphased, phase, parameters)
                end
            end
        end
    end
    return pulse_cache
end

function γiterate_pulse_propagators!(pulse_cache, parameters, γ_iteration)
    γ_steps = parameters.γ_steps

    for rf in values(pulse_cache)
        for timing_pair in rf.timings
            timing, timing_collection = timing_pair
            if timing[2] <= 2*γ_steps
                iterated_timing = tuple(timing[1]+γ_steps*(γ_iteration-1),timing[2])
                timing_collection.unphased[1] =
                    build_propagator!(timing_collection.unphased[1], rf, iterated_timing, parameters)
            else
                γiterate_propagator!(timing_collection.unphased[1], rf, timing, parameters, γ_iteration)
            end
        end
    end
    generate_phased_propagators!(pulse_cache, parameters)
    return pulse_cache
end

function γiterate_propagator!(U, rf, timing, parameters, γ_iteration)
    γ_steps = parameters.γ_steps
    nγ = parameters.nγ
    temp = parameters.temps[1]

    combinations = rf.combinations[(mod1(timing[1], γ_steps), mod(timing[2], γ_steps))]

    if mod(timing[2], γ_steps) == 0
        start = fld1(timing[1], γ_steps)+γ_iteration-2
        stop = div(timing[2], γ_steps)+start-1

        mul!(temp, U, combinations[mod1(start,nγ)], 'N', 'C')
        mul!(U, combinations[mod1(stop+1, nγ)], temp)
    else
        start = fld1(timing[1], γ_steps)*2-1+2*(γ_iteration-2)
        stop = div(timing[2], γ_steps)*2+start

        mul!(temp, U, combinations[mod1(start, 2*nγ)], 'N', 'C')
        mul!(U, temp, combinations[mod1(start+1, 2*nγ)], 'N', 'C')
        mul!(temp, combinations[mod1(stop+1,2*nγ)], U)
        mul!(U, combinations[mod1(stop+2,2*nγ)], temp)
    end
    return U
end

function build_combined_propagators!(prop_cache, Hinternal::SphericalTensor{Hamiltonian{T,A}}, parameters) where {T,N,A}
    period_steps = parameters.period_steps
    angles = parameters.angles
    step_propagators = prop_cache.step_propagators

    Hrotated = prop_cache.step_hamiltonians
    for n = 1:period_steps
        angles2 = EulerAngles{T}(angles.α+360/period_steps*(n-1), angles.β, angles.γ)
        Hrotated[n].data .= rotate_component2(Hinternal, 0, angles2).data.+Hinternal.s00.data
    end
    temps = [Hamiltonian(similar(Hrotated[1].data, T)) for j = 1:2]
    for (γB1, rf_cache) in prop_cache.pulses
        step_propagators!(step_propagators, γB1, Hrotated, parameters, temps)
        combine_propagators!(rf_cache, step_propagators, parameters)
    end
    return prop_cache
end

function step_propagators!(propagators, rf, Hrotated::Vector{Hamiltonian{T,A}}, parameters, temps) where {T,A}
    period_steps = parameters.period_steps
    step_size = parameters.step_size
    xyz = parameters.xyz

    Hrf = array_wrapper_type(A)(pulse_H(rf,xyz))
    H = similar(temps[1])
    for n = 1:period_steps
        H = real_add!(H, Hrotated[n], Hrf)
        U = expm_cheby!(propagators[n], H, step_size/10^6, temps)
    end
    return propagators
end

function allocate_combinations!(rf_cache::PropagatorCollectionRF{T,N,A}, parameters) where {T,N,A}
    nγ = parameters.nγ
    γ_steps = parameters.γ_steps

    unique_timings = Set{Tuple{Int,Int}}()
    for timing in keys(rf_cache.timings)
        push!(unique_timings, (mod1(timing[1], γ_steps), mod(timing[2], γ_steps)))
    end
    for timing in unique_timings
        num_combinations = timing[2] == 0 ? nγ : 2*nγ
        rf_cache.combinations[timing] = [similar(parameters.temps[1]) for n=1:num_combinations]
    end
end

function allocate_propagators!(pulse_cache, parameters)
    for rf_cache in values(pulse_cache)
        allocate_combinations!(rf_cache, parameters)
        for (timing, timing_cache) in rf_cache.timings
            push!(timing_cache.unphased, similar(parameters.temps[1]))
            for phase in timing_cache.phases
                timing_cache.phased[phase] = similar(parameters.temps[1])
            end
        end
    end
end

function allocate_propagators!(block_cache::BlockCache, parameters)
    for collection in block_cache.ranks
        for key in keys(collection.steps)
            collection.propagators[key] = similar(parameters.temps[1])
        end
    end
end

function allocate_propagators!(prop_generator::PropagationGenerator, parameters)
    for chunk in prop_generator.loops
        for index in eachindex(chunk.current)
            chunk.current[index] = similar(parameters.temps[1])
        end
    end
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

function build_prop_cache(prop_generator::PropagationGenerator{A,T,N}, dims, parameters) where {A,T,N}
    period_steps = parameters.period_steps

    prop_cache = SimCache{T,N,A}(period_steps)
    for n = 1:period_steps
        prop_cache.step_hamiltonians[n] = Hamiltonian(A(undef, dims))
        prop_cache.step_propagators[n] = Propagator(A(undef, dims))
    end
    find_pulses!(prop_cache, prop_generator, parameters)
    allocate_propagators!(prop_cache.pulses, parameters)
    allocate_propagators!(prop_cache.blocks, parameters)
    return prop_cache
end

"""
    γ_average!(spec, sequence, Hinternal, ρ0, detector, prop_generator, prop_cache, parameters)

Run the simulation of a single crystallite over all γ angles and add the results
to 'spec'.
"""
function γ_average!(spec, sequence::Sequence{T,N}, Hinternal::SphericalTensor{Hamiltonian{T,A}}, ρ0, detector,
        prop_generator, prop_cache, parameters::SimulationParameters{M,T}) where {T,N,A,M}

    steps = parameters.period_steps
    step_size = parameters.step_size
    nγ = parameters.nγ

    build_combined_propagators!(prop_cache, Hinternal, parameters)
    build_pulse_props!(prop_cache.pulses, parameters)

    Uloop = similar(parameters.temps[1])
    for n = 1:nγ
        if n != 1
            γiterate_pulse_propagators!(prop_cache.pulses, parameters, n)
        end

        build_block_props!(prop_cache, parameters)
        prop_generator = fill_generator!(prop_generator, prop_cache, parameters)
        spec, Uloop = propagate!(spec, Uloop, ρ0, detector, prop_generator, parameters)
    end
    return spec
end

function prepare_structures(parameters::SimulationParameters{M,T,A}, sequence::Sequence{T,N}, dims) where {M,T,A,N}
    push!(parameters.temps, Propagator(A(undef, dims)))
    prop_generator = build_generator(sequence, parameters, A)
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
    loops = sequence.repeats
    spec = zeros(Complex{T}, 1, loops)
    spec_crystallite = zeros(Complex{T}, 1, loops)

    H = [Hamiltonian(euler_rotation(Hint, crystal_angles)) for crystal_angles in crystallites.angles]

    parameters, prop_generator, prop_cache = prepare_structures(parameters, sequence, size(H[1].s00.data))

    for n = 1:length(crystallites)
        fill!(spec_crystallite, 0)
        spec .+= crystallites.weights[n].*γ_average!(spec_crystallite, sequence, H[n], ρ0, detector, prop_generator,
            prop_cache, parameters)
    end
    return spec./nγ
end

function powder_average(sequence::Sequence{T}, Hint::SphericalTensor, ρ0, detector, crystallites::Crystallites{T},
    parameters::SimulationParameters{GPUSingleMode,T,A}) where {T,A}

    nγ = parameters.nγ
    loops = sequence.repeats
    spec = zeros(Complex{T}, 1, loops)
    spec_crystallite = CuArray(zeros(Complex{T}, 1, loops, length(occupied_columns(detector))))

    H = [Hamiltonian(euler_rotation(Hint, crystal_angles), CuArray) for crystal_angles in crystallites.angles]

    parameters, prop_generator, prop_cache = prepare_structures(parameters, sequence, size(H[1].s00.data))

    for n = 1:length(crystallites)
        fill!(spec_crystallite, 0)
        spec3 = Array(γ_average!(spec_crystallite, sequence, H[n], CuSparseMatrixCSC(ρ0), CuSparseMatrixCSC(detector),
            prop_generator, prop_cache, parameters))
        spec2 = dropdims(sum(spec3, dims=3), dims=3)
        spec .+= crystallites.weights[n].*spec2
    end
    return spec./nγ
end

function powder_average(sequence::Sequence{T}, Hint::SphericalTensor, ρ0, detector, crystallites::Crystallites{T},
    parameters::SimulationParameters{GPUBatchedMode,T,A}) where {T,A}

    nγ = parameters.nγ
    loops = sequence.repeats

    H = Hamiltonian([euler_rotation(Hint, crystal_angles) for crystal_angles in crystallites.angles], CuArray)

    parameters, prop_generator, prop_cache = prepare_structures(parameters, sequence, size(H.s00.data))

    spec_d = CuArray(zeros(Complex{T}, length(crystallites), loops, length(occupied_columns(detector))))

    spec3 = Array(γ_average!(spec_d, sequence, H, CuSparseMatrixCSC(ρ0), CuSparseMatrixCSC(detector), prop_generator,
        prop_cache, parameters))
    spec2 = dropdims(sum(spec3, dims=3), dims=3)
    spec2 .*= crystallites.weights
    spec = sum(spec2, dims=1)
    return spec./nγ
end

const DStructures{A,T,N} = Tuple{SimulationParameters{CPUMultiProcess,T,A},
                           PropagationGenerator{A,T,N},
                           SimCache{T,N,A}} where {A,T,N}

function powder_average(sequence::Sequence{T,N}, Hint::SphericalTensor, ρ0, detector, crystallites::Crystallites{T},
    parameters::SimulationParameters{CPUMultiProcess,T,A}) where {T,A,N}

    nγ = parameters.nγ
    loops = sequence.repeats

    dims = size(Hint.s00)

    # allocate the necessary data structures on each process
    structures = ddata(T=DStructures{A,T,N}, init=I->prepare_structures(parameters, sequence, dims))

    spec = @distributed (+) for n = 1:length(crystallites)
        parameters, prop_generator, prop_cache = structures[:L]
        spec_crystallite = zeros(Complex{T}, 1, loops)
        H = Hamiltonian(euler_rotation(Hint, crystallites.angles[n]))
        crystallites.weights[n].*γ_average!(spec_crystallite, sequence, H, ρ0, detector, prop_generator, prop_cache,
            parameters)
    end
    return spec./nγ
end
