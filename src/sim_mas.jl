import SpecialFunctions.besselj0,SpecialFunctions.besselj1,SpecialFunctions.besselj

"""
    expm_cheby(H, dt, temps)

Generate a propagator from a Hamiltonian ('H') and a time interval ('dt') using
a Chebyshev expansion. Expects 'temps' to be an indexable collection of dense
Arrays of the same size and element type as 'H'.
"""
function expm_cheby(H::Hamiltonian{T,A}, dt, temps) where {T,A}
    nmax = 25
    thresh = T(1E-10)
    bound = eig_max_bound(H.data)
    x = scaledn!(H,bound)
    y = T(-2*dt*pi*bound)
    out = similar(temps[1], Propagator)
    fill_diag!(out, besselj0(y))
    axpy!(2*im*besselj1(y), x, out)

    t1, t2 = temps
    fill_diag!(t1, 1)
    copyto!(t2, x)

    for n = 3:nmax
        mul!(t1, x, t2, 2, -1)
        j = 2*im^(n-1)*besselj(Cint(n-1), y) # Cast to Cint for type stability when using Float32 as of v1.0.0
        axpy!(j, t1, out)
        if threshold(t1.data, T(thresh/abs(j)))
            break
        end
        t1, t2 = t2, t1
    end
    return out
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
    build_generator(sequence, prop_dict, parameters, temp)

Build a generator which can subsequently be used to generate the propagators for
each step of the detection loop of 'sequence'.
"""
function build_generator(sequence::Sequence, prop_dict, parameters, temp::A) where {A}
    detection_loop = sequence.detection_loop
    issorted(detection_loop) || error("detection loop must be sorted")

    first_prop, loop_steps, nonloop_steps, temp = build_first_looped(sequence, prop_dict, parameters, temp)
    loop_props = Vector{PropagationChunk{Looped,A}}()
    for n = 2:length(detection_loop)
        prop, loop_steps, nonloop_steps, temp =
            build_looped(sequence, n, loop_steps, nonloop_steps, prop_dict, parameters, temp)
        push!(loop_props, prop)
    end

    if detection_loop[end]<length(sequence.pulses)
        last_prop, loop_steps, nonloop_steps, temp =
            build_nonlooped(sequence, n, loop_steps, nonloop_steps, prop_dict, parameters, temp)
    else
        last_prop = PropagationChunk{NonLooped}(Vector{A}(), Vector{Vector{A}}())
    end
    prop_generator = PropagationGenerator(first_prop, loop_props, last_prop)
    return prop_generator, temp
end

"""
    propagate!(spec, Uloop, ρ0, detector, prop_generator, temp)

Iterate through 'prop_generator' and calculate the signal using the resulting
propagators, the 'detector' operator, and the initial density operator 'ρ0'.
"""
function propagate!(spec, Uloop, ρ0, detector, prop_generator, temp)
    unique_cols = occupied_columns(detector)
    state = 1
    for n = 1:size(spec, 2)
        Uloop,state,temp = next!(prop_generator, Uloop, state, temp)
        spec = detect!(spec, Uloop, ρ0, detector, unique_cols, n, temp)
    end

    return spec, Uloop, temp
end

function occupied_columns(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    occupied = Vector{Ti}()
    for n = 1:A.n
        A.colptr[n] == A.colptr[n+1] || push!(occupied, n)
    end
    return occupied
end

function combine_propagators!(prop_dict, parameters, temp)
    γ_steps = parameters.γ_steps

    for rf in values(prop_dict)
        a = Set{Tuple{Int,Int}}()
        for timing in keys(rf.timings)
            push!(a, (mod1(timing[1], γ_steps), mod(timing[2], γ_steps)))
        end
        for combination in a
            rf.combinations[combination], temp =
                combine_propagators(rf.step_propagators, combination, parameters, temp)
        end
    end
    return temp
end

function combine_propagators(propagators::A, combination, parameters, temp) where {A}
    period_steps = parameters.period_steps
    nγ = parameters.nγ
    γ_steps = parameters.γ_steps

    combinations = A()

    if combination[2] == 0
        for n = 1:nγ
            position = (n-1)*γ_steps+combination[1]
            U = copy(propagators[mod1(position, period_steps)])
            for m = position+1:position+γ_steps-1
                mul!(temp, propagators[mod1(m, period_steps)], U)
                U, temp = temp, U
            end
            push!(combinations, U)
        end
    else
        for n = 1:nγ
            position = (n-1)*γ_steps+combination[1]
            U1 = copy(propagators[mod1(position, period_steps)])

            for m = position+1:position+combination[2]-1
                mul!(temp,propagators[mod1(m, period_steps)],U1)
                U1, temp = temp, U1
            end
            push!(combinations, U1)

            U2 = copy(propagators[mod1(position+combination[2], period_steps)])
            for m = position+combination[2]+1:position+γ_steps-1
                mul!(temp, propagators[mod1(m, period_steps)], U2)
                U2, temp = temp, U2
            end
            push!(combinations, U2)
        end
    end
    return combinations, temp
end

function build_pulse_props!(prop_dict, parameters, temp)
    for rf in values(prop_dict)
        for timing in keys(rf.timings)
            unphased, temp = build_propagator!(similar(temp), rf, timing, parameters, temp)
            push!(rf.timings[timing].unphased, unphased)
        end
    end
    generate_phased_propagators!(prop_dict, parameters)
    return temp
end

function build_propagator!(U, rf, timing, parameters, temp)
    nγ = parameters.nγ
    γ_steps = parameters.γ_steps

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

    copyto!(U, combinations[mod1(start, cycle_size)])
    for n = start+1:start+remain-1
        mul!(temp, combinations[mod1(n, cycle_size)], U)
        U, temp = temp, U
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
    return U, temp
end

function generate_phased_propagators!(prop_dict, parameters)
    for rf in values(prop_dict)
        for timing in values(rf.timings)
            unphased = timing.unphased[1]
            for phase in timing.phases
                if ! haskey(timing.phased, phase)
                    timing.phased[phase] = similar(unphased)
                end

                if all(phase .== 0)
                    copyto!(timing.phased[phase], unphased)
                else
                    A = typeof(unphased.data)
                    rotator = array_wrapper_type(A)(phase_rotator(phase, parameters.xyz))
                    rotate!(timing.phased[phase], unphased, rotator)
                end
            end
        end
    end
end

function γiterate_pulse_propagators!(prop_dict, parameters, γ_iteration, temp)
    γ_steps = parameters.γ_steps

    for rf in values(prop_dict)
        for timing_pair in rf.timings
            timing, timing_collection = timing_pair
            if timing[2] <= 2*γ_steps
                iterated_timing = tuple(timing[1]+γ_steps*(γ_iteration-1),timing[2])
                timing_collection.unphased[1], temp =
                    build_propagator!(timing_collection.unphased[1], rf, iterated_timing, parameters, temp)
            else
                γiterate_propagator!(timing_collection.unphased[1], rf, timing, parameters, γ_iteration, temp)
            end
        end
    end
    generate_phased_propagators!(prop_dict, parameters)
    return temp
end

function γiterate_propagator!(U, rf, timing, parameters, γ_iteration, temp)
    γ_steps = parameters.γ_steps
    nγ = parameters.nγ

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
end

function build_step_propagators!(prop_dict, Hinternal::SphericalTensor{Hamiltonian{T,A}}, parameters) where {T,N,A}
    period_steps = parameters.period_steps
    angles = parameters.angles

    Hrotated = Vector{Hamiltonian{T,A}}()
    for n = 1:period_steps
        angles2 = EulerAngles{T}(angles.α+360/period_steps*(n-1), angles.β, angles.γ)
        H = Hamiltonian(rotate_component2(Hinternal, 0, angles2).data.+Hinternal.s00.data)
        push!(Hrotated, H)
    end
    temps = [Hamiltonian(similar(Hrotated[1].data, T)) for j = 1:2]
    for rf_pair in prop_dict
        step_propagators!(rf_pair, Hrotated, parameters, temps)
    end
end

function step_propagators!(rf_pair, Hrotated::Vector{Hamiltonian{T,A}}, parameters, temps) where {T,A}
    period_steps = parameters.period_steps
    step_size = parameters.step_size
    xyz = parameters.xyz
    rf = rf_pair[1]
    propagators = rf_pair[2].step_propagators

    Hrf = array_wrapper_type(A)(pulse_H(rf,xyz))
    for n = 1:period_steps
        H = real_add(Hrotated[n], Hrf)
        U = expm_cheby(H, step_size/10^6, temps)
        push!(propagators, U)
    end
    return propagators
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
function A_mul_Bc_single_element(A::Array{T,3}, B::Array{T,3}, row, col, n) where {T}
    x, y, z = size(A)
    (x, y, z) == size(B) || throw(DimensionMismatch)
    x == y || throw(DimensionMismatch)
    row <= x || throw(DomainError)
    col <= x || throw(DomainError)

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
    γ_average!(spec, sequence, Hinternal, ρ0, detector, parameters)

Run the simulation of a single crystallite over all γ angles and add the results
to 'spec'.
"""
function γ_average!(spec, sequence::Sequence{T,N}, Hinternal::SphericalTensor{Hamiltonian{T,A}}, ρ0, detector,
        parameters::SimulationParameters{M,T}) where {T,N,A,M}

    steps = parameters.period_steps
    step_size = parameters.step_size
    nγ = parameters.nγ

    temp = similar(Hinternal.s00, Propagator)

    prop_dict = Dict{NTuple{N,T}, PropagatorCollectionRF{T,N,A}}()
    find_pulses!(prop_dict, sequence, 1, parameters)
    build_step_propagators!(prop_dict, Hinternal, parameters)
    temp = combine_propagators!(prop_dict, parameters, temp)
    temp = build_pulse_props!(prop_dict, parameters, temp)

    if M <: GPUBatchedMode
        GC.gc()
    end

    Uloop = similar(temp)
    for n = 1:nγ
        if n != 1
            temp = γiterate_pulse_propagators!(prop_dict, parameters, n, temp)
        end

        prop_generator, temp = build_generator(sequence, prop_dict, parameters, temp)
        spec, Uloop, temp = propagate!(spec, Uloop, ρ0, detector, prop_generator, temp)
    end
    return spec
end

"""
    get_crystallites(crystal_file, T=Float64)

Read 'crystal_file' and return a vector of EulerAngles{T} using α and β values
from the file and γ=0 as well as a vector of weights for each crystallite.
"""
function get_crystallites(crystal_file, ::Type{T}=Float64) where {T<:AbstractFloat}
    f = open(crystal_file)
    number = parse(Int64, chomp(readline(f)))
    raw = readlines(f)
    close(f)

    angles = Array{EulerAngles{T}}(undef, number)
    weights = Array{T}(undef, number)
    for n = 1:number
        as_str = split(chomp(raw[n]))
        angles[n] = EulerAngles{T}(parse(T, as_str[1]), parse(T, as_str[2]), 0)
        weights[n] = parse(T, as_str[3])
    end
    return angles, weights
end

"""
    powder_average(sequence, Hint, ρ0, detector, crystallites, weights, parameters)

Run the simulation for each crystallite and return a weighted sum of the
results.
"""
function powder_average(sequence::Sequence{T}, Hint::SphericalTensor{A}, ρ0, detector,
    crystallites::Vector{EulerAngles{T}}, weights::Vector{T}, parameters::SimulationParameters{CPUSingleMode,T}) where {T,A}

    nγ = parameters.nγ
    loops = sequence.repeats
    spec = zeros(Complex{T}, 1, loops)
    spec_crystallite = zeros(Complex{T}, 1, loops)

    H = [Hamiltonian(euler_rotation(Hint, crystal_angles)) for crystal_angles in crystallites]

    for n = 1:length(crystallites)
        fill!(spec_crystallite, 0)
        spec .+= weights[n].*γ_average!(spec_crystallite, sequence, H[n], ρ0, detector, parameters)
    end
    return spec./nγ
end

function powder_average(sequence::Sequence{T}, Hint::SphericalTensor{A}, ρ0, detector,
    crystallites::Vector{EulerAngles{T}}, weights::Vector{T}, parameters::SimulationParameters{GPUSingleMode,T}) where {T,A}

    nγ = parameters.nγ
    loops = sequence.repeats
    spec = zeros(Complex{T}, 1, loops)
    spec_crystallite = CuArray(zeros(Complex{T}, 1, loops, length(occupied_columns(detector))))

    H = [Hamiltonian(euler_rotation(Hint, crystal_angles), CuArray) for crystal_angles in crystallites]

    for n = 1:length(crystallites)
        fill!(spec_crystallite, 0)
        spec3 = Array(γ_average!(spec_crystallite, sequence, H[n], CuSparseMatrixCSC(ρ0), CuSparseMatrixCSC(detector), parameters))
        spec2 = dropdims(sum(spec3, dims=3), dims=3)
        spec .+= weights[n].*spec2
    end
    return spec./nγ
end

function powder_average(sequence::Sequence{T}, Hint::SphericalTensor{A}, ρ0, detector,
    crystallites::Vector{EulerAngles{T}}, weights::Vector{T},parameters::SimulationParameters{GPUBatchedMode,T}) where {T,A}

    nγ = parameters.nγ
    loops = sequence.repeats

    H = Hamiltonian([euler_rotation(Hint, crystal_angles) for crystal_angles in crystallites], CuArray)

    spec_d = CuArray(zeros(Complex{T}, length(crystallites), loops, length(occupied_columns(detector))))

    spec3 = Array(γ_average!(spec_d, sequence, H, CuSparseMatrixCSC(ρ0), CuSparseMatrixCSC(detector), parameters))
    spec2 = dropdims(sum(spec3, dims=3), dims=3)
    spec2 .*= weights
    spec = sum(spec2, dims=1)
    return spec./nγ
end

function powder_average(sequence::Sequence{T}, Hint::SphericalTensor{A}, ρ0, detector,
    crystallites::Vector{EulerAngles{T}}, weights::Vector{T},parameters::SimulationParameters{CPUMultiProcess,T}) where {T,A}

    nγ = parameters.nγ
    loops = sequence.repeats

    spec = @distributed (+) for n = 1:length(crystallites)
        spec_crystallite = zeros(Complex{T}, 1, loops)
        H = Hamiltonian(euler_rotation(Hint, crystallites[n]))
        weights[n].*γ_average!(spec_crystallite, sequence, H, ρ0, detector, parameters)
    end
    return spec./nγ
end
