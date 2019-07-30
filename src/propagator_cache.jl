import SpecialFunctions: besselj0, besselj1, besselj
import Base: haskey, getindex, setindex!
import LinearAlgebra: BLAS.BlasReal

"""
    PropagatorCollectionTiming

A 'PropagatorCollectionTiming' contains a cache of propagators that all share
the same RF powers and timing relative to the rotor period. 'phases' stores the
different combinations of RF phases that are used with this timing and set of
powers. 'unphased' holds a propagator with phase of 0 on all channels. 'phased'
holds the set of propagators with the phases in 'phases'. These can be generated
by rotating the 0 phase propagator for the full pulse, because these rotations
use the Z operator for each channel. The secular internal spin hamiltonian
must commute with the Z operators at all times and therefore this rotation only
alters the RF pulse term in the hamiltonian.
"""
struct PropagatorCollectionTiming{T<:BlasReal,N,A<:AbstractArray{Complex{T}}}
    phases::Set{NTuple{N,T}}
    unphased::Vector{Propagator{A}}
    phased::Dict{NTuple{N,T}, Propagator{A}}

    PropagatorCollectionTiming{T,N,A}() where {T,N,A} = new(Set{NTuple{N,T}}(), Vector{Propagator{A}}(),
        Dict{NTuple{N,T}, Propagator{A}}())
end

"""
    PropagatorCollectionRF

A 'PropagatorCollectionRF' contains a cache of propagators that all share the
same RF powers. 'timings' holds propagators that correspond to a unique time
period for a pulse. The keys for 'timings' are a tuple holding the starting step
modulo the rotor period and the length of the pulse in steps.

'combinations' holds combined propagators taht are used to construct the
propagators in 'timings'. The keys for 'combinations' are a tuple containing the
starting step and number of steps modulo the number of steps per γ angle. Thus
multiple entries in 'timings' can correspond to a single entry in
'combinations'.
"""
struct PropagatorCollectionRF{T<:BlasReal,N,A<:AbstractArray{Complex{T}}}
    combinations::Dict{NTuple{2,Int}, Vector{Propagator{A}}}
    timings::Dict{NTuple{2,Int}, PropagatorCollectionTiming{T,N,A}}

    PropagatorCollectionRF{T,N,A}() where {T,N,A } = new(Dict{NTuple{2,Int}, Vector{Propagator{A}}}(),
        Dict{NTuple{2,Int}, PropagatorCollectionTiming{T,N,A}}())
end

"""
    add_timing!(collection, timing)

Allocate and add a new PropagatorCollectionTiming to 'collection' if the
'timing' is not already present.
"""
function add_timing!(collection::PropagatorCollectionRF{T,N,A}, timing::NTuple{2,Int}) where {T,N,A}
    if ! haskey(collection.timings, timing)
        collection.timings[timing] = PropagatorCollectionTiming{T,N,A}()
    end
end

"""
    PulseCache

A PulseCache contains a cache of propagators corresponding to Pulses but not
Blocks, organized by their rf powers.
"""
const PulseCache{T,N,A} = Dict{NTuple{N,T}, PropagatorCollectionRF{T,N,A}} where {T,N,A}

"""
    add_rf!(pulse_cache, rf)

Allocate and add a new PropagatorCollectionRF to 'pulse_cache' if the specified
'rf' powers are not already present.
"""
function add_rf!(pulse_cache::PulseCache{T,N,A}, rf) where {T,N,A}
    if ! haskey(pulse_cache, rf)
        pulse_cache[rf] = PropagatorCollectionRF{T,N,A}()
    end
end

"""
    PropagatorCollectionBlock

A 'PropagatorCollectionBlock' contains a cache of propagators for Blocks that
all share the same rank. 'steps' is a Dict storing the number of steps for each
key, while 'propagators' is a Dict storing the propagator for each key. The keys
are Blocks and the step they start on. This combination fully specifies which
propagator to construct.
"""
struct PropagatorCollectionBlock{T<:BlasReal,N,A<:AbstractArray{Complex{T}}}
    steps::Dict{Tuple{Block{T,N}, Int}, Int}
    propagators::Dict{Tuple{Block{T,N}, Int}, Propagator{A}}

    PropagatorCollectionBlock{T,N,A}() where {T,N,A} = new{T,N,A}(Dict{Tuple{Block{T,N}, Int}, Int}(),
                                                                  Dict{Tuple{Block{T,N}, Int}, Propagator{A}}())
end

"""
    BlockCache

A BlockCache stores a vector of PropagatorCollectionBlocks each of which caches
the propagators for Blocks of a given rank. Organization by rank allows for
constructing propagators for lower rank Blocks first.
"""
struct BlockCache{T,N,A}
    ranks::Vector{PropagatorCollectionBlock{T,N,A}}

    BlockCache{T,N,A}() where {T,N,A} = new{T,N,A}(Vector{PropagatorCollectionBlock{T,N,A}}())
end

"""
    add_block!(block_cache, key, steps)

Add the block specified by 'key' to the 'block_cache'. The number of 'steps' in
the block is also stored in the cache.
"""
function add_block!(block_cache::BlockCache{T,N,A}, key::Tuple{Block{T,N}, Int}, steps::Int) where {T,N,A}
    rank = key[1].rank
    while rank > length(block_cache.ranks)
        push!(block_cache.ranks, PropagatorCollectionBlock{T,N,A}())
    end
    block_cache.ranks[rank].steps[key] = steps
end

"""
    steps(block_cache, key)

Fetch the number of steps in the block specified by 'key' from the
'pulse_cache'.
"""
function steps(block_cache::BlockCache{T,N}, key::Tuple{Block{T,N}, Int}) where {T,N}
    rank = key[1].rank
    rank <= length(block_cache.ranks) || throw(KeyError("key $key not found"))
    return block_cache.ranks[rank].steps[key]
end

"""
    haskey(block_cache, key)

Check whether the block specified by 'key' is in the 'pulse_cache'.
"""
function haskey(block_cache::BlockCache{T,N}, key::Tuple{Block{T,N}, Int}) where {T,N}
    rank = key[1].rank
    if rank > length(block_cache.ranks)
        return false
    end
    return haskey(block_cache.ranks[rank].steps, key)
end

"""
    getindex(block_cache, key)

Fetch the propagator corresponding to and number of steps in the block specified
by 'key' from the 'pulse_cache'.
"""
function getindex(block_cache::BlockCache{T,N}, key::Tuple{Block{T,N}, Int}) where {T,N}
    rank = key[1].rank
    rank <= length(block_cache.ranks) || throw(KeyError("key $key not found"))
    haskey(block_cache.ranks[rank].steps, key) || throw(KeyError("key $key not found"))
    haskey(block_cache.ranks[rank].propagators, key) || throw(KeyError("key $key not initiallized"))
    return block_cache.ranks[rank].propagators[key], block_cache.ranks[rank].steps[key]
end

"""
    setindex!(block_cache, prop, key)

Store the propagator ('prop') corresponding to the block specified by 'key' from
the 'pulse_cache'.
"""
function setindex!(block_cache::BlockCache{T,N,A}, prop::Propagator{A}, key::Tuple{Block{T,N}, Int}) where {T,N,A}
    rank = key[1].rank
    rank <= length(block_cache.ranks) || throw(KeyError("key $key not initiallized"))
    haskey(block_cache.ranks[rank].steps, key) || throw(KeyError("key $key not initiallized"))
    block_cache.ranks[rank].propagators[key] = prop
end

"""
    SimCache

A SimCache stores caches of pulse and block propagators and has pre-allocated
operators for hamiltonians and propagators corresponding to each step in a rotor
period.
"""
struct SimCache{T<:BlasReal,N,A<:AbstractArray{Complex{T}}}
    step_hamiltonians::Vector{Hamiltonian{A}}
    step_propagators::Vector{Propagator{A}}
    pulses::PulseCache{T,N,A}
    blocks::BlockCache{T,N,A}

    SimCache{T,N,A}(steps) where {T,N,A} = new{T,N,A}(Vector{Hamiltonian{A}}(undef, steps),
        Vector{Propagator{A}}(undef,steps), Dict{NTuple{N,T}, PropagatorCollectionRF{T,N,A}}(), BlockCache{T,N,A}())
end

"""
    build_prop_cache(prop_generator, dims, parameters)

Build a SimCache based on the pulse sequence element specified in a
PropagationGenerator ('prop_generator'). The cache structures are built and
operators are allocated but the actual correct propagators are not constructed.
'dims' specifies the size of the allocated operators. Return the SimCache.
"""
function build_prop_cache(prop_generator::PropagationGenerator{T,N,A}, dims, parameters::SimulationParameters{M,T,A}) where {M,T,N,A}
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
    find_pulses!(pulse_cache, prop_generator, parameters)

Add all of the pulse sequence elements specified by a PropagationGenerator
('prop_generator') to the 'pulse_cache'.
"""
function find_pulses!(prop_cache::SimCache{T,N,A}, prop_generator::PropagationGenerator{T,N,A},
        parameters::SimulationParameters{M,T,A}) where {M,T,N,A}

    for dim in prop_generator.loops
        for chunk in dim.chunks
            find_pulses!(prop_cache, chunk.current, parameters)
            find_pulses!(prop_cache, chunk.incrementors, parameters)
        end
    end
    find_pulses!(prop_cache, prop_generator.final, parameters)
    return prop_cache
end

"""
    find_pulses!(pulse_cache, props, parameters)

Add all of the pulse sequence elements specified by a SpecifiedPropagators
object 'props' to the 'pulse_cache'.
"""
function find_pulses!(prop_cache::SimCache{T,N,A}, props::SpecifiedPropagators{T,N,A},
        parameters::SimulationParameters{M,T,A}) where {M,T,N,A}

    for index in eachindex(props)
        if isassigned(props.elements, index)
            element, start = props.elements[index]
            find_pulses!(prop_cache, element, start, parameters)
        end
    end
    return prop_cache
end

"""
    find_pulses!(pulse_cache, block, start, parameters)

Add a 'block' with a particular 'start' step and all of its constituent pulses
to the 'pulse_cache'.
"""
function find_pulses!(prop_cache::SimCache{T,N,A}, block::Block{T,N}, start::Int, parameters::SimulationParameters{M,T,A}) where {M,T,N,A}
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

"""
    find_pulses!(pulse_cache, pulse, start, parameters)

Add a 'pulse' with a particular 'start' step to the 'pulse_cache'.
"""
function find_pulses!(prop_cache::SimCache{T,N,A}, pulse::Pulse{T,N}, start::Int, parameters::SimulationParameters{M,T,A}) where {M,T,N,A}
    period_steps = parameters.period_steps
    step_size = parameters.step_size
    pulse_cache = prop_cache.pulses

    steps = Int(pulse.t/step_size)
    rf = pulse.γB1
    add_rf!(pulse_cache, rf)

    timing = (mod1(start, period_steps), steps)
    add_timing!(pulse_cache[rf], timing)

    push!(pulse_cache[rf].timings[timing].phases, pulse.phase)
    return steps
end

"""
    allocate_propagators!(block_cache, parameters)

Allocate a propagator for each block in the 'block_cache'.
"""
function allocate_propagators!(block_cache::BlockCache{T,N,A}, parameters::SimulationParameters{M,T,A}) where {M,T,N,A}
    for collection in block_cache.ranks
        for key in keys(collection.steps)
            collection.propagators[key] = similar(parameters.temps[1])
        end
    end
end

"""
    allocate_propagators!(pulse_cache, parameters)

Allocate a propagator for each pulse and combination in the 'pulse_cache'.
"""
function allocate_propagators!(pulse_cache::PulseCache{T,N,A}, parameters::SimulationParameters{M,T,A}) where {M,T,N,A}
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

"""
    allocate_combinations!(rf_cache, parameters)

Allocate each of the combination propagators for each unique timing in the
'rf_cache'.
"""
function allocate_combinations!(rf_cache::PropagatorCollectionRF{T,N,A}, parameters::SimulationParameters{M,T,A}) where {M,T,N,A}
    nγ = parameters.nγ
    γ_steps = parameters.γ_steps

    unique_timings = Set{Tuple{Int,Int}}()
    for timing in keys(rf_cache.timings)
        push!(unique_timings, combination_timing(timing, γ_steps))
    end
    for timing in unique_timings
        num_combinations = timing[2] == 0 ? nγ : 2*nγ
        rf_cache.combinations[timing] = [similar(parameters.temps[1]) for n=1:num_combinations]
    end
end

"""
    combination_timing(timing, γ_steps)

Get the key to specify a set of combination propagators for a particular pulse
'timing'.
"""
combination_timing(timing, γ_steps) = (mod1(timing[1], γ_steps), mod(timing[2], γ_steps))

"""
    build_combined_propagators!(prop_cache, Hinternal, parameters)

Caclulate the actual propagators for each of the combination propagators in the
'prop_cache'. This is accomplished by first rotating the internal spin
hamiltonian 'Hinternal' to the angle corresponding to each step in the rotor
period. For each rf power propagators for each step are calculated from these
rotated hamiltonians and combined to form the combination propagators. The
results are stored in the 'prop_cache'.
"""
function build_combined_propagators!(prop_cache::SimCache{T,N,A}, Hinternal::SphericalTensor{Hamiltonian{A}},
        parameters::SimulationParameters{M,T,A}) where {M,T,N,A}

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

"""
    step_propagators!(propagators, rf, Hrotated, parameters, temps)

Generate a propagator with a given 'rf' power for each step in a rotor period
from a set of rotated hamiltonians ('Hrotated'). 'temps' holds temporary memory
used for exponentiating the hamiltonians. The results are stored in
'propagators'.
"""
function step_propagators!(propagators::Vector{Propagator{A}}, rf::NTuple{N,T}, Hrotated::Vector{Hamiltonian{A}},
        parameters::SimulationParameters{M,T,A}, temps::Vector{<:Hamiltonian{Ar}}) where {M,T,N,A,Ar<:AbstractArray{T}}

    wrapper = array_wrapper_type(A)
    wrapper == array_wrapper_type(Ar) || throw(TypeError("Temporary arrays have incorrect array type"))

    period_steps = parameters.period_steps
    step_size = parameters.step_size
    xyz = parameters.xyz

    Hrf = wrapper(pulse_H(rf,xyz))
    H = similar(temps[1])
    for n = 1:period_steps
        H = real_add!(H, Hrotated[n], Hrf)
        U = expm_cheby!(propagators[n], H, step_size/10^6, temps)
    end
    return propagators
end

"""
    expm_cheby!(U, H, dt, temps)

Modify 'U' to hold a propagator generated from a Hamiltonian ('H') and a time
interval ('dt') using a Chebyshev expansion to calculate the matrix exponential.
The contents of 'temps' and 'H' will be modified.
"""
function expm_cheby!(U::Propagator{Ac}, H::Hamiltonian{Ar}, dt::Real, temps::Vector{Hamiltonian{Ar}}) where {T,N,Ac<:AbstractArray{Complex{T},N},Ar<:AbstractArray{T,N}}
    array_wrapper_type(Ac) == array_wrapper_type(Ar) || throw(TypeError("Array types must match"))

    nmax = 25
    thresh = T(1E-10)
    bound = eig_max_bound(H.data)
    x = scaledn!(H,bound) # scale H so its eigenvalues are <1 so the Chebyshev expansion converges
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
        if threshold(t1.data, T(thresh/abs(j))) # stop if the new expansion term is below a threshold
            break
        end
        t1, t2 = t2, t1
    end
    return U
end

"""
    threshold(A, thresh)

Return true if the absolute value of all elements in 'A' are less than 'thresh'.
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
theorem. Return the bound.
"""
function eig_max_bound(A::AbstractArray{T}) where T
    R = real(T)
    x, y = size(A)
    out = zero(R)
    @inbounds for j = 1:y
        current = zero(R)
        for k = 1:x
            current += abs(A[k,j])
        end
        out = max(out, current)
    end
    return out
end

"""
    combine_propagators!(rf_cache, step_propagators, parameters)

Generate the combined propagators for each of the timings for a given rf power
from the propagators for each step in the rotor period ('step_propagators').
The results are stored in the 'rf_cache'.
"""
function combine_propagators!(rf_cache::PropagatorCollectionRF{T,N,A}, step_propagators::Vector{Propagator{A}},
        parameters::SimulationParameters{M,T,A}) where {M,T,N,A}

    γ_steps = parameters.γ_steps

    for (combination, combinations) in rf_cache.combinations
        if combination[2] == 0
            combine_propagators_unsplit!(combinations, step_propagators, combination, parameters)
        else
            combine_propagators_split!(combinations, step_propagators, combination, parameters)
        end
    end
    return rf_cache
end

"""
    combine_propagators_unsplit!(combinations, step_propagators, combination, parameters)

Generate the combined propagators for a timing that has a length divisible by
the number of steps per γ angle. A single combination is generated per γ angele
from the propagators for each step in the rotor period ('step_propagators'). The
results are stored in 'combinations'.
"""
function combine_propagators_unsplit!(combinations::Vector{Propagator{A}}, step_propagators::Vector{Propagator{A}},
        combination::NTuple{2,Int}, parameters::SimulationParameters{M,T,A}) where {M,T,N,A}

    nγ = parameters.nγ
    γ_steps = parameters.γ_steps
    temp = parameters.temps[1]

    for n = 1:nγ
        U = combinations[n]
        U, temp = build_combination!(U, temp, (n-1)*γ_steps+combination[1], γ_steps, step_propagators)
        combinations[n] = U
    end
    parameters.temps[1] = temp
end

"""
    combine_propagators_split!(combinations, step_propagators, combination, parameters)

Generate the combined propagators for a timing that has a length not divisible
by the number of steps per γ angle. The steps for each γ angle are split between
two combination propagators. This split ensures that a propagator γ_steps long
starting from both the beginning or ending of a pulse (used by
γiterate_propagator!) can easily be generated from two of the combinations.

The combinations are generated from the propagators for each step in the rotor
period ('step_propagators') and the results are stored in 'combinations'.
"""
function combine_propagators_split!(combinations::Vector{Propagator{A}}, step_propagators::Vector{Propagator{A}},
        combination::NTuple{2,Int}, parameters::SimulationParameters{M,T,A}) where {M,T,N,A}

    nγ = parameters.nγ
    γ_steps = parameters.γ_steps
    temp = parameters.temps[1]

    counts = (combination[2], γ_steps-combination[2])

    for n = 1:nγ
        position1 = (n-1)*γ_steps+combination[1]
        U1 = combinations[2*n-1]
        U1, temp = build_combination!(U1, temp, position1, counts[1], step_propagators)
        combinations[2*n-1] = U1

        position2 = position1+combination[2]
        U2 = combinations[2*n]
        U2, temp = build_combination!(U2, temp, position2, counts[2], step_propagators)
        combinations[2*n] = U2
    end
    parameters.temps[1] = temp
end

"""
    build_combination!(U, temp, start, count, step_propagators)

Combine a number of propagators specified by 'count' taken from
'step_propagators' starting at position 'start'. 'U' and 'temp' are modified and
returned. They may switch names so it is important to actually use the returned
propagators.
"""
function build_combination!(U::Propagator{A}, temp::Propagator{A}, start::Int, count::Int,
        step_propagators::Vector{Propagator{A}}) where {A}

    count > 0 || throw(ArgumentError("Combination cannot have 0 steps"))

    period_steps = length(step_propagators)

    start_ind = mod1(start, period_steps)
    copyto!(U, step_propagators[start_ind])
    for n = start+1:start+count-1
        position = mod1(n, period_steps)
        mul!(temp, step_propagators[position], U)
        U, temp = temp, U
    end
    return U, temp
end

"""
    build_pulse_props!(pulse_cache, parameters)

Calculate the propagators for each pulse in 'pulse_cache'. First the unphased
propagators are generated from the combination propagators, and then the phased
propagators are generated by rotating the unphased propagators. The results are
stored in the 'pulse_cache'.
"""
function build_pulse_props!(pulse_cache::PulseCache{T,N,A}, parameters::SimulationParameters{M,T,A}) where {M,T,N,A}
    for rf in values(pulse_cache)
        for (timing, timing_cache) in rf.timings
            timing_cache.unphased[1] = build_propagator!(timing_cache.unphased[1], rf, timing, parameters)
        end
    end
    generate_phased_propagators!(pulse_cache, parameters)
    return pulse_cache
end

"""
    build_propagator!(U, rf_cache, timing, parameters)

Calculate the unphased propagator for a particular 'timing' by combining
combination propagators stored in 'rf_cache' using 'U' for storage. Return the
propagator.
"""
function build_propagator!(U::Propagator{A}, rf_cache::PropagatorCollectionRF{T,N,A}, timing::NTuple{2,Int},
        parameters::SimulationParameters{M,T,A}) where {M,T,N,A}

    nγ = parameters.nγ
    γ_steps = parameters.γ_steps
    temp = parameters.temps[1]

    combinations = rf_cache.combinations[combination_timing(timing, γ_steps)]

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

"""
    generate_phased_propagators!(pulse_cache, parameters)

Calculate all of the properly phased propagators in 'pulse_cache' by rotating
the unphased propagators around Z. The results are stored in the 'pulse_cache'.
"""
function generate_phased_propagators!(pulse_cache::PulseCache{T,N,A}, parameters::SimulationParameters{M,T,A}) where {M,T,N,A}
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

"""
    build_block_props!(prop_cache, parameters)

Calculate the propagators for each block in 'prop_cache' using the pulse
propagators stored therein. This process proceeds in order of block rank,
because the propagators for higher rank blocks depend on those of lower rank
blocks. The results are stored in the 'prop_cache'.
"""
function build_block_props!(prop_cache::SimCache{T,N,A}, parameters::SimulationParameters{M,T,A}) where {M,T,N,A}
    for collection in prop_cache.blocks.ranks
        for key in keys(collection.steps)
            build_block_prop!(prop_cache, key, parameters)
        end
    end
    return prop_cache
end

"""
    build_block_prop!(prop_cache, key, parameters)

Calculate the propagator for the block specified by key using the propagators in
'prop_cache' and store the result in 'prop_cache'.
"""
function build_block_prop!(prop_cache::SimCache{T,N,A}, key::Tuple{Block{T,N}, Int},
        parameters::SimulationParameters{M,T,A}) where {M,T,N,A}

    U, _ = prop_cache.blocks[key]
    block, start = key
    if block.repeats == 1
        U = build_nonrepeat_block!(U, block, start, prop_cache, parameters)
    else
        U = build_repeat_block!(U, block, start, prop_cache, parameters)
    end
    prop_cache.blocks[key] = U
    return prop_cache
end

"""
    build_nonrepeat_block!(U, block, start, prop_cache, parameters)

Build a propagator for a 'block' with only 1 repeat and a particular 'start'
step using 'U' for storage. The propagator is caclulated by combining
propagators for the elements in the 'block' retreived from 'prop_cache'. Return
the propagator.
"""
function build_nonrepeat_block!(U::Propagator{A}, block::Block{T,N}, start::Int, prop_cache::SimCache{T,N,A},
        parameters::SimulationParameters{M,T,A}) where {M,T,N,A}

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

"""
    build_nonrepeat_block!(U, block, start, prop_cache, parameters)

Build a propagator for a 'block' with multiple repeats and a particular 'start'
step using 'U' for storage. The propagator is caclulated by combining
propagators for single repeats of the 'block' retreived from 'prop_cache'.
Return the propagator.
"""
function build_repeat_block!(U::Propagator{A}, block::Block{T,N}, start::Int, prop_cache::SimCache{T,N,A},
        parameters::SimulationParameters{M,T,A}) where {M,T,N,A}

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

"""
    fetch_propagator(pulse, start, prop_cache, parameters)

Fetch the propagator corresponding to a given 'pulse' and 'start' step from
'prop_cache'. Return the propagator and the number of steps used.
"""
function fetch_propagator(pulse::Pulse{T,N}, start::Int, prop_cache::SimCache{T,N,A},
        parameters::SimulationParameters{M,T,A}) where {M,T,N,A}

    period_steps = parameters.period_steps
    step_size = parameters.step_size

    steps = Int(pulse.t/step_size)
    timing = (mod1(start, period_steps), steps)
    propagator = prop_cache.pulses[pulse.γB1].timings[timing].phased[pulse.phase]
    return propagator, steps
end

"""
    fetch_propagator(block, start, prop_cache, parameters)

Fetch the propagator corresponding to a given 'block' and 'start' step from
'prop_cache'. Return the propagator and the number of steps used.
"""
function fetch_propagator(block::Block{T,N}, start::Int, prop_cache::SimCache{T,N,A},
        parameters::SimulationParameters{M,T,A}) where {M,T,N,A}

    period_steps = parameters.period_steps

    start_period = mod1(start, period_steps)
    U, steps = prop_cache.blocks[(block,start_period)]
    return U, steps
end

"""
    γiterate_pulse_propagators!(pulse_cache, parameters, γ_iteration)

Transform all of the propagators in the 'pulse_cache' to be correct for the next
γ angle. 'γ_iteration' specifies which γ angle is next.
"""
function γiterate_pulse_propagators!(pulse_cache::PulseCache{T,N,A}, parameters::SimulationParameters{M,T,A},
        γ_iteration::Int) where {M,T,N,A}

    γ_steps = parameters.γ_steps

    for rf in values(pulse_cache)
        for timing_pair in rf.timings
            timing, timing_collection = timing_pair
            if timing[2] <= 2*γ_steps # short pulses are cheaper to generate from combinations than to γiterate
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

"""
    γiterate_pulse_propagators!(U, rf, timing, parameters, γ_iteration)

Transform the propagator 'U' in 'rf' that has the specified 'timing' to be
correct for the next γ angle. The effective start step of 'U' is shifted forward
by multiplying it with a propagator γ_steps long starting at the same time. The
effective end step of 'U' is shifted by multiplying a propagator γ_steps long
starting after its end with it. 'γ_iteration' specifies which γ angle is next.
"""
function γiterate_propagator!(U::Propagator{A}, rf::PropagatorCollectionRF{T,N,A}, timing::NTuple{2,Int},
        parameters::SimulationParameters{M,T,A}, γ_iteration::Int) where {M,T,N,A}

    γ_steps = parameters.γ_steps
    nγ = parameters.nγ
    temp = parameters.temps[1]

    combinations = rf.combinations[combination_timing(timing, γ_steps)]

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
