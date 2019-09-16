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
"""
struct PropagatorCollectionRF{T<:BlasReal,N,A<:AbstractArray{Complex{T}}}
    timings::Dict{NTuple{2,Int}, PropagatorCollectionTiming{T,N,A}}

    PropagatorCollectionRF{T,N,A}() where {T,N,A } = new(Dict{NTuple{2,Int}, PropagatorCollectionTiming{T,N,A}}())
end

"""
    add_pulse!(collection, timing, phases, parameters)

For each γ angle ensure that a PropagatorCollectionTiming for the given 'timing'
is present in the 'collection', allocating a new one if one does not already
exist. Add the 'phases' to each of the PropagatorCollectionTiming's.
"""
function add_pulse!(collection::PropagatorCollectionRF{T,N,A}, timing::NTuple{2,Int}, phases::NTuple{N,T},
        parameters::SimulationParameters{M,T,A}) where {M,T,N,A}

    nγ = parameters.nγ
    γ_steps = parameters.γ_steps
    period_steps = parameters.period_steps

    for n = 1:nγ
        start = mod1(timing[1]+n*γ_steps, period_steps)
        γ_timing = (start, timing[2])
        if ! haskey(collection.timings, γ_timing)
            collection.timings[γ_timing] = PropagatorCollectionTiming{T,N,A}()
        end
        push!(collection.timings[γ_timing].phases, phases)
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
    add_block!(block_cache, key, steps, parameters)

Add the block specified by 'key' to the 'block_cache'. The number of 'steps' in
the block is also stored in the cache.
"""
function add_block!(block_cache::BlockCache{T,N,A}, key::Tuple{Block{T,N}, Int}, steps::Int,
    parameters::SimulationParameters{M,T,A}) where {M,T,N,A}

    nγ = parameters.nγ
    γ_steps = parameters.γ_steps
    period_steps = parameters.period_steps

    rank = key[1].rank
    while rank > length(block_cache.ranks)
        push!(block_cache.ranks, PropagatorCollectionBlock{T,N,A}())
    end

    for n = 1:nγ
        start = mod1(key[2]+n*γ_steps, period_steps)
        γ_key = (key[1], start)
        block_cache.ranks[rank].steps[γ_key] = steps
    end
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
function find_pulses!(prop_cache::SimCache{T,N,A}, block::Block{T,N}, start::Int,
        parameters::SimulationParameters{M,T,A}) where {M,T,N,A}

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
    add_block!(block_cache, cache_key, step_total, parameters)
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
    add_pulse!(pulse_cache[rf], timing, pulse.phase, parameters)
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
        for (timing, timing_cache) in rf_cache.timings
            push!(timing_cache.unphased, similar(parameters.temps[1]))
            for phase in timing_cache.phases
                timing_cache.phased[phase] = similar(parameters.temps[1])
            end
        end
    end
end

"""
    build_propagators!(prop_cache, Hinternal, parameters)

Caclulate the actual propagators for each pulse and block in the 'prop_cache'.
This is accomplished by first rotating the internal spin hamiltonian 'Hinternal'
to the angle corresponding to each step in the rotor period. For each set of rf
powers propagators for each step are calculated from these rotated hamiltonians
and combined to form the combination propagators. The combination propagators
are used to build the pulse propagators which are in turn used to build the
Block propagators. The results are stored in the 'prop_cache'.
"""
function build_propagators!(prop_cache::SimCache{T,N,A}, Hinternal::SphericalTensor{Hamiltonian{A}},
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
        combine_propagators!(step_propagators, parameters)
        build_pulse_propagators!(rf_cache, step_propagators, parameters)
    end
    generate_phased_propagators!(prop_cache.pulses, parameters)
    build_block_props!(prop_cache, parameters)
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
    combine_propagators(step_propagators, parameters)

Create the combined propagators by sequentially multiplying the
'step_propagators'. The results are stored in the same vector as the
'step_propagators'.
"""
function combine_propagators!(step_propagators::Vector{Propagator{A}}, parameters::SimulationParameters{M,T,A}) where {M,T,A}
    temp = parameters.temps[1]

    for n = 2:length(step_propagators)
        mul!(temp, step_propagators[n], step_propagators[n-1])
        step_propagators[n], temp = temp, step_propagators[n]
    end
    parameters.temps[1] = temp
end

"""
    build_pulse_propagators!(rf_cache, combined_propagators, parameters)

Build a propagator for every timing in the 'rf_cache' from the
'combined_propagators'. The results are stored in the 'rf_cache'.
"""
function build_pulse_propagators!(rf_cache::PropagatorCollectionRF{T,N,A}, combined_propagators::Vector{Propagator{A}},
        parameters::SimulationParameters{M,T,A}) where {M,T,N,A}

    period_steps = parameters.period_steps

    for (timing, timing_cache) in rf_cache.timings
        timing_cache.unphased[1] = build_pulse_propagator!(timing_cache.unphased[1], combined_propagators, timing, parameters)
    end
end

"""
    build_pulse_propagator(U, combined_propagators, timing, parameters)

Calculate the unphased propagator for a particular 'timing' from the
'combined_propagators' using 'U' for storage. Return the propagator.
"""
function build_pulse_propagator!(U::Propagator{A}, combined_propagators::Vector{Propagator{A}}, timing::NTuple{2,Int},
        parameters::SimulationParameters{M,T,A}) where {M,T,A}

    period_steps = length(combined_propagators)
    temp = parameters.temps[1]

    start = timing[1]
    finish = timing[1]+timing[2]-1
    cycles, base = divrem(finish, period_steps)

    # Ustart,finish = U1,base*(U1,nγ)^cycles*U1,start-1†
    # if any of these propagators are I (cycles=0, start=1, base=0) skip that multiplication
    if cycles == 0
        copyto!(U, combined_propagators[base])
    else
        copyto!(U, combined_propagators[end])
        if cycles > 1
            U, temp = pow!(similar(U), U, cycles, temp)
        end
        if base != 0
            mul!(temp, combined_propagators[base], U)
            U, temp = temp, U
        end
    end

    if start != 1
        mul!(temp, U, combined_propagators[start-1], 'N', 'C')
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
