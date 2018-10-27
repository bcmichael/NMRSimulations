"""
    fetch_propagator(pulse, start, prop_cache, parameters, temp)

Fetch the propagator corresponding to a given 'pulse' and 'start' step from
'prop_cache'. Return the propagator and the number of steps used.
"""
function fetch_propagator(pulse::Pulse, start, prop_cache, parameters, temp)
    period_steps = parameters.period_steps
    step_size = parameters.step_size
    xyzs = parameters.xyz

    steps = Int(pulse.t/step_size)
    timing = (mod1(start, period_steps), steps)
    propagator = prop_cache.pulses[pulse.γB1].timings[timing].phased[pulse.phase]
    return propagator, steps, temp
end

"""
    fetch_propagator(block, start, prop_cache, parameters, temp)

Fetch the propagator corresponding to a given 'block' and 'start' step from
'prop_cache'. If the propagator is not in 'prop_cache', generate it and add it
to the cache. Return the propagator and the number of steps used.
"""
function fetch_propagator(block::Block, start, prop_cache, parameters, temp)
    period_steps = parameters.period_steps

    start_period = mod1(start, period_steps)
    if !((block, start_period) in keys(prop_cache.blocks))
        U, steps, temp = build_block_propagator(block, start_period, prop_cache, parameters, temp)
        prop_cache.blocks[(block, start_period)] = (U, steps)
    else
        U, steps = prop_cache.blocks[(block, start_period)]
    end

    return U, steps, temp
end

"""
    build_block_propagator(block, start, prop_cache, parameters, temp)

Generate a propagator for the entire 'block' including its repeats using the
propagators in 'prop_cache'. Return the propagator and the number of steps used.
"""
function build_block_propagator(block, start, prop_cache, parameters, temp)
    step_total = 0
    if block.repeats > 1
        # This reorganization ensures that each repeat gets treated as a block for caching purposes
        block = reorganize_repeated_block(block)
    end

    Uelement, steps, temp = fetch_propagator(block.pulses[1], start, prop_cache, parameters, temp)
    U = copy(Uelement)
    step_total += steps
    for element in block.pulses[2:end]
        Uelement, steps, temp = fetch_propagator(element, start+step_total, prop_cache, parameters, temp)
        mul!(temp, Uelement,U)
        step_total += steps
        U, temp = temp, U
    end
    return U, step_total, temp
end

"""
    reorganize_repeated_block(block)

Take a 'block' with repeats greater than 1 and reorganize it to have repeats=1.
The new block will contain several copies of the old block with repeats set to 1
as its pulses. This creates an equivalent block that is more caching friendly.
"""
function reorganize_repeated_block(block)
    block.repeats > 1 || throw(DomainError)
    single_repeat = Block(block.pulses, 1)
    pulses = [single_repeat for n=1:block.repeats]
    return Block(pulses,1)
end

abstract type PropagationType end
struct FirstLooped<:PropagationType end
struct Looped<:PropagationType end
struct NonLooped<:PropagationType end

"""
    PropagationChunk

A PropagationChunk corresponds to a chunk of a pulse sequence consisting of a
looped element and the non-looped elements that precede it. The presence of
looped elements in previous chunks means that the chunk can have multiple
different starting steps that occur cyclically over different iterations of the
pulse sequence detection loop.

The propagator corrseponding to the chunk for each iteration can be generated
using the chunk propagator for the first instance of each starting step as well
as a series of propagators to increment them for subsequent instances of that
starting step. 'initial_elements' and 'incrementor_elements' hold tuples of
pulse sequence elements and starting steps that correspond to these propagators,
while 'current' and 'incrementors' hold the actual propagators. 'current' will
intially hold propagators matching 'initial_elements', but will be altered
by multiplication with the incrementors over the course of iteration thorugh
the detection loop.

The precise behaviour depends on the PropagationType. FirstLooped is for the
first chunk of a sequence and therefore will have only one starting step.
NonLooped is for a final chunk of the sequence which comes after all the looped
elements. It can have multiple starting steps, but does not need any
incrementors. Looped is for all other chunks and can have both multiple starting
steps and incrementors.
"""
struct PropagationChunk{L<:PropagationType,A<:AbstractArray,T<:AbstractFloat,N}
    initial_elements::Vector{Tuple{Union{Pulse{T,N},Block{T,N}},Int}}
    incrementor_elements::Array{Tuple{Union{Pulse{T,N},Block{T,N}},Int},2}
    current::Vector{Propagator{T,A}}
    incrementors::Array{Propagator{T,A},2}

    function PropagationChunk{L,A}(initial_elements::Vector{Tuple{Union{Pulse{T,N},Block{T,N}},Int}},
        incrementor_elements::Array{Tuple{Union{Pulse{T,N},Block{T,N}},Int},2}) where {L<:PropagationType,A<:AbstractArray,T<:AbstractFloat,N}

        current = Vector{Propagator{T,A}}(undef, length(initial_elements))
        incrementors = Array{Propagator{T,A},2}(undef, size(incrementor_elements))
        new{L,A,T,N}(initial_elements, incrementor_elements, current, incrementors)
    end

    PropagationChunk{L,A,T,N}() where {L,A,T,N} = new{L,A,T,N}(
        Vector{Tuple{Union{Pulse{T,N},Block{T,N}},Int}}(),
        Array{Tuple{Union{Pulse{T,N},Block{T,N}},Int},2}(undef,(0,0)),
        Vector{Propagator{T,A}}(),
        Array{Propagator{T,A},2}(undef,(0,0)))
end

struct PropagationGenerator{A<:AbstractArray,T<:AbstractFloat,N}
    first::PropagationChunk{FirstLooped,A,T,N}
    loops::Vector{PropagationChunk{Looped,A,T,N}}
    nonloop::PropagationChunk{NonLooped,A,T,N}
end

function next!(A::PropagationGenerator, U, state, temp)
    Uchunk, temp = next!(A.first, state, temp)
    copyto!(U, Uchunk)
    for n = 1:length(A.loops)
        Uchunk, temp = next!(A.loops[n], state,temp)
        mul!(temp, Uchunk, U)
        U, temp = temp, U
    end
    if length(A.nonloop.current)>0
        Uchunk, temp = next!(A.nonloop, state,temp)
        mul!(temp, Uchunk, U)
        U, temp = temp, U
    end
    return U, state+1, temp
end

function next!(A::PropagationChunk{Looped}, state, temp)
    index = mod1(state, length(A.current))
    inc_index = mod1(fld1(state, length(A.current))-1, size(A.incrementors)[2])
    if state > length(A.current)
        mul!(temp, A.incrementors[index, inc_index], A.current[index])
        A.current[index], temp = temp, A.current[index]
    end
    return A.current[index], temp
end

function next!(A::PropagationChunk{FirstLooped}, state, temp)
    index = mod1(state-1, size(A.incrementors)[2])
    if state > 1
        mul!(temp, A.incrementors[1, index], A.current[1])
        A.current[1], temp = temp, A.current[1]
    end
    return A.current[1], temp
end

function next!(A::PropagationChunk{NonLooped}, state, temp)
    index = mod1(state, length(A.current))
    return A.current[index], temp
end

"""
    build_generator!(sequence, parameters, A)

Build a generator which can subsequently be used to generate the propagators for
each step of the detection loop of 'sequence'. 'A' is the type of Array to be
used in the propagators.
"""
function build_generator(sequence::Sequence{T,N}, parameters, ::Type{A}) where {T,N,A}
    detection_loop = sequence.detection_loop
    issorted(detection_loop) || error("detection loop must be sorted")

    first_chunk, loop_steps, nonloop_steps = build_first_looped(sequence, parameters, A)
    loop_chunks = Vector{PropagationChunk{Looped,A,T,N}}()
    for n = 2:length(detection_loop)
        chunk, loop_steps, nonloop_steps = build_looped(sequence, n, loop_steps, nonloop_steps, parameters, A)
        push!(loop_chunks, chunk)
    end

    if detection_loop[end] < length(sequence.pulses)
        last_chunk, loop_steps, nonloop_steps = build_nonlooped(sequence, n, loop_steps, nonloop_steps, parameters, A)
    else
        last_chunk = PropagationChunk{NonLooped,A,T,N}()
    end
    prop_generator = PropagationGenerator(first_chunk, loop_chunks, last_chunk)
    return prop_generator
end

"""
    build_before_loop(sequence, loop, parameters)

Build a Block containing the non-looped pulse sequence elements in 'sequence'
prior to the looped element specified by 'loop'. The set of included elements
starts just after the previous looped element or at the beginning of the
'sequence' if there is none. It ends just before the specified looped element or
at the end of the 'sequence' if 'loop' exceeds the number of looped elements in
'sequence'. Return the Block and the length of the block in steps. If there are
no such pulses return 'nothing' and '0'.
"""
function build_before_loop!(sequence, loop, parameters)
    step_size = parameters.step_size

    # select the range of sequence elements to examine
    # for the first loop start at the beginning of the sequence otherwise just after the previous loop
    # if out of loops end at the end of the sequence otherwise just before the current loop
    first_index = (loop == 1) ? 1 : sequence.detection_loop[loop-1]+1
    last_index = loop > length(sequence.detection_loop) ? length(sequence.pulses) : sequence.detection_loop[loop]-1

    if last_index > first_index
        element = Block(sequence.pulses[first_index:last_index])
        nonloop_steps = Int(duration(element)/step_size)
        return element, nonloop_steps
    else
        return nothing, 0
    end
end

function build_first_looped(sequence::Sequence{T,N}, parameters, ::Type{A}) where {T,N,A}
    period_steps = parameters.period_steps
    step_size = parameters.step_size

    initial_element = Vector{Tuple{Union{Pulse{T,N},Block{T,N}},Int}}(undef, 1)
    nonloop_element, nonloop_steps = build_before_loop!(sequence, 1, parameters)
    nonloop_steps += 1
    if nonloop_element != nothing
        initial_element[1] = (nonloop_element, 1)
    end

    loop_element = sequence.pulses[sequence.detection_loop[1]]
    loop_steps = Int(duration(loop_element)/step_size)
    loop_cycle = div(lcm(loop_steps,period_steps), loop_steps)
    if loop_cycle >= sequence.repeats
        loop_cycle = sequence.repeats-1
    end

    incrementor_elements = Array{Tuple{Union{Pulse{T,N},Block{T,N}},Int},2}(undef, 1, loop_cycle)
    for n = 1:loop_cycle
        start = nonloop_steps+(n-1)*loop_steps
        incrementor_elements[1, n] = (loop_element, start)
    end

    chunk = PropagationChunk{FirstLooped,A}(initial_element, incrementor_elements)
    return chunk, loop_steps, nonloop_steps
end

function build_looped(sequence::Sequence{T,N}, loop, old_loop_steps, old_nonloop_steps, parameters, ::Type{A}) where {T,N,A}
    period_steps = parameters.period_steps
    step_size = parameters.step_size

    old_loop_cycle = (old_loop_steps == 0) ? 1 : div(lcm(old_loop_steps, period_steps), old_loop_steps)
    loop_element = sequence.pulses[sequence.detection_loop[loop]]
    loop_steps = Int(duration(loop_element)/step_size)
    incrementor_cycle = div(lcm(loop_steps*old_loop_cycle, period_steps), loop_steps)

    initial_elements = Vector{Tuple{Union{Pulse{T,N},Block{T,N}},Int}}(undef, old_loop_cycle)
    incrementor_elements = Array{Tuple{Union{Pulse{T,N},Block{T,N}},Int},2}(undef, old_loop_cycle, incrementor_cycle)
    nonloop_steps = 0
    for n = 1:old_loop_cycle
        start = old_nonloop_steps+(n-1)*old_loop_steps
        nonloop_element, nonloop_steps = build_before_loop!(sequence, loop, parameters)
        if n == 1
            if nonloop_element != nothing
                initial_elements[1] = (nonloop_element, 1)
            end
        else
            loop_block = Block([loop_element], n-1)
            if nonloop_element == nothing
                initial_elements[n] = (loop_block, start)
            else
                initial_elements[n] = (Block([nonloop_element, loop_block]), start)
            end
        end

        incrementor_block = Block([loop_element], incrementor_cycle)
        for j = 1:incrementor_cycle
            incrementor_start = start+n*loop_steps
            incrementor_elements[n ,j] = (incrementor_block, incrementor_start)
        end
    end
    chunk = PropagationChunk{Looped,A}(initial_elements, incrementor_elements)
    return chunk, old_loop_steps+loop_steps, old_nonloop_steps+nonloop_steps
end

function build_nonlooped(sequence::Sequence{T,N}, loop, old_loop_steps, old_nonloop_steps, parameters, ::Type{A}) where {T,N,A}
    period_steps = parameters.period_steps

    old_loop_cycle = (old_loop_steps == 0) ? 1 : div(lcm(old_loop_steps, period_steps), old_loop_steps)
    nonloop_steps = 0
    initial_elements = Vector{Tuple{Union{Pulse{T,N},Block{T,N}},Int}}(undef, old_loop_cycle)

    for n = 1:old_loop_cycle
        start = old_nonloop_steps+(n-1)*old_loop_steps
        nonloop_element, nonloop_steps = build_before_loop!(sequence, loop, parameters)
        initial[n] = (loop_element, start)
    end

    incrementor_elements = Array{Tuple{Union{Pulse{T,N},Block{T,N}},Int},2}()
    chunk = PropagationChunk{NonLooped,A}(initial_elements, incrementor_elements)
    return chunk, old_loop_steps, old_nonloop_steps+nonloop_steps
end

"""
    fill_generator!(prop_generator, prop_cache, parameters, temp)

Fill in the Propagators in each PropagationChunk in 'prop_generator'.
"""
function fill_generator!(prop_generator, prop_cache, parameters, temp)
    _, temp = fill_chunk!(prop_generator.first, prop_cache, parameters, temp)
    for chunk in prop_generator.loops
        chunk, temp = fill_chunk!(chunk, prop_cache, parameters, temp)
    end
    _, temp = fill_chunk!(prop_generator.nonloop, prop_cache, parameters, temp)
    return prop_generator, temp
end

"""
    fill_chunk!(chunk, prop_cache, parameters, temp)

Fill in the current and incrementor fields of 'chunk' with Propagators from
'prop_cache' corresponding to the pulse sequence elements in the
initial_elements and incrementor_elements fields.
"""
function fill_chunk!(chunk, prop_cache, parameters, temp)
    _, temp = fill_array!(chunk.current, chunk.initial_elements, prop_cache, parameters, temp)
    _, temp = fill_array!(chunk.incrementors, chunk.incrementor_elements, prop_cache, parameters, temp)
    return chunk, temp
end

function fill_array!(props, elements, prop_cache, parameters, temp)
    for index in eachindex(elements)
        if isassigned(elements, index)
            props[index],_,temp = fetch_propagator(elements[index][1], elements[index][2], prop_cache, parameters, temp)
        else
            props[index] = fill_diag!(similar(temp), 1)
        end
    end
    return props, temp
end
