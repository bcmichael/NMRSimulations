"""
    fetch_propagator(pulse, start, prop_cache, parameters)

Fetch the propagator corresponding to a given 'pulse' and 'start' step from
'prop_cache'. Return the propagator and the number of steps used.
"""
function fetch_propagator(pulse::Pulse, start, prop_cache, parameters)
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
function fetch_propagator(block::Block, start, prop_cache, parameters)
    period_steps = parameters.period_steps

    start_period = mod1(start, period_steps)
    U, steps = prop_cache.blocks[(block,start_period)]
    return U, steps
end

abstract type PropagationType end
struct Looped<:PropagationType end
struct NonLooped<:PropagationType end

"""
    SeqElement

A SeqElement is a tuple containing either a Pulse or a Block and the step that
it starts at. This is all the information needed to generate the actual
propagator for this pulse sequence element.
"""
const SeqElement{T,N} = Tuple{Union{Pulse{T,N},Block{T,N}},Int} where {T,N}

"""
    PropagationChunk

A PropagationChunk corresponds to a chunk of a pulse sequence consisting of a
looped element and the non-looped elements that precede it. The presence of
looped elements in previous chunks means that the chunk can have multiple
different starting steps that occur cyclically over different iterations of the
pulse sequence detection loop.

The propagator corresponding to the chunk for each iteration can be generated
using the chunk propagator for the first instance of each starting step as well
as a series of propagators to increment them for subsequent instances of that
starting step. 'initial_elements' and 'incrementor_elements' hold SeqElement
tuples that correspond to these propagators, while 'current' and 'incrementors'
hold the actual propagators. 'current' will intially hold propagators matching
'initial_elements', but will be altered by multiplication with the incrementors
over the course of iteration through the detection loop.

The precise behaviour depends on the PropagationType. NonLooped is for a final
chunk of the sequence which comes after all the looped elements. It can have
multiple starting steps, but does not need any incrementors. Looped is for all
other chunks and can have both multiple starting steps and incrementors.
"""
struct PropagationChunk{L<:PropagationType,A<:AbstractArray,T<:AbstractFloat,N}
    initial_elements::Vector{SeqElement{T,N}}
    incrementor_elements::Array{SeqElement{T,N},2}
    current::Vector{Propagator{T,A}}
    incrementors::Array{Propagator{T,A},2}

    function PropagationChunk{L,A}(initial_elements::Vector{SeqElement{T,N}},
        incrementor_elements::Array{SeqElement{T,N},2}) where {L<:PropagationType,A<:AbstractArray,T<:AbstractFloat,N}

        current = Vector{Propagator{T,A}}(undef, length(initial_elements))
        incrementors = Array{Propagator{T,A},2}(undef, size(incrementor_elements))
        new{L,A,T,N}(initial_elements, incrementor_elements, current, incrementors)
    end

    PropagationChunk{L,A,T,N}() where {L,A,T,N} = new{L,A,T,N}(
        Vector{SeqElement{T,N}}(),
        Array{SeqElement{T,N},2}(undef,(0,0)),
        Vector{Propagator{T,A}}(),
        Array{Propagator{T,A},2}(undef,(0,0)))
end

struct PropagationGenerator{A<:AbstractArray,T<:AbstractFloat,N}
    loops::Vector{PropagationChunk{Looped,A,T,N}}
    nonloop::PropagationChunk{NonLooped,A,T,N}
end

function next!(A::PropagationGenerator, U, state, temps)
    Uchunk = next!(A.loops[1], state, temps)
    copyto!(U, Uchunk)
    for n = 2:length(A.loops)
        Uchunk = next!(A.loops[n], state, temps)
        mul!(temps[1], Uchunk, U)
        U, temps[1] = temps[1], U
    end
    if length(A.nonloop.current)>0
        Uchunk = next!(A.nonloop, state)
        mul!(temps[1], Uchunk, U)
        U, temps[1] = temps[1], U
    end
    return U, state+1
end

function next!(A::PropagationChunk{Looped}, state, temps)
    index = mod1(state, length(A.current))
    inc_index = mod1(fld1(state, length(A.current))-1, size(A.incrementors)[2])
    if state > length(A.current)
        mul!(temps[1], A.incrementors[index, inc_index], A.current[index])
        A.current[index], temps[1] = temps[1], A.current[index]
    end
    return A.current[index]
end

function next!(A::PropagationChunk{NonLooped}, state)
    index = mod1(state, length(A.current))
    return A.current[index]
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

    loop_steps = 0
    nonloop_steps = 1
    loop_chunks = Vector{PropagationChunk{Looped,A,T,N}}()
    for n = 1:length(detection_loop)
        chunk, loop_steps, nonloop_steps = build_looped(sequence, n, loop_steps, nonloop_steps, parameters, A)
        push!(loop_chunks, chunk)
    end

    if detection_loop[end] < length(sequence.pulses)
        last_chunk, loop_steps, nonloop_steps = build_nonlooped(sequence, length(detection_loop)+1, loop_steps,
            nonloop_steps, parameters, A)
    else
        last_chunk = PropagationChunk{NonLooped,A,T,N}()
    end
    prop_generator = PropagationGenerator(loop_chunks, last_chunk)
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

    if last_index >= first_index
        element = Block(sequence.pulses[first_index:last_index])
        nonloop_steps = Int(duration(element)/step_size)
        return element, nonloop_steps
    else
        return nothing, 0
    end
end

function build_looped(sequence::Sequence{T,N}, loop, loop_steps, nonloop_steps, parameters, ::Type{A}) where {T,N,A}
    period_steps = parameters.period_steps

    old_cycle = (loop_steps == 0) ? 1 : div(lcm(loop_steps, period_steps), loop_steps)

    initial, nonloop_steps = build_looped_initials(sequence, loop, loop_steps, nonloop_steps, old_cycle, parameters)
    incrementors, loop_steps = build_incrementors(sequence, loop, loop_steps, nonloop_steps, old_cycle, parameters)

    chunk = PropagationChunk{Looped,A}(initial, incrementors)
    return chunk, loop_steps, nonloop_steps
end

function build_looped_initials(sequence::Sequence{T,N}, loop, loop_steps, nonloop_steps, old_cycle, parameters) where {T,N}
    loop_element = sequence.pulses[sequence.detection_loop[loop]]
    nonloop_element, steps = build_before_loop!(sequence, loop, parameters)
    elements = Vector{SeqElement{T,N}}(undef, old_cycle)

    start = nonloop_steps
    if nonloop_element != nothing
        elements[1] = (nonloop_element, start)
    end

    for n = 2:old_cycle
        start += loop_steps
        loop_block = Block([loop_element], n-1)
        if nonloop_element == nothing
            elements[n] = (loop_block, start)
        else
            elements[n] = (Block([nonloop_element, loop_block]), start)
        end
    end

    return elements, steps+nonloop_steps
end

function build_incrementors(sequence::Sequence{T,N}, loop, loop_steps, nonloop_steps, old_cycle, parameters) where {T,N}
    period_steps = parameters.period_steps
    step_size = parameters.step_size

    loop_element = sequence.pulses[sequence.detection_loop[loop]]
    steps = Int(duration(loop_element)/step_size)
    loop_steps += steps

    incrementor_steps = steps*old_cycle
    incrementor_cycle = div(lcm(incrementor_steps, period_steps), steps)

    elements = Array{SeqElement{T,N},2}(undef, old_cycle, incrementor_cycle)
    incrementor_block = Block([loop_element], old_cycle)
    start = nonloop_steps
    for n = 1:old_cycle
        incrementor_start = start
        for j = 1:incrementor_cycle
            elements[n ,j] = (incrementor_block, incrementor_start)
            incrementor_start += incrementor_steps
        end
        start += loop_steps
    end
    return elements, loop_steps
end

function build_nonlooped(sequence::Sequence{T,N}, loop, loop_steps, nonloop_steps, parameters, ::Type{A}) where {T,N,A}
    period_steps = parameters.period_steps

    old_cycle = (loop_steps == 0) ? 1 : div(lcm(loop_steps, period_steps), loop_steps)
    initial_elements = Vector{SeqElement{T,N}}(undef, old_cycle)

    nonloop_element, steps = build_before_loop!(sequence, loop, parameters)
    start = nonloop_steps
    for n = 1:old_cycle
        initial_elements[n] = (nonloop_element, start)
        start += loop_steps
    end

    incrementor_elements = Array{SeqElement{T,N},2}(undef, 0, 0)
    chunk = PropagationChunk{NonLooped,A}(initial_elements, incrementor_elements)
    return chunk, loop_steps, nonloop_steps+steps
end

"""
    fill_generator!(prop_generator, prop_cache, parameters)

Fill in the Propagators in each PropagationChunk in 'prop_generator'.
"""
function fill_generator!(prop_generator, prop_cache, parameters)
    for chunk in prop_generator.loops
        fill_chunk_copy!(chunk.current, chunk.initial_elements, prop_cache, parameters)
        fill_chunk!(chunk.incrementors, chunk.incrementor_elements, prop_cache, parameters)
    end
    fill_chunk!(prop_generator.nonloop.current, prop_generator.nonloop.initial_elements, prop_cache, parameters)
    return prop_generator
end

function fill_chunk_copy!(props, elements, prop_cache, parameters)
    for index in eachindex(elements)
        if isassigned(elements, index)
            U, _ = fetch_propagator(elements[index][1], elements[index][2], prop_cache, parameters)
            copyto!(props[index], U)
        else
            fill_diag!(props[index], 1)
        end
    end
    return props
end

function fill_chunk!(props, elements, prop_cache, parameters)
    for index in eachindex(elements)
        if isassigned(elements, index)
            props[index],_ = fetch_propagator(elements[index][1], elements[index][2], prop_cache, parameters)
        else
            props[index] = fill_diag!(similar(parameters.temps[1]), 1)
        end
    end
    return props
end
