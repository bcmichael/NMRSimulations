import Base: length, iterate

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
"""
struct PropagationChunk{A<:AbstractArray,T<:AbstractFloat,N}
    initial_elements::Vector{SeqElement{T,N}}
    incrementor_elements::Array{SeqElement{T,N},2}
    current::Vector{Propagator{T,A}}
    incrementors::Array{Propagator{T,A},2}

    function PropagationChunk{A}(initial_elements::Vector{SeqElement{T,N}},
        incrementor_elements::Array{SeqElement{T,N},2}) where {A<:AbstractArray,T<:AbstractFloat,N}

        current = Vector{Propagator{T,A}}(undef, length(initial_elements))
        incrementors = Array{Propagator{T,A},2}(undef, size(incrementor_elements))
        new{A,T,N}(initial_elements, incrementor_elements, current, incrementors)
    end
end

struct PropagationFinal{A<:AbstractArray,T<:AbstractFloat,N}
    elements::Vector{SeqElement{T,N}}
    propagators::Vector{Propagator{T,A}}

    PropagationFinal{A}(elements::Vector{SeqElement{T,N}}) where {A,T,N} = new{A,T,N}(elements,
        Vector{Propagator{T,A}}(undef, length(elements)))

    PropagationFinal{A,T,N}() where {A,T,N} = new{A,T,N}(Vector{SeqElement{T,N}}(), Vector{Propagator{T,A}}())
end

struct PropagationDimension{A<:AbstractArray,T<:AbstractFloat,N}
    chunks::Array{PropagationChunk{A,T,N},2}
    propagators::Array{Propagator{T,A},2}
    start_cycle::Int
    cycle::Int

    function PropagationDimension(chunks::Array{PropagationChunk{A,T,N},2}, count, cycle) where {A,T,N}
        start_cycle = size(chunks,2)
        propagators = Array{Propagator{T,A},2}(undef, (count, start_cycle))
        new{A,T,N}(chunks, propagators, start_cycle, cycle)
    end
end

struct PropagationGenerator{A<:AbstractArray,T<:AbstractFloat,N,D}
    loops::NTuple{D,PropagationDimension{A,T,N}}
    final::PropagationFinal{A,T,N}
    size::NTuple{D,Int}
    temps::Vector{Propagator{T,A}} # will be the same as parameters.temps
end

length(G::PropagationGenerator) = reduce(*, G.size)

function iterate(G::PropagationGenerator{A,T,N,D}, state=1) where {A,T,N,D}
    state <= length(G) || return nothing

    U = pop!(G.temps)
    temp = pop!(G.temps)

    position = CartesianIndices(G.size)[state]
    copyto!(U, G.loops[1].propagators[position[1], 1])
    end_index = Rational(position[1], G.loops[1].cycle)
    for d = 2:D
        start_index = Int(mod1(end_index,1)*G.loops[d].start_cycle)
        Uchunk = G.loops[d].propagators[position[d], start_index]
        mul!(temp, Uchunk, U)
        U, temp = temp, U
        end_index += Rational(position[d], G.loops[d].cycle)
    end
    if length(G.final.propagators)>0
        start_index = Int(end_index%1*length(G.final.propagators))
        Uchunk = G.final.propagators[start_index]
        mul!(temp, Uchunk, U)
        U, temp = temp, U
    end
    push!(G.temps,temp)
    push!(G.temps,U)
    return (U, position), state+1
end

function next!(A::PropagationChunk, state, temps)
    index = mod1(state, length(A.current))
    inc_index = mod1(fld1(state, length(A.current))-1, size(A.incrementors)[2])
    if state > length(A.current)
        mul!(temps[1], A.incrementors[index, inc_index], A.current[index])
        A.current[index], temps[1] = temps[1], A.current[index]
    end
    return A.current[index]
end

"""
    build_generator!(sequence, parameters)

Build a generator which can subsequently be used to generate the propagators for
each step of the detection loop of 'sequence'.
"""
function build_generator(sequence::Sequence{T,N,D}, parameters::SimulationParameters{M,T,A}) where {M,T,N,A,D}
    period_steps = parameters.period_steps

    dim_loops = Vector{PropagationDimension{A,T,N}}()
    start_cycle = 1
    nonloop_steps = 1
    for d = 1:D
        dimension, nonloop_steps, start_cycle = build_dim(sequence, d, nonloop_steps, start_cycle, parameters, A)
        push!(dim_loops, dimension)
    end
    loop_steps = Int(period_steps/start_cycle)

    if sequence.dimensions[end].elements[end] < length(sequence.pulses)
        last_chunk = build_nonlooped(sequence, nonloop_steps, start_cycle, parameters, A)
    else
        last_chunk = PropagationFinal{A,T,N}()
    end
    prop_generator = PropagationGenerator(Tuple(dim_loops), last_chunk, Tuple(d.size for d in sequence.dimensions), parameters.temps)
    return prop_generator
end

function build_dim(sequence::Sequence{T,N}, dim, start_step, start_cycle, parameters, ::Type{A}) where {T,N,A}
    detection_loop = sequence.dimensions[dim].elements
    period_steps = parameters.period_steps

    chunks = Array{PropagationChunk{A,T,N},2}(undef, (length(detection_loop), start_cycle))
    cycle_steps = Int(period_steps/start_cycle)

    local loop_steps, nonloop_steps
    for i in 1:start_cycle
        loop_steps = 0
        nonloop_steps = start_step+(i-1)*cycle_steps
        for n = 1:length(detection_loop)
            chunk, loop_steps, nonloop_steps = build_looped(sequence, dim, n, loop_steps, nonloop_steps, parameters, A)
            chunks[n, i] = chunk
        end
    end
    cycle = div(lcm(loop_steps, period_steps), loop_steps)
    out = PropagationDimension(chunks, sequence.dimensions[dim].size, cycle)
    return out, nonloop_steps-(start_cycle-1)*cycle_steps, lcm(start_cycle, cycle)
end

"""
    build_before_loop(sequence, dim, loop, parameters)

Build a Block containing the non-looped pulse sequence elements in 'sequence'
prior to the looped element specified by 'loop' in the dimension specified by
'dim'. The set of included elements starts just after the previous looped
element or at the beginning of the 'sequence' if there is none. It ends just
before the specified looped element. Return the Block and the length of the
block in steps. If there are no such pulses return 'nothing' and '0'.
"""
function build_before_loop(sequence, dim, loop, parameters)
    step_size = parameters.step_size

    if dim == 1 && loop == 1 # beginning of sequence
        first_index = 1
    elseif dim > 1 && loop == 1 # last loop from previous dimension
        first_index = sequence.dimensions[dim-1].elements[end]+1
    elseif loop > 1 # previous loop from same dimension
        first_index = sequence.dimensions[dim].elements[loop-1]+1
    end
    last_index = sequence.dimensions[dim].elements[loop]-1

    if last_index >= first_index
        element = Block(sequence.pulses[first_index:last_index])
        nonloop_steps = Int(duration(element)/step_size)
        return element, nonloop_steps
    else
        return nothing, 0
    end
end

function build_looped(sequence::Sequence{T,N}, dim, loop, loop_steps, nonloop_steps, parameters, ::Type{A}) where {T,N,A}
    period_steps = parameters.period_steps

    old_cycle = (loop_steps == 0) ? 1 : div(lcm(loop_steps, period_steps), loop_steps)

    initial, nonloop_steps = build_looped_initials(sequence, dim, loop, loop_steps, nonloop_steps, old_cycle, parameters)
    incrementors, loop_steps = build_incrementors(sequence, dim, loop, loop_steps, nonloop_steps, old_cycle, parameters)

    chunk = PropagationChunk{A}(initial, incrementors)
    return chunk, loop_steps, nonloop_steps
end

function build_looped_initials(sequence::Sequence{T,N}, dim, loop, loop_steps, nonloop_steps, old_cycle, parameters) where {T,N}
    loop_element = sequence.pulses[sequence.dimensions[dim].elements[loop]]
    nonloop_element, steps = build_before_loop(sequence, dim, loop, parameters)
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

function build_incrementors(sequence::Sequence{T,N}, dim, loop, loop_steps, nonloop_steps, old_cycle, parameters) where {T,N}
    period_steps = parameters.period_steps
    step_size = parameters.step_size

    loop_element = sequence.pulses[sequence.dimensions[dim].elements[loop]]
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

function build_after_loops(sequence)
    first_index = sequence.dimensions[end].elements[end]+1
    last_index = length(sequence.pulses)

    if last_index >= first_index
        element = Block(sequence.pulses[first_index:last_index])
        nonloop_steps = Int(duration(element)/step_size)
        return element, nonloop_steps
    else
        return nothing, 0
    end
end

function build_nonlooped(sequence::Sequence{T,N}, nonloop_steps, start_cycle, parameters, ::Type{A}) where {T,N,A}
    period_steps = parameters.period_steps

    elements = Vector{SeqElement{T,N}}(undef, start_cycle)

    nonloop_element = build_after_loops(sequence)
    cycle_steps = Int(period_steps/start_cycle)
    start = nonloop_steps
    for n = 1:start_cycle
        elements[n] = (nonloop_element, start)
        start += cycle_steps
    end

    chunk = PropagationFinal{A}(elements)
    return chunk
end

"""
    fill_generator!(prop_generator, prop_cache, parameters)

Fill in the Propagators in 'prop_generator'.
"""
function fill_generator!(prop_generator, prop_cache, parameters)
    for dim in prop_generator.loops
        for chunk in dim.chunks
            fill_chunk_copy!(chunk.current, chunk.initial_elements, prop_cache, parameters)
            fill_chunk!(chunk.incrementors, chunk.incrementor_elements, prop_cache, parameters)
        end
        materialize_dimension(dim, parameters)
    end
    fill_chunk!(prop_generator.final.propagators, prop_generator.final.elements, prop_cache, parameters)
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

function materialize_dimension(dimension::PropagationDimension, parameters)
    temps = parameters.temps
    for start in 1:size(dimension.chunks, 2)
        for n = 1:size(dimension.propagators, 1)
            U = dimension.propagators[n, start]
            Uchunk = next!(dimension.chunks[1, start], n, temps)
            copyto!(U, Uchunk)
            for loop in 2:size(dimension.chunks, 1)
                Uchunk = next!(dimension.chunks[loop, start], n, temps)
                mul!(temps[1], Uchunk, U)
                U, temps[1] = temps[1], U
            end
            dimension.propagators[n, start] = U
        end
    end
end

function allocate_propagators!(prop_generator::PropagationGenerator, parameters)
    for dim in prop_generator.loops
        for chunk in dim.chunks
            for index in eachindex(chunk.current)
                chunk.current[index] = similar(parameters.temps[1])
            end
        end
        for index in eachindex(dim.propagators)
            dim.propagators[index] = similar(parameters.temps[1])
        end
    end
end
