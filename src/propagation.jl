"""
    propagate(pulse, start, pulse_cache, parameters, temp)

Fetch the propagator corrseponding to a given 'pulse' and 'start' step from
'pulse_cache'. Return the propagator and the number of steps used.
"""
function propagate(pulse::Pulse, start, pulse_cache, parameters, temp)
    period_steps = parameters.period_steps
    step_size = parameters.step_size
    xyzs = parameters.xyz

    steps = Int(pulse.t/step_size)
    timing = (mod1(start, period_steps), steps)
    propagator = pulse_cache[pulse.γB1].timings[timing].phased[pulse.phase]
    return propagator, steps, temp
end

"""
    propagate(block, start, pulse_cache, parameters, temp)

Generate a propagator for the entire 'block' including its repeats using the
propagators in 'pulse_cache'. Return the propagator and the number of steps used.
"""
function propagate(block::Block, start,pulse_cache, parameters, temp)
    step_total = 0
    Uelement,steps,temp = propagate(block.pulses[1], start+step_total, pulse_cache, parameters, temp)
    U = copy(Uelement)
    step_total += steps
    for j = 1:block.repeats
        iter = j == 1 ? 2 : 1
        for n = iter:length(block.pulses)
            Uelement, steps, temp = propagate(block.pulses[n], start+step_total, pulse_cache, parameters, temp)
            mul!(temp, Uelement,U)
            step_total += steps
            U, temp = temp, U
        end
    end
    return U, step_total, temp
end

"""
    propagate_before_loop(sequence,loop,start_step,pulse_cache,parameters,temp)

Generate a propagator for a group of non-looped pulse sequence elements in
'sequence' prior to the looped element specified by 'loop'. The set of includede
lements starts just after the previous looped element or at the beginning of the
'sequence' if there is none. It ends just before the specified looped element or
at the end of the 'sequence' if 'loop' exceeds the number of looped elements in
'sequence'.
"""
function propagate_before_loop(sequence, loop, start_step, pulse_cache, parameters, temp)
    U = similar(temp)
    fill_diag!(U, 1)
    nonloop_steps = 0

    # select the range of sequence elements to combine into this propagator
    # for the first loop start at the beginning of the sequence otherwise just after the previous loop
    # if out of loops end at the end of the sequence otherwise just before the current loop
    start_iter = (loop == 1) ? 1 : sequence.detection_loop[loop-1]+1
    end_iter = loop>length(sequence.detection_loop) ? length(sequence.pulses) : sequence.detection_loop[loop]-1

    for n = start_iter:end_iter
        U1, steps, temp = propagate(sequence.pulses[n], nonloop_steps+start_step, pulse_cache, parameters, temp)
        mul!(temp, U1, U)
        U, temp = temp, U
        nonloop_steps += steps
    end

    return U, nonloop_steps, temp
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
different starting steps. The PropagationChunk initially contains the first
propagator that will be needed for each starting step in 'current'. For each
starting step it also contains a collection of propagators in 'incrementors'
that can be multiplied sequentially onto these initial propagators to generate
the propagators for subsequent occurrences of the same starting step.

The precise behaviour depends on the PropagationType. FirstLooped is for the
first chunk of a sequence and therefore will have only one starting step.
NonLooped is for a final chunk of the sequence which comes after all the looped
elements. It can have multiple starting steps, but does not need any
incrementors. Looped is for all other chunks and can have both multiple starting
steps and incrementors.
"""
struct PropagationChunk{L<:PropagationType,T<:Propagator}
    current::Vector{T}
    incrementors::Vector{Vector{T}}

    PropagationChunk{L}(current::Vector{T}, incrementors::Vector{Vector{T}}) where {T<:Propagator,L<:PropagationType} =
        new{L,T}(current, incrementors)
end

struct PropagationGenerator{T<:Propagator}
    first::PropagationChunk{FirstLooped,T}
    loops::Vector{PropagationChunk{Looped,T}}
    nonloop::PropagationChunk{NonLooped,T}
end

function build_looped(sequence::Sequence, loop, old_loop_steps, old_nonloop_steps, pulse_cache, parameters, temp::A) where {A}
    period_steps = parameters.period_steps
    step_size = parameters.step_size

    old_loop_cycle = (old_loop_steps == 0) ? 1 : div(lcm(old_loop_steps, period_steps), old_loop_steps)
    Unonlooped = Vector{A}()
    nonloop_steps = 0
    for n = 1:old_loop_cycle
        start = old_nonloop_steps+(n-1)*old_loop_steps
        U, nonloop_steps, temp = propagate_before_loop(sequence, loop, start, pulse_cache, parameters, temp)
        push!(Unonlooped, U)
    end
    loop_element = sequence.pulses[sequence.detection_loop[loop]]
    loop_steps = Int(duration(loop_element)/step_size)
    loop_cycle = div(lcm(loop_steps, period_steps), loop_steps)

    incrementor_cycle = div(lcm(loop_steps, old_loop_steps), loop_steps)

    Uloops = Array{A,2}(undef, old_loop_cycle, loop_cycle)
    for n = 1:old_loop_cycle
        for j = 1:loop_cycle
            start = old_nonloop_steps+nonloop_steps+(n-1)*old_loop_steps+(j-1)*loop_steps
            Uloop,_,temp = propagate(loop_element, start+(n-1)*loop_steps, pulse_cache, parameters, temp)
            Uloops[n,j] = Uloop
        end
    end
    Ustart = Vector{A}()
    Uincrementors = Vector{Vector{A}}()
    for n = 1:old_loop_cycle
        U = Unonlooped[n]
        for j = 1:n-1
            mul!(temp, Uloops[n, j], U)
            U, temp = temp, U
        end
        push!(Ustart, U)

        Uincs = Vector{A}()
        for j = 1:incrementor_cycle
            increment_start = n+(j-1)*old_loop_cycle
            Uinc = copy(Uloops[n, mod1(increment_start, loop_cycle)])
            for k = increment_start+1:increment_start+old_loop_cycle-1
                a = mod1(k, loop_cycle)
                mul!(temp, Uloops[n, a], Uinc)
                Uinc, temp = temp, Uinc
            end
            push!(Uincs, Uinc)
        end
        push!(Uincrementors, Uincs)
    end

    chunk = PropagationChunk{Looped}(Ustart, Uincrementors)

    return chunk, old_loop_steps+loop_steps, old_nonloop_steps+nonloop_steps, temp
end

function build_first_looped(sequence::Sequence, pulse_cache, parameters, temp::A) where {A}
    period_steps = parameters.period_steps
    step_size = parameters.step_size

    U, nonloop_steps, temp=propagate_before_loop(sequence, 1, 1, pulse_cache, parameters, temp)
    nonloop_steps += 1
    initial = [U]

    loop_element = sequence.pulses[sequence.detection_loop[1]]
    loop_steps = Int(duration(loop_element)/step_size)
    loop_cycle = div(lcm(loop_steps, period_steps), loop_steps)
    if loop_cycle >= sequence.repeats
        loop_cycle = sequence.repeats-1
    end

    Uloops = Vector{A}()
    for n = 1:loop_cycle
        Uloop, _, temp = propagate(loop_element, nonloop_steps+(n-1)*loop_steps, pulse_cache, parameters, temp)
        push!(Uloops, Uloop)
    end

    return PropagationChunk{FirstLooped}(initial, [Uloops]), loop_steps, nonloop_steps, temp
end

function build_nonlooped(sequence::Sequence, loop, old_loop_steps, old_nonloop_steps, pulse_cache, parameters, temp::A) where {A}
    period_steps = parameters.period_steps

    old_loop_cycle = (old_loop_steps == 0) ? 1 : div(lcm(old_loop_steps, period_steps), old_loop_steps)
    Unonlooped = Vector{A}()
    nonloop_steps = 0
    for n = 1:old_loop_cycle
        start = old_nonloop_steps+(n-1)*old_loop_steps
        U, nonloop_steps, temp = propagate_before_loop(sequence, loop, start, pulse_cache, parameters, temp)
        push!(Unonlooped, U)
    end

    chunk = PropagationChunk{NonLooped}(Unonlooped, Vector{Vector{A}}())
    return chunk, old_loop_steps, old_nonloop_steps+nonloop_steps, temp
end


function next!(A::PropagationGenerator{T}, U::T, state, temp::T) where {T<:Propagator}
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
    inc_index = mod1(fld1(state, length(A.current))-1, length(A.incrementors[1]))
    if state > length(A.current)
        mul!(temp, A.incrementors[index][inc_index], A.current[index])
        A.current[index], temp = temp, A.current[index]
    end
    return A.current[index], temp
end

function next!(A::PropagationChunk{FirstLooped}, state, temp)
    index = mod1(state-1, length(A.incrementors[1]))
    if state > 1
        mul!(temp, A.incrementors[1][index], A.current[1])
        A.current[1], temp = temp, A.current[1]
    end
    return A.current[1], temp
end

function next!(A::PropagationChunk{NonLooped}, state, temp)
    index = mod1(state, length(A.current))
    return A.current[index], temp
end


function find_pulses!(pulse_cache, pulse::Pulse, start, parameters) where {T,N}
    period_steps = parameters.period_steps
    step_size = parameters.step_size

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

function find_pulses!(pulse_cache, block::Block, start, parameters)
    step_total = 0
    for j = 1:block.repeats
        for n = 1:length(block.pulses)
            step_total += find_pulses!(pulse_cache, block.pulses[n], start+step_total, parameters)
        end
    end
    return step_total
end

function find_pulses!(pulse_cache, sequence::Sequence, start, parameters)
    detection_loop = sequence.detection_loop
    issorted(detection_loop) || error("detection loop must be sorted")

    loop_steps, nonloop_steps = find_first_looped!(pulse_cache, sequence, parameters)
    for n = 2:length(detection_loop)
        loop_steps, nonloop_steps = find_looped!(pulse_cache, sequence, n, loop_steps, nonloop_steps, parameters)
    end

    if detection_loop[end] < length(sequence.pulses)
        loop_steps, nonloop_steps = find_nonlooped!(pulse_cache, sequence, n, loop_steps, nonloop_steps, parameters)
    end
    return pulse_cache
end

function find_before_loop!(pulse_cache, sequence, loop, start_step, parameters)
    nonloop_steps = 0

    # select the range of sequence elements to examine
    # for the first loop start at the beginning of the sequence otherwise just after the previous loop
    # if out of loops end at the end of the sequence otherwise just before the current loop
    start_iter = (loop == 1) ? 1 : sequence.detection_loop[loop-1]+1
    end_iter = loop > length(sequence.detection_loop) ? length(sequence.pulses) : sequence.detection_loop[loop]-1

    for n = start_iter:end_iter
        nonloop_steps += find_pulses!(pulse_cache, sequence.pulses[n], nonloop_steps+start_step, parameters)
    end

    return nonloop_steps
end

function find_first_looped!(pulse_cache, sequence, parameters)
    period_steps = parameters.period_steps
    step_size = parameters.step_size

    nonloop_steps = find_before_loop!(pulse_cache, sequence, 1, 1, parameters)
    nonloop_steps += 1

    loop_element = sequence.pulses[sequence.detection_loop[1]]
    loop_steps = Int(duration(loop_element)/step_size)
    loop_cycle = div(lcm(loop_steps,period_steps), loop_steps)
    if loop_cycle >= sequence.repeats
        loop_cycle = sequence.repeats-1
    end

    for n = 1:loop_cycle
        start = nonloop_steps+(n-1)*loop_steps
        find_pulses!(pulse_cache, loop_element, start, parameters)
    end

    return loop_steps, nonloop_steps
end

function find_looped!(pulse_cache, sequence, loop, old_loop_steps, old_nonloop_steps, parameters)
    period_steps = parameters.period_steps
    step_size = parameters.step_size

    old_loop_cycle = (old_loop_steps == 0) ? 1 : div(lcm(old_loop_steps, period_steps), old_loop_steps)
    nonloop_steps = 0
    for n = 1:old_loop_cycle
        start = old_nonloop_steps+(n-1)*old_loop_steps
        nonloop_steps = find_before_loop!(pulse_cache, sequence, loop, start, parameters)
    end
    loop_element = sequence.pulses[sequence.detection_loop[loop]]
    loop_steps = Int(duration(loop_element)/step_size)
    loop_cycle = div(lcm(loop_steps, period_steps), loop_steps)

    for n = 1:old_loop_cycle
        for j = 1:loop_cycle
            start = old_nonloop_steps+nonloop_steps+(n-1)*old_loop_steps+(j-1)*loop_steps
            find_pulses!(pulse_cache, loop_element, start+(n-1)*loop_steps, parameters)
        end
    end
    return old_loop_steps+loop_steps, old_nonloop_steps+nonloop_steps
end

function find_nonlooped!(pulse_cache, sequence, loop, old_loop_steps, old_nonloop_steps, parameters)
    period_steps = parameters.period_steps

    old_loop_cycle = (old_loop_steps == 0) ? 1 : div(lcm(old_loop_steps, period_steps), old_loop_steps)
    nonloop_steps = 0
    for n = 1:old_loop_cycle
        start = old_nonloop_steps+(n-1)*old_loop_steps
        nonloop_steps = find_before_loop!(pulse_cache, sequence, loop, start, parameters)
    end

    return old_loop_steps, old_nonloop_steps+nonloop_steps
end
