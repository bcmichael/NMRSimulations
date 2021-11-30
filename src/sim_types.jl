import Base: +, *, convert, hash, isequal
import LinearAlgebra: BLAS.BlasReal

struct SphericalTensor{T}
    s00::T
    s20::T
    s21::T
    s2m1::T
    s22::T
    s2m2::T
end

+(x::SphericalTensor{T1}, y::SphericalTensor{T2}) where {T1<:Number,T2<:Number} =
    SphericalTensor(x.s00+y.s00, x.s20+y.s20, x.s21+y.s21, x.s2m1+y.s2m1, x.s22+y.s22, x.s2m2+y.s2m2)

+(x::SphericalTensor{T1}, y::SphericalTensor{T2}) where {T1<:AbstractArray,T2<:AbstractArray} =
    SphericalTensor(x.s00+y.s00, x.s20+y.s20, x.s21+y.s21, x.s2m1+y.s2m1, x.s22+y.s22, x.s2m2+y.s2m2)

*(x::SphericalTensor{T}, y::AbstractArray) where {T<:Number} =
    SphericalTensor(x.s00*y, x.s20*y, x.s21*y, x.s2m1*y, x.s22*y, x.s2m2*y)

struct Spin{T<:AbstractFloat}
    channel::Int
    sigma_iso::T
    anisotropy::T
    asymmetry::T
    angles::EulerAngles{T}

    Spin{T}(channel, sigma_iso, anisotropy, asymmetry, α, β, γ) where T<:AbstractFloat =
        new{T}(channel, sigma_iso, anisotropy, asymmetry, EulerAngles{T}(α, β, γ))
    Spin{T}(channel, sigma_iso, anisotropy, asymmetry, angles::EulerAngles) where T<:AbstractFloat =
        new{T}(channel, sigma_iso, anisotropy, asymmetry, angles)
    Spin(channel, sigma_iso::T, anisotropy::T, asymmetry::T, α::T, β::T, γ::T) where T<:AbstractFloat =
        new{T}(channel, sigma_iso, anisotropy, asymmetry, EulerAngles{T}(α,β,γ))
    Spin(channel, sigma_iso::T, anisotropy::T, asymmetry::T, angles::EulerAngles{T}) where T<:AbstractFloat =
        new{T}(channel, sigma_iso, anisotropy, asymmetry, angles)
end

Spin(channel, sigma_iso, anisotropy, asymmetry, α, β, γ) =
    Spin{Float64}(channel, sigma_iso, anisotropy, asymmetry, EulerAngles(α, β, γ))

struct Pulse{T<:AbstractFloat,N}
    t::T
    γB1::NTuple{N,T}
    phase::NTuple{N,T}

    Pulse{T,N}(t, γB1, phase) where {T,N} = new{T,N}(t, γB1, phase)
    Pulse{T}(t, args...) where {T} = Pulse(T(t), T.(args)...)
    Pulse{T}(pulse::Pulse{T1,N}) where{T<:AbstractFloat,T1,N} = new{T,N}(pulse.t, pulse.γB1, pulse.phase)
end

function Pulse(t::T, args::T...) where {T<:AbstractFloat}
    N = Int(length(args)/2)
    Pulse{T,N}(t, args[1:2:end], args[2:2:end])
end

function Pulse(t, args...)
    N = Int(length(args)/2)
    Pulse{Float64,N}(t, args[1:2:end], args[2:2:end])
end

convert(::Type{Pulse{T,N}}, pulse::Pulse{T1,N}) where {T,T1,N} = Pulse{T,N}(pulse.t, pulse.γB1, pulse.phase)

"""
    Block

A Block represents a series of pulses that are executes sequentially in a pulse
sequence. A Block holds a vector of Pulses and Blocks ('pulses') to include in a
pulse sequence and a number of 'repeats' of this seqeunce to execute. The Block
will also hold a 'collapsed' form of the 'pulses'. This form is the same for all
equivalent Blocks no matter how they are specified.

A Block also has a rank to determine the order in which propagators for Blocks
are constructed. A Block that contains only Pulses and has 1 repeat is rank 1.
Otherwise a Block has a rank 1 higher than the highest rank Block in its
'pulses' if it has 1 repeat or 2 higher if it has mulitple repeats. Low rank
Blocks are constructed first ensuring that all of the Blocks needed to construct
higher rank Blocks will have already been constructed before they are needed.
"""
struct Block{T<:AbstractFloat,N}
    pulses::Vector{Union{Pulse{T,N},Block{T,N}}}
    repeats::Int
    collapsed::Vector{Pulse{T,N}}
    rank::Int

    function Block{T,N}(pulses::Vector{Union{Pulse{T,N},Block{T,N}}}, repeats=1) where {T<:AbstractFloat,N}
        collapsed, rank = collapse_block(pulses, repeats)
        if repeats > 1
            rank += 1
        end
        return new{T,N}(pulses, repeats, collapsed, rank+1)
    end

    function Block{T}(block::Block{T1,N}) where {T<:AbstractFloat,T1,N}
        pulses = convert_pulses(block.pulses, T, N)
        return Block{T,N}(pulses, block.repeats)
    end
end

function Block(pulses, repeats=1)
    if length(pulses) == 1 && repeats == 1 && pulses[1] isa Block
        return pulses[1]
    end
    T, N = partype(pulses[1])
    return Block{T,N}(Vector{Union{Pulse{T,N},Block{T,N}}}(pulses), repeats)
end

partype(x::Union{Pulse{T,N},Block{T,N}}) where {T,N} = T,N

duration(pulse::Pulse) = return pulse.t

function duration(block::Block{T}) where {T}
    total = zero(T)
    for n = 1:length(block.pulses)
        total += duration(block.pulses[n])
    end
    total *= block.repeats
    return total
end

# The propagator for a Block only depends on the contents and the start step
# hash/compare must be based on the contents instead of the object id
hash(block::Block) = hash(block.collapsed)
hash(block::Block, mix::UInt) = hash(block.collapsed, mix)
isequal(a::Block, b::Block) = isequal(a.collapsed, b.collapsed)

function collapse_block(pulses::Vector{Union{Pulse{T,N},Block{T,N}}}, repeats) where {T,N}
    as_pulses = Vector{Pulse{T,N}}()
    rank = 0
    for n in pulses
        if n isa Pulse
            push!(as_pulses, n)
        elseif n isa Block
            append!(as_pulses, n.collapsed)
            rank = max(rank, n.rank)
        end
    end
    as_pulses = repeat(as_pulses, repeats)
    out = Vector{Pulse{T,N}}()
    push!(out, as_pulses[1])
    for n = 2:length(as_pulses)
        if out[end].γB1 == as_pulses[n].γB1 && out[end].phase == as_pulses[n].phase
            pulse = pop!(out)
            push!(out, Pulse{T,N}(pulse.t+as_pulses[n].t, pulse.γB1, pulse.phase))
        else
            push!(out, as_pulses[n])
        end
    end
    return out, rank
end

struct Dimension
    elements::Vector{Int}
    size::Int

    function Dimension(elements, size)
        issorted(elements) || throw(ArgumentError("Dimension elements must be sorted"))
        new(elements, size)
    end
end

Dimension(a::Tuple{Vector{Int},Int}) = Dimension(a[1], a[2])

struct Sequence{T<:AbstractFloat,N,D}
    pulses::Vector{Union{Pulse{T,N},Block{T,N}}}
    dimensions::NTuple{D,Dimension}

    function Sequence{T,N}(pulses, dimensions) where {T<:AbstractFloat,N}
        D = length(dimensions)
        dimensions = NTuple{D,Dimension}(Dimension(n) for n in dimensions)
        check_dimensions(pulses, dimensions)
        new{T,N,D}(pulses, dimensions)
    end

    function Sequence(pulses, dimensions)
        T, N = partype(pulses[1])
        return Sequence{T,N}(pulses, dimensions)
    end

    function Sequence{T}(pulses, dimensions) where T<:AbstractFloat
        _, N = partype(pulses[1])
        if typeof(pulses) != Vector{Union{Pulse{T,N},Block{T,N}}}
            pulses = convert_pulses(pulses, T, N)
        end
        return Sequence{T,N}(pulses, dimensions)
    end

    function Sequence{T}(sequence::Sequence{T1,N,D}) where {T<:AbstractFloat,T1,N,D}
        pulses = convert_pulses(sequence.pulses, T, N)
        return new{T,N,D}(pulses, sequence.dimensions)
    end
end

function check_dimensions(pulses, dimensions)
    for n in 2:length(dimensions)
        dimensions[n].elements[1] > dimensions[n-1].elements[end] || throw(ArgumentError("All elements of dimension $n must be after those of dimension $(n-1)"))
    end
    dimensions[end].elements[end] <= length(pulses) || throw(ArgumentError("Dimension elements cannot exceed the number of elements"))
end

function convert_pulses(pulses, ::Type{T}, N) where {T}
    out = Vector{Union{Pulse{T,N},Block{T,N}}}()
    for n in pulses
        if isa(n,Pulse)
            push!(out, Pulse{T}(n))
        elseif isa(n, Block)
            push!(out, Block{T}(n))
        else
            throw(ArgumentError("Sequences may only contain Blocks and Pulses"))
        end
    end
    out
end

include("hilbertoperators.jl")

abstract type CalculationMode end
abstract type CPUMode <: CalculationMode end
abstract type GPUMode <: CalculationMode end
struct CPUSingleMode <: CPUMode end
struct CPUMultiProcess <: CPUMode end
struct GPUBatchedMode <: GPUMode end
struct GPUSingleMode <: GPUMode end

struct SimulationParameters{M<:CalculationMode,T<:BlasReal,A<:AbstractArray{Complex{T}}}
    period_steps::Int
    step_size::T
    nγ::Int
    γ_steps::Int
    angles::EulerAngles{T}
    xyz::Vector{Array{Complex{T},2}}
    temps::Vector{Propagator{A}}

    function SimulationParameters{M,T}(period_steps::Integer,step_size,nγ::Integer,spins) where {M<:CalculationMode,T<:AbstractFloat}
        γ_steps = Int(period_steps/nγ)
        angles = magic_angle(T)

        x = X(Array{Complex{T}})
        y = Y(Array{Complex{T}})
        z = Z(Array{Complex{T}})
        channels = check_spins(spins)
        xyz=channel_XYZ(spins, channels, x, y, z)

        if M == CPUSingleMode || M == CPUMultiProcess
            A = Array{Complex{T},2}
        elseif M == GPUBatchedMode
            A = CuArray{Complex{T},3,CUDA.Mem.DeviceBuffer}
        elseif M == GPUSingleMode
            A = CuArray{Complex{T},2,CUDA.Mem.DeviceBuffer}
        end
        temps = Vector{Propagator{A}}()

        new{M,T,A}(period_steps, step_size, nγ, γ_steps, angles, xyz, temps)
    end

    SimulationParameters{M}(period_steps, step_size ,nγ, spins) where M<:CalculationMode =
        SimulationParameters{M,Float64}(period_steps, step_size, nγ, spins)
end

SimulationParameters(period_steps, step_size, nγ, spins) =
    SimulationParameters{CPUSingleMode,Float64}(period_steps, step_size, nγ, spins)

function check_spins(spins)
    channels = [spin.channel for spin in spins]
    any(channels .< 1) && throw(ArgumentError("Channel indices cannot be less than 1"))
    maximum(channels) == length(unique(channels)) || throw(ArgumentError("Channel indices should not skip any numbers"))

    return maximum(channels)
end
