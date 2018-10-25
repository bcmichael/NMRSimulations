import Base: +, *, convert, keys, haskey, copy, copyto!, similar

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

struct EulerAngles{T<:AbstractFloat}
    α::T
    β::T
    γ::T

    EulerAngles(α::T, β::T, γ::T) where T<:AbstractFloat = new{T}(α%360, β%360, γ%360)
    EulerAngles{T}(α, β, γ) where T<:AbstractFloat = new{T}(α%360, β%360, γ%360)
    EulerAngles{T}(angles::EulerAngles) where T = convert(EulerAngles{T}, angles)
end

EulerAngles(α ,β, γ) = EulerAngles{Float64}(α, β, γ)

convert(::Type{EulerAngles{T}}, angles::EulerAngles) where {T<:AbstractFloat} =
    EulerAngles(T(angles.α), T(angles.β), T(angles.γ))

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

struct Block{T<:AbstractFloat,N}
    pulses::Vector{Union{Pulse{T,N},Block{T,N}}}
    repeats::Int

    Block{T,N}(pulses::Vector{Union{Pulse{T,N},Block{T,N}}}, repeats=1) where {T<:AbstractFloat,N} =
        new{T,N}(pulses, repeats)

    function Block(pulses, repeats=1)
        T, N = partype(pulses[1])
        return new{T,N}(pulses, repeats)
    end

    function Block{T}(block::Block{T1,N}) where {T<:AbstractFloat,T1,N}
        pulses = Vector{Union{Pulse{T,N},Block{T,N}}}()
        for n in block.pulses
            if isa(n, Pulse)
                push!(pulses, Pulse{T}(n))
            elseif isa(n,Block)
                push!(pulses, Block{T}(n))
            end
        end
        return Block{T,N}(pulses, block.repeats)
    end
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

struct Sequence{T<:AbstractFloat,N}
    pulses::Vector{Union{Pulse{T,N},Block{T,N}}}
    repeats::Int
    detection_loop::Vector{Int}

    Sequence{T,N}(pulses::Vector{Union{Pulse{T,N},Block{T,N}}}, repeats, detection_loop) where {T<:AbstractFloat,N} =
        new{T,N}(pulses, repeats, detection_loop)

    function Sequence(pulses, repeats, detection_loop)
        T, N = partype(pulses[1])
        return new{T,N}(pulses, repeats, detection_loop)
    end

    function Sequence{T}(sequence::Sequence{T1,N}) where {T<:AbstractFloat,T1,N}
        pulses = Vector{Union{Pulse{T,N},Block{T,N}}}()
        for n in sequence.pulses
            if isa(n,Pulse)
                push!(pulses, Pulse{T}(n))
            elseif isa(n, Block)
                push!(pulses, Block{T}(n))
            end
        end
        return new{T,N}(pulses, sequence.repeats, sequence.detection_loop)
    end
end

include("hilbertoperators.jl")

struct PropagatorDict{T<:AbstractFloat,N,A<:AbstractArray}
    steps::Dict{NTuple{N,T}, Vector{Propagator{T,A}}}
    pulse_timings::Dict{NTuple{N,T}, Dict{NTuple{2,Int}, Set{NTuple{N,T}}}}
    prop_combinations::Dict{NTuple{N,T}, Dict{NTuple{2,Int}, Vector{Propagator{T,A}}}}
    unphased_propagators::Dict{NTuple{N,T}, Dict{NTuple{2,Int}, Propagator{T,A}}}
    pulse_props::Dict{NTuple{N,T}, Dict{Tuple{NTuple{2,Int}, NTuple{N,T}}, Propagator{T,A}}}

    PropagatorDict{T,N,A}() where {T<:AbstractFloat,N,A<:AbstractArray} =
        new{T,N,A}(Dict{NTuple{N,T},Vector{Propagator{T,A}}}(),
        Dict{NTuple{N,T}, Dict{NTuple{2,Int}, NTuple{N,T}}}(),
        Dict{NTuple{N,T}, Dict{NTuple{2,Int}, Vector{Propagator{T,A}}}}(),
        Dict{NTuple{N,T}, Dict{NTuple{2,Int}, Propagator{T,A}}}(),
        Dict{NTuple{N,T}, Dict{Tuple{NTuple{2,Int}, NTuple{N,T}}, Propagator{T,A}}}())
end

function add_rf!(prop_dict::PropagatorDict{T,N,A}, rf::NTuple{N,T}) where {T,N,A}
    prop_dict.steps[rf] = Vector{Propagator{T,A}}()
    prop_dict.pulse_timings[rf] = Dict{NTuple{2,Int}, Set{NTuple{N,T}}}()
    prop_dict.prop_combinations[rf] = Dict{NTuple{2,Int}, Vector{Propagator{T,A}}}()
    prop_dict.unphased_propagators[rf] = Dict{NTuple{2,Int}, Propagator{T,A}}()
    prop_dict.pulse_props[rf] = Dict{Tuple{NTuple{2,Int}, NTuple{N,T}}, Propagator{T,A}}()
    return prop_dict
end

function add_timing!(prop_dict::PropagatorDict{T,N,A}, rf::NTuple{N,T}, timing::NTuple{2,Int}) where {T,N,A}
    prop_dict.pulse_timings[rf][timing] = Set{NTuple{N,T}}()
end

keys(prop_dict::PropagatorDict) = keys(prop_dict.steps)
haskey(prop_dict::PropagatorDict,key) = haskey(prop_dict.steps,key)

get_number_type(A::AbstractArray{T,N}) where {T<:Number,N} = T
get_precision(A::AbstractArray) = real(get_number_type(A))

abstract type CalculationMode end
abstract type CPUMode <: CalculationMode end
abstract type GPUMode <: CalculationMode end
struct CPUSingleMode <: CPUMode end
struct CPUMultiProcess <: CPUMode end
struct GPUBatchedMode <: GPUMode end
struct GPUSingleMode <: GPUMode end

struct SimulationParameters{M<:CalculationMode,T<:AbstractFloat}
    period_steps::Int
    step_size::T
    nγ::Int
    γ_steps::Int
    angles::EulerAngles{T}
    xyz::Vector{Array{Complex{T},2}}

    function SimulationParameters{M,T}(period_steps::Integer,step_size,nγ::Integer,spins) where {M<:CalculationMode,T<:AbstractFloat}
        γ_steps = Int(period_steps/nγ)
        angles = magic_angle(T)

        x = X(Array{Complex{Float64}})
        y = Y(Array{Complex{Float64}})
        z = Z(Array{Complex{Float64}})
        channels = check_spins(spins)
        xyz=channel_XYZ(spins, channels, x, y, z)

        new{M,T}(period_steps, step_size, nγ, γ_steps, angles, xyz)
    end

    SimulationParameters{M}(period_steps, step_size ,nγ, spins) where M<:CalculationMode =
        SimulationParameters{M,Float64}(period_steps, step_size, nγ, spins)
end

SimulationParameters(period_steps, step_size, nγ, spins) =
    SimulationParameters{CPUSingleMode,Float64}(period_steps, step_size, nγ, spins)

function check_spins(spins)
    channels = [spin.channel for spin in spins]
    any(channels .< 1) && error("Channel indices cannot be less than 1")
    maximum(channels) == length(unique(channels)) || ("Channel indices should not skip any numbers")

    return maximum(channels)
end
