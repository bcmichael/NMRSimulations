function rfdr(M=CPUSingleMode, ::Type{T} = Float64) where T
    spins = [Spin{T}(1, 0, 0, 0, 0, 0, 0), Spin{T}(1, 1000, 0, 0, 0, 0, 0)]
    cs = initial_cs(spins)
    dip = dipole_coupling(spins,1,2,1000)
    H = cs+dip

    rfdr = Block([Pulse(48,0,0),
        Pulse(4,125,0),
        Pulse(48,0,0)])

    sequence = Sequence{T}([rfdr, rfdr], [([1, 2], 500)])

    parameters = SimulationParameters{M,T}(100, 1, 10, spins)

    x = X(Array{Complex{T}})
    y = Y(Array{Complex{T}})
    z = Z(Array{Complex{T}})
    p = sparse(kron_up(x, 1, 2))
    detect = sparse(kron_up(x+im*y, 2, 2))

    crystallites = read_crystallites("test/rep100.cry", T)
    spec = powder_average(sequence, H, p, detect, crystallites, parameters)

    return spec
end

function sidebands()
    spins = [Spin(1, 0, 10000, -0.5, 0, 0, 0)]
    cs = initial_cs(spins)
    x = X(Array{Complex{Float64}})
    y = Y(Array{Complex{Float64}})
    z = Z(Array{Complex{Float64}})
    p = sparse(kron_up(x, 1, 1))
    detect = sparse(kron_up(x+im*y, 1, 1))
    crystallites = read_crystallites("test/rep100.cry")
    sequence = Sequence([Pulse(31.25, 0, 0)], [([1], 1024)])

    parameters = SimulationParameters(800, 1.25, 100, spins)

    spec = powder_average(sequence, cs, p, detect, crystallites, parameters)

    return spec
end

function redor()
    spins = [Spin(1, 2000, 0, 0, 0, 0, 0), Spin(2, 0, 0, 0, 0, 0, 0)]
    cs = initial_cs(spins)
    dip = dipole_coupling(spins, 1, 2, 1199)
    H = cs+dip
    x = X(SparseMatrixCSC{Complex{Float64}})
    y = Y(SparseMatrixCSC{Complex{Float64}})
    z = Z(SparseMatrixCSC{Complex{Float64}})
    p = kron_up(x, 1, 2)
    detect = kron_up(x+im*y, 1, 2)
    crystallites = read_crystallites("test/rep100.cry")

    d45 = Pulse(45, 0, 0, 0, 0)
    redor = Block([d45,
        Pulse(5, 0, 0, 100, 0),
        d45,
        Pulse(5, 0, 0, 100, 90)], 1)

    sequence = Sequence([redor, d45, Pulse(5,0,0,100,0), d45, Pulse(5,100,0,0,0), redor, redor], [([1,7], 61)])

    parameters = SimulationParameters(100, 1, 100, spins)

    spec = powder_average(sequence, H, p, detect, crystallites, parameters)
    return spec
end

function ubi(a,T::Type = Float64)
    d = readdlm("test/ubi/ubi.j")
    spins = [Spin{T}(1, zeros(6)...) for n in 1:a]
    dips = Vector{SphericalTensor{Array{Complex{T},2}}}()
    for n = 1:size(d, 1)
        if d[n,1] <= a && d[n,2] <=a
            dip = dipole_coupling(spins, Int(d[n, 1]), Int(d[n, 2]), d[n, 4])
            rot = EulerAngles{T}(0, d[n, 7], d[n, 8])
            push!(dips, euler_rotation(dip, rot))
        end
    end
    dip = sum(dips)
    cs = initial_cs(spins)
    H = cs+dip

    rfdr = Block([Pulse(48, 0, 0),
        Pulse(4, 125, 0),
        Pulse(48, 0, 0)])
    sequence = Sequence{T}([rfdr], [([1], 1000)])

    x = X(Array{Complex{T}})
    y = Y(Array{Complex{T}})
    z = Z(Array{Complex{T}})

    p = sparse(kron_up(x, 1, a))
    detect = sparse(kron_up(x+im*y, 2, a))
    crystallites = read_crystallites("test/rep100.cry", T)

    parameters = SimulationParameters{CPUSingleMode,T}(100, 1, 10, spins)

    spec = powder_average(sequence, H, p, detect, crystallites, parameters)
    return spec
end

# not really a meaningful example but tests a pulse longer than a cycle
function rfdr_long(M=CPUSingleMode)
    spins = [Spin(1, 0, 0, 0, 0, 0, 0), Spin(1, 1000, 0, 0, 0, 0, 0)]
    cs = initial_cs(spins)
    dip = dipole_coupling(spins,1,2,1000)
    H = cs+dip

    rfdr = Block([Pulse(48,0,0),
        Pulse(504,125,0),
        Pulse(48,0,0)])

    sequence = Sequence([rfdr], [([1], 500)])

    parameters = SimulationParameters{M}(100, 1, 10, spins)

    x = X(Array{Complex{Float64}})
    y = Y(Array{Complex{Float64}})
    z = Z(Array{Complex{Float64}})
    p = sparse(kron_up(x, 1, 2))
    detect = sparse(kron_up(x+im*y, 2, 2))

    crystallites = read_crystallites("test/rep100.cry")
    spec = powder_average(sequence, H, p, detect, crystallites, parameters)

    return spec
end

function rfdr_2d(size, ::Type{T} = Float64) where T
    cs_iso = [-11.732, 0.199, 2.154, 2.829, 3.543, 3.637]
    spins = [Spin{T}(1, i*1000, 0, 0, 0, 0, 0) for i in cs_iso[1:3]]
    pos = [3.734 6.733 2.822; 3.522 7.597 1.589; 4.043 6.87 0.351; 4.541 7.765 -0.791; 3.423 8.516 -1.497; 5.34 6.943 -1.789]
    dips = Vector{SphericalTensor{Array{Complex{T},2}}}()
    for a in 2:3
        for b = 1:a-1
            dif = pos[b,:].-pos[a,:]
            distance = sqrt(sum(dif.^2))
            strength = 7598.1028703221855/(distance^3)
            β = acosd(dif[3]/distance)
            γ = 180-atand(dif[2], dif[1])
            dip = dipole_coupling(spins, a, b, strength)
            rot = EulerAngles{T}(0,β,γ)
            push!(dips, euler_rotation(dip, rot))
        end
    end
    dip = sum(dips)
    cs = initial_cs(spins)
    H = cs+dip

    rfdr_xy8 = Block([Pulse(48, 0, 0), Pulse(4, 125, 0), Pulse(48, 0, 0),
                   Pulse(48, 0, 0), Pulse(4, 125, 90), Pulse(48, 0, 0),
                   Pulse(48, 0, 0), Pulse(4, 125, 0), Pulse(48, 0, 0),
                   Pulse(48, 0, 0), Pulse(4, 125, 90), Pulse(48, 0, 0),
                   Pulse(48, 0, 0), Pulse(4, 125, 90), Pulse(48, 0, 0),
                   Pulse(48, 0, 0), Pulse(4, 125, 0), Pulse(48, 0, 0),
                   Pulse(48, 0, 0), Pulse(4, 125, 90), Pulse(48, 0, 0),
                   Pulse(48, 0, 0), Pulse(4, 125, 0), Pulse(48, 0, 0)])

    parameters = SimulationParameters{CPUSingleMode, T}(100, 1, 25, spins)
    p = sparse(parameters.xyz[1])
    detect = sparse(parameters.xyz[1]+im*parameters.xyz[2])
    crystallites = read_crystallites("test/rep100.cry", T)

    sequence = Sequence{T}([Pulse(20, 0, 0),
                            Pulse(2, 125, 270),
                            Block([rfdr_xy8],2),
                            Pulse(2, 125, 90),
                            Pulse(20, 0, 0)],
                            [([1], size), ([5], size)])
    spec = powder_average(sequence, H, p, detect, crystallites, parameters)
    return spec
end
