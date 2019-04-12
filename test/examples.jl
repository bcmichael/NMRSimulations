function rfdr(M=CPUSingleMode)
    spins = [Spin(1, 0, 0, 0, 0, 0, 0), Spin(1, 1000, 0, 0, 0, 0, 0)]
    cs = initial_cs(spins)
    dip = dipole_coupling(spins,1,2,1000)
    H = cs+dip

    rfdr = Block([Pulse(48,0,0),
        Pulse(4,125,0),
        Pulse(48,0,0)])

    sequence = Sequence([rfdr, rfdr], 500, [1, 2])

    parameters = SimulationParameters{M}(100, 1, 10, spins)

    x = X(Array{Complex{Float64}})
    y = Y(Array{Complex{Float64}})
    z = Z(Array{Complex{Float64}})
    p = sparse(kron_up(x, 1, 2))
    detect = sparse(kron_up(x+im*y, 2, 2))

    crystallites, weights = get_crystallites("test/rep100.cry")
    spec = powder_average(sequence, H, p, detect, crystallites, weights, parameters)

    GC.gc()
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
    crystallites, weights = get_crystallites("test/rep100.cry")
    sequence = Sequence([Pulse(31.25, 0, 0)], 1024, [1])

    parameters = SimulationParameters(800, 1.25, 100, spins)

    spec = powder_average(sequence, cs, p, detect, crystallites, weights, parameters)

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
    crystallites, weights = get_crystallites("test/rep100.cry")

    d45 = Pulse(45, 0, 0, 0, 0)
    redor = Block([d45,
        Pulse(5, 0, 0, 100, 0),
        d45,
        Pulse(5, 0, 0, 100, 90)], 1)

    sequence = Sequence([redor, d45, Pulse(5,0,0,100,0), d45, Pulse(5,100,0,0,0), redor, redor], 61, [1,7])

    parameters = SimulationParameters(100, 1, 100, spins)

    spec = powder_average(sequence, H, p, detect, crystallites, weights, parameters)
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
    sequence = Sequence{T}([rfdr], 1000, [1])

    x = X(Array{Complex{T}})
    y = Y(Array{Complex{T}})
    z = Z(Array{Complex{T}})

    p = sparse(kron_up(x, 1, a))
    detect = sparse(kron_up(x+im*y, 2, a))
    crystallites,weights = get_crystallites("test/rep100.cry", T)

    parameters = SimulationParameters{CPUSingleMode,T}(100, 1, 10, spins)

    spec = powder_average(sequence, H, p, detect, crystallites, weights, parameters)
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

    sequence = Sequence([rfdr], 500, [1])

    parameters = SimulationParameters{M}(100, 1, 10, spins)

    x = X(Array{Complex{Float64}})
    y = Y(Array{Complex{Float64}})
    z = Z(Array{Complex{Float64}})
    p = sparse(kron_up(x, 1, 2))
    detect = sparse(kron_up(x+im*y, 2, 2))

    crystallites, weights = get_crystallites("test/rep100.cry")
    spec = powder_average(sequence, H, p, detect, crystallites, weights, parameters)

    GC.gc()
    return spec
end
