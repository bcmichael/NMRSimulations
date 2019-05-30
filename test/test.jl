using Test
include("../src/sim.jl")
include("examples.jl")

@testset "sim_types" begin
    @testset "SphericalTensor" begin
        @testset "constructor" begin
            for a in (rand(6),[rand(2,2) for n in 1:6])
                @test SphericalTensor(a...) isa SphericalTensor
            end
            @test SphericalTensor{Float32}(rand(6)...) isa SphericalTensor{Float32}
        end

        @testset "+" begin
            for n in ((rand(6),rand(6)),([rand(2,2) for n in 1:6],[rand(2,2) for n in 1:6]))
                a,b=n
                c,d=SphericalTensor(a...),SphericalTensor(b...)
                @test c+d isa SphericalTensor
                f=c+d
                for n in zip(fieldnames(SphericalTensor),a.+b)
                    @test getfield(f,n[1])==n[2]
                end
            end
        end

        @testset "*" begin
            a,b=rand(6),rand(2,2)
            @test SphericalTensor(a...)*b isa SphericalTensor
            c=SphericalTensor(a...)*b
            for n in zip(fieldnames(SphericalTensor),a)
                @test getfield(c,n[1])==n[2]*b
            end
        end
    end

    @testset "Spin" begin
        a=rand(3)
        b=rand(3).*360
        c=EulerAngles(b...)
        @test Spin(1,a...,b...) isa Spin{Float64}
        @test Spin(1,a...,c) isa Spin{Float64}
        @test Spin{Float32}(1,a...,b...) isa Spin{Float32}
        @test Spin{Float32}(1,a...,c) isa Spin{Float32}
        @test Spin(1,Float32.(a)...,b...) isa Spin{Float64}
        @test check_spins([Spin(1,a...,c)]) == 1
        @test check_spins([Spin(1,a...,c),Spin(1,b...,c)]) == 1
        @test check_spins([Spin(1,a...,c),Spin(2,b...,c)]) == 2
        @test_throws ArgumentError check_spins([Spin(1,a...,c),Spin(3,b...,c)])
        @test_throws ArgumentError check_spins([Spin(0,a...,c)])
    end

    @testset "Pulse" begin
        for n=1:3
            a=rand()
            b=rand(2*n)
            @test Pulse(a,b...) isa Pulse{Float64,n}
            @test Pulse(Float32(a),b...) isa Pulse{Float64,n}
            @test Pulse(Float32(a),Float32.(b)...) isa Pulse{Float32,n}
            @test Pulse{Float32}(a,b...) isa Pulse{Float32,n}
            c=Pulse(a,b...)
            @test c.t==a
            @test all(c.γB1.==b[1:2:end])
            @test all(c.phase.==b[2:2:end])
            @test Pulse{Float32}(c) isa Pulse{Float32,n}
            @test duration(c)==a
            @test partype(c)==(Float64,n)
        end
    end

    @testset "Block" begin
        for n=1:3
            a=Pulse(rand(2*n+1)...)
            @test Block([a]) isa Block{Float64,n}
            @test Block([a,a]) isa Block{Float64,n}
            @test Block([a],2) isa Block{Float64,n}
            b=Block([a,a])
            @test Block([a,b]) isa Block{Float64,n}
            @test partype(b)==(Float64,n)
            @test duration(b)==a.t*2
            @test Block{Float32}(b) isa Block{Float32,n}
            c=Block([a,a])
            d = Block([a,a],2)
            e = Block([a],2)
            @test hash(b) == hash(c)
            @test hash(b) == hash(e)
            @test hash(b) != hash(d)
            @test isequal(b, c)
            @test isequal(b, e)
            @test ! isequal(b, d)
            @test b.rank == 1
            @test Block([b]).rank == 1
            @test Block([b, a]).rank == 2
            @test Block([b,a], 2).rank == 3
        end
    end

    @testset "Sequence" begin
        for n=1:3
            a=Pulse(rand(2*n+1)...)
            b=Block([a,a])
            @test Sequence([a],[([1], 2)]) isa Sequence{Float64,n,1}
            @test Sequence([b],[([1], 2)]) isa Sequence{Float64,n,1}
            @test Sequence([a,b],[([1], 2)]) isa Sequence{Float64,n,1}
            @test Sequence{Float32}([a,b],[([1], 2)]) isa Sequence{Float32,n,1}
            @test Sequence([a,b],[([1], 2), ([2], 2)]) isa Sequence{Float64,n,2}
            c=Sequence([a,b],[([1], 2)])
            @test Sequence{Float32}(c) isa Sequence{Float32,n,1}
            @test_throws ArgumentError Sequence([a,b],[([2, 1], 2)])
            @test_throws MethodError Sequence{Float32,4}([a,b],[([1], 2)])
            @test_throws ArgumentError Sequence([a,b],[([3], 2)])
            @test_throws ArgumentError Sequence([a,b],[([2], 2), ([1], 2)])
        end
    end

    @testset "HilbertOperator" begin
        @testset "Propagator" begin
            @test_throws MethodError Propagator(rand(4,4,1))
            @test_throws MethodError Propagator(rand(Char,4,4,1))
            @test Propagator(rand(Complex{Float64},4,4,1)) isa Propagator
        end

        @testset "Hamiltonian" begin
            @test Hamiltonian(rand(4,4,1)) isa Hamiltonian
            @test_throws MethodError Hamiltonian(rand(Char,4,4,1))
            @test Hamiltonian(rand(Complex{Float64},4,4,1)) isa Hamiltonian
        end

        @testset "Math" begin
            a=Propagator(rand(Complex{Float64},4,4))
            b=Propagator(rand(Complex{Float64},4,4))
            c=Propagator(rand(Complex{Float64},4,4))
            mul!(c,a,b)
            @test a.data[:,:,1]*b.data[:,:,1]==c.data[:,:,1]
            mul!(c,a,b,'N','C')
            @test a.data[:,:,1]*b.data[:,:,1]'==c.data[:,:,1]
            mul!(c,a,b,2,0)
            @test 2*a.data[:,:,1]*b.data[:,:,1]==c.data[:,:,1]

            d = b.data^(4)
            pow!(a, b, 4, c)
            @test all(a.data .== d)
        end

        @testset "Convenience functions" begin
            a=Propagator(rand(Complex{Float64},4,4,1))
            b=Propagator(rand(Complex{Float64},4,4,1))
            c=Propagator(rand(Complex{Float64},4,4,1))
            d=Hamiltonian(rand(4,4,1))
            @test copy(a).data==a.data
            @test size(similar(a).data)==size(a.data)
            @test typeof(similar(a).data)==typeof(a.data)
            @test operator_iter(a)==(4,1)
        end
    end
end

@testset "hamiltonians" begin
    @testset "pauli matrices" begin
        for  pauli in zip((X,Y,Z),([0*im 1;1 0]/2,[0 -1*im;1*im 0]/2,[1 0*im;0 -1]/2)),T1 in (Float64,Float32),T in (Array,SparseMatrixCSC)
            @test pauli[1](T{Complex{T1}})==pauli[2]
        end
    end

    @testset "rotations" begin
        @testset "wigner2_elements" begin
            #test that some of the properties of the wigner rotation matrix are obeyed
            a=rand()*360
            for n=-2:2,m=-2:2
                @test wigner2_elements[n+3,m+3](a)==wigner2_elements[-m+3,-n+3](a)
                @test wigner2_elements[n+3,m+3](a)==wigner2_elements[m+3,n+3](a)*(-1)^(n-m)
                @test wigner2_elements[n+3,m+3](0)==Int(n==m)
            end
        end

        @testset "wigner2" begin
            a=rand(3).*360
            b=EulerAngles(a...)
            for n=-2:2,m=-2:2
                @test begin
                    c=wigner2(n,m,b)
                    c.im/c.re≈tand(-(m*a[1]+n*a[3]))
                end
                @test abs(wigner2(n,m,b))≈abs(wigner2_elements[n+3,m+3](a[2]))
            end
        end

        @testset "rotate_component2" begin
            a=rand(6)
            b=SphericalTensor(a...)
            c=EulerAngles(rand(),0.0,rand())
            for n=-2:2
                @test rotate_component2(b,n,c) isa Complex
                d=Symbol(n<0 ? "s2m$(-n)" : "s2$n")
                @test rotate_component2(b,n,c)≈wigner2(n,n,c)*getfield(b,d)
            end
        end

        @testset "euler_rotation" begin
            a=EulerAngles(rand(3).*360...)
            b=SphericalTensor(rand(6)...)
            @test euler_rotation(b,a).s00==b.s00
            @test euler_rotation(b,a) isa SphericalTensor{Complex{Float64}}
            c=euler_rotation(b,a)
            d=EulerAngles(-a.γ,-a.β,-a.α)
            f=euler_rotation(c,d)
            for n in fieldnames(typeof(b))
                @test getfield(b,n)≈getfield(f,n)
            end
        end
    end

    @testset "krons" begin
        @testset "kron_up" begin
            for n=1:10
                a=rand(Complex{Float64},2,2)
                b=rand(1:n)
                @test size(kron_up(a,b,n))==(2^n,2^n)

                #check a predictable entry to make sure it is right
                x=[0.0*im 1;1 0]
                @test kron_up(x,b,n)[(2^(n-b))+1,1]==1
            end
        end

        @testset "kron_double" begin
            for n=2:10
                a=rand(Complex{Float64},2,2)
                b=rand(Complex{Float64},2,2)
                c=rand(1:n)
                d=rand(1:n)
                while d==c
                    d=rand(1:n)
                end
                @test size(kron_double(a,b,c,d,n))==(2^n,2^n)
                @test_throws ArgumentError kron_double(a,b,c,c,n)
            end
            a=rand(Complex{Float64},2,2)
            b=rand(Complex{Float64},2,2)
            @test kron_double(a,b,1,2,2)==kron(a,b)
            @test kron_double(a,b,2,1,2)==kron(b,a)
        end
    end

    @testset "chemical shift" begin
        a=[Spin(1,rand(3)...,rand(3).*360...) for n=1:5]
        for n=1:5
            @test initial_cs(a[1:n]) isa SphericalTensor{Array{Complex{Float64},2}}
            b=initial_cs(a[1:n])
            for m in fieldnames(typeof(b))
                c=getfield(b,m)
                @test size(c)==(2^n,2^n)
                @test isdiag(c)
            end
        end
        c=Spin(1,rand(3)...,0,0,0)
        d=initial_cs([c])
        @test d.s00[1,1]==c.sigma_iso/2
        @test d.s20[1,1]==c.anisotropy/2
        @test all(d.s21.==0)
        @test d.s2m1==d.s21
        @test d.s22==d.s2m2
    end

    @testset "rf" begin
        @testset "channel_XYZ" begin
            a=[Spin(n,rand(6)...) for n=1:3]
            x=[0*im 1;1 0]/2
            y=[0 -1*im;1*im 0]/2
            z=[1 0*im;0 -1]/2
            for m=1:3
                @test channel_XYZ(a[1:m],m,x,y,z) isa Vector{Array{Complex{Float64},2}}
                b=channel_XYZ(a[1:m],m,x,y,z)
                @test length(b)==3*m
                @test maximum(getfield.(b[1],:re))==0.5
                @test all(getfield.(b[1],:im).==0)
                @test maximum(getfield.(b[2],:im))==0.5
                @test all(getfield.(b[2],:re).==0)
                @test maximum(getfield.(b[3],:re))==0.5
                @test all(getfield.(b[3],:im).==0)
            end
            @test channel_XYZ(a[1:1],1,x,y,z)==[x,y,z]
        end

        @testset "pulse_H" begin
            x=[0*im 1;1 0]/2
            y=[0 -1*im;1*im 0]/2
            z=[1 0*im;0 -1]/2
            a=[Spin(n,rand(6)...) for n=1:3]
            for n=1:3
                b=tuple(rand(n).*100...)
                c=channel_XYZ(a[1:n],n,x,y,z)
                @test pulse_H(b,c) isa Array{Complex{Float64},2}
                # @test pulse_H(b,c)[2] isa Diagonal{Complex{Float64}}
                d=pulse_H(b,c)
                @test size(d)==(2^n,2^n)
                @test all(((x,y)-> (x==0 && y==0) || (x!=0 && y!=0)).(d,sum(c[1:3:end])+sum(c[2:3:end])))
            end
        end

        @testset "phase_rotator" begin
            x=[0*im 1;1 0]/2
            y=[0 -1*im;1*im 0]/2
            z=[1 0*im;0 -1]/2
            a=[Spin(n,rand(6)...) for n=1:3]
            for n=1:3
                b=tuple(rand(n).*360...)
                c=channel_XYZ(a[1:n],n,x,y,z)
                @test phase_rotator(b,c) isa Array{Complex{Float64},2}
                d=phase_rotator(b,c)
                @test d == d'
                @test all(Diagonal(d).diag.≈1)
            end
        end
    end

    @testset "couplings" begin
        @testset "dipole_coupling" begin
            a=[Spin(n,rand(6)...) for n in (1,1,2)]
            b=rand()
            @test_throws ArgumentError dipole_coupling(a,1,1,b)
            @test dipole_coupling(a,1,2,b).s20==dipole_coupling(a,2,1,b).s20
            @test !isdiag(dipole_coupling(a,1,2,b).s20)
            @test isdiag(dipole_coupling(a,1,3,b).s20)
            c=dipole_coupling(a,1,2,b)
            for n in fieldnames(typeof(c))
                if n==:s20
                    @test !all(getfield(c,n).==0)
                else
                    @test all(getfield(c,n).==0)
                end
            end
        end

        @testset "j_coupling" begin
            a=[Spin(n,rand(6)...) for n in (1,1,2)]
            b=rand(3)
            @test_throws ArgumentError j_coupling(a,1,1,b...)
            @test j_coupling(a,1,2,b...).s20==j_coupling(a,2,1,b...).s20
            @test !isdiag(j_coupling(a,1,2,b...).s20)
            @test isdiag(j_coupling(a,1,3,b...).s20)
            @test !isdiag(j_coupling(a,1,2,b...).s00)
            @test isdiag(j_coupling(a,1,3,b...).s00)
            c=j_coupling(a,1,2,b...)
            for n in fieldnames(typeof(c))
                if n==:s21 || n==:s2m1
                    @test all(getfield(c,n).==0)
                else
                    @test !all(getfield(c,n).==0)
                end
            end
        end
    end
end

@testset "propagation" begin
    @testset "sidebands" begin
        seq = Sequence([Pulse(31.25, 0, 0)], [([1], 1024)])
        par = SimulationParameters(800, 1.25, 100, [Spin(1, 0, 10000, -0.5, 0, 0, 0)])
        @test build_generator(seq, par) isa PropagationGenerator{Float64,1,Array{Complex{Float64},2},1}
        gen = build_generator(seq, par)
        @test length(gen) == 1024
        @test gen.size == (1024,)
        @test size(gen.loops[1].chunks) == (1,1)
        @test gen.loops[1].start_cycle == 1
        @test gen.loops[1].cycle == 32
        @test length(gen.final) == 0
    end

    @testset "redor" begin
        d45 = Pulse(45, 0, 0, 0, 0)
        redor_b = Block([d45,
            Pulse(5, 0, 0, 100, 0),
            d45,
            Pulse(5, 0, 0, 100, 90)], 1)
        seq = Sequence([redor_b, d45, Pulse(5,0,0,100,0), d45, Pulse(5,100,0,0,0), redor_b, redor_b], [([1,7], 61)])
        par = SimulationParameters(100, 1, 100, [Spin(1, 2000, 0, 0, 0, 0, 0), Spin(2, 0, 0, 0, 0, 0, 0)])
        @test build_generator(seq, par) isa PropagationGenerator{Float64,2,Array{Complex{Float64},2},1}
        gen = build_generator(seq, par)
        @test length(gen) == 61
        @test gen.size == (61,)
        @test size(gen.loops[1].chunks) == (2,1)
        @test gen.loops[1].start_cycle == 1
        @test gen.loops[1].cycle == 1
        @test length(gen.final) == 0
    end

    @testset "rfdr2d" begin
        seq = Sequence{Float32}([Pulse(20, 0, 0),
                                 Pulse(2, 125, 270),
                                 Block([Pulse(48, 0, 0), Pulse(4, 125, 0), Pulse(48, 0, 0)],2),
                                 Pulse(2, 125, 90),
                                 Pulse(4, 0, 0)],
                                 [([1], 256), ([5], 512)])
        par = SimulationParameters{CPUSingleMode, Float32}(100, 1, 25, [Spin{Float32}(1,0,0,0,0,0,0),Spin{Float32}(1,10000,0,0,0,0,0)])
        @test build_generator(seq, par) isa PropagationGenerator{Float32,1,Array{Complex{Float32},2},2}
        gen = build_generator(seq, par)
        @test length(gen) == 256*512
        @test gen.size == (256,512)
        @test size(gen.loops[1].chunks) == (1,1)
        @test gen.loops[1].start_cycle == 1
        @test gen.loops[1].cycle == 5
        @test gen.loops[2].start_cycle == 5
        @test gen.loops[2].cycle == 25
        @test length(gen.final) == 0
    end
end

@testset "prop cache" begin
    @testset "threshold" begin
        a=rand(16,16)
        @test threshold(a,1)
        @test !threshold(a,maximum(a)/2)
    end

    @testset "eig_max_bound" begin
        for n=1:6
            a=Symmetric(rand(2^n,2^n))
            @test eig_max_bound(a)>=eigmax(a)
        end
    end

    @testset "expm_cheby" begin
        for n=1:2,T in (Float64,Float32)
            temps=[Hamiltonian(rand(T,(2^n,2^n))) for j in 1:2]
            c = Propagator(rand(Complex{T},(2^n,2^n)))
            for a in (rand(T,(2^n,2^n)),)
                b=exp(-2*pi*im*a[:,:,1]*1E-6)
                @test expm_cheby!(c, Hamiltonian(a),1E-6,temps).data≈b atol=1E-4
            end
        end
    end

    @testset "SimCache" begin
        @testset "sidebands" begin
            seq = Sequence([Pulse(31.25, 0, 0)], [([1], 1024)])
            par = SimulationParameters(800, 1.25, 100, [Spin(1, 0, 10000, -0.5, 0, 0, 0)])
            push!(par.temps, Propagator(Array{ComplexF64,2}(undef,(2,2))))
            gen = build_generator(seq, par)
            cache = build_prop_cache(gen, (2,2), par)
            @test cache isa SimCache{Float64, 1, Array{ComplexF64,2}}
            @test length(cache.pulses) == 1
            @test haskey(cache.pulses, (0,))
            rf = cache.pulses[(0,)]
            @test length(rf.timings) == 32
            @test length(rf.combinations) == 8
            for c in values(rf.combinations)
                length(c) == 200
            end
            @test length(cache.blocks.ranks) == 1
        end

        @testset "redor" begin
            d45 = Pulse(45, 0, 0, 0, 0)
            redor_b = Block([d45,
                Pulse(5, 0, 0, 100, 0),
                d45,
                Pulse(5, 0, 0, 100, 90)], 1)
            seq = Sequence([redor_b, d45, Pulse(5,0,0,100,0), d45, Pulse(5,100,0,0,0), redor_b, redor_b], [([1,7], 61)])
            par = SimulationParameters(100, 1, 100, [Spin(1, 2000, 0, 0, 0, 0, 0), Spin(2, 0, 0, 0, 0, 0, 0)])
            push!(par.temps, Propagator(Array{ComplexF64,2}(undef,(4,4))))
            gen = build_generator(seq, par)

            cache = build_prop_cache(gen, (2,2), par)
            @test cache isa SimCache{Float64, 2, Array{ComplexF64,2}}
            @test length(cache.pulses) == 3
            for rf in values(cache.pulses)
                @test length(rf.combinations) == 1
                @test length(rf.combinations[(1,0)]) == 100
                for t in values(rf.timings)
                    @test length(t.phases) == 1
                end
            end
            @test length(cache.pulses[(0,0)].timings) == 2
            @test length(cache.pulses[(0,100)].timings) == 2
            @test length(cache.pulses[(100,0)].timings) == 1
            @test (0,90) in cache.pulses[(0,100)].timings[(96,5)].phases
            @test length(cache.blocks.ranks) == 2
        end
    end
end

@testset "examples" begin
    @test rfdr()[1:5]≈[0.0+0.0im, -0.000304554-9.93079e-9im, -0.00121739-7.93902e-8im, -0.00273603-2.67627e-7im, -0.00485636-6.33332e-7im] atol=1E-5
    @test sidebands()[1:5]≈[0.5+0.0im, 0.321036-0.02204im, 0.0383774-0.094324im, -0.0511445-0.100016im, -0.0164907-0.0230406im] atol=1E-5
    @test redor()[1:5]≈[0.938156+0.0587413im, 0.774492+0.0491454im, 0.547457+0.0340485im, 0.311615+0.0198252im, 0.115145+0.0072829im] atol=1E-5
    @test rfdr_long()[1:5]≈[0.0+0.0im, -6.16922e-5-1.91144e-5im, -0.000203267-0.000139403im, -0.000317271-0.000401642im, -0.000290586-0.000755885im] atol=1E-5
    @test rfdr_2d(10)[1:100]≈[5.94705-0.291208im, 3.94143+0.29294im, 1.60466+0.888975im, 2.65656+1.03046im, 4.60187+1.09292im, 3.15039+1.35715im, 0.0328145+1.56404im, 0.0199595+1.45841im, 2.11525+1.00569im, 1.70712+0.729686im, 3.64488-1.30457im, 2.36274+0.0756529im, 1.27366+1.27119im, 1.676+0.822803im, 2.19231-0.0161242im, 1.0728+0.641779im, -0.525871+1.66838im, -0.478352+1.19183im, 0.286308-0.167459im, -0.149357-0.424713im, 1.58762+1.17066im, 1.27315+1.60072im, 0.931834+1.46479im, 0.6035+1.54142im, 0.10234+1.87338im, -0.349667+1.94337im, -0.695979+1.46913im, -0.961717+0.990982im, -1.15781+0.925729im, -1.31725+0.655551im, 2.99488+3.55313im, 1.94546+2.73533im, 0.592872+1.45425im, 0.733655+2.16429im, 1.24526+3.53781im, 0.462462+2.8248im, -0.902864+0.870142im, -1.01548+0.688069im, -0.0766618+1.83508im, -0.109601+1.44977im, 4.73168+2.36862im, 2.52302+2.02124im, 0.0999452+1.51854im, 0.970558+1.66842im, 2.68387+2.02592im, 1.32971+1.50896im, -1.35061+0.459443im, -1.03546+0.196577im, 1.27367+0.392034im, 1.25762-0.0105418im, 2.88574+0.347241im, 1.10365+0.982685im, -0.356996+1.53334im, 0.16592+0.89058im, 1.01193-0.141554im, -0.0738062-0.166613im, -1.65158+0.201073im, -1.13667-0.298985im, 0.329831-1.41892im, 0.307237-1.76836im, -0.0837511+1.60909im, -0.596119+1.74807im, -0.713821+1.35468im, -0.924729+0.929104im, -1.44008+0.628309im, -1.71476+0.423992im, -1.50456-0.0798125im, -1.1711-0.649217im, -1.11744-0.87081im, -0.975532-1.02137im, 0.319908+3.96648im, -0.305051+2.65476im, -0.974572+0.956509im, -0.937677+1.36289im, -0.765+2.388im, -0.931996+1.3684im, -1.17553-0.720169im, -0.925563-0.81735im, -0.228921+0.518148im, 0.120744+0.277155im, 2.35229+3.0502im, 0.598474+1.79808im, -1.17577+0.518513im, -0.244606+0.716286im, 1.39601+1.25212im, 0.649611+0.287692im, -1.07165-1.27816im, -0.433495-1.18813im, 1.68847-0.244862im, 1.98765-0.429904im, 1.50836+0.184157im, -0.0741303+0.143524im, -1.32778+0.170424im, -0.40283-0.514611im, 0.987334-1.46853im, 0.35182-1.68473im, -0.917296-1.45709im, -0.0959247-1.58837im, 1.69711-2.09943im, 1.92907-2.10628im] atol=1E-4
end

@testset "cuda_examples" begin
    @test rfdr(GPUBatchedMode, Float32)[1:5]≈[0.0+0.0im, -0.000304554-9.93079e-9im, -0.00121739-7.93902e-8im, -0.00273603-2.67627e-7im, -0.00485636-6.33332e-7im] atol=1E-5
    @test rfdr(GPUSingleMode, Float32)[1:5]≈[0.0+0.0im, -0.000304554-9.93079e-9im, -0.00121739-7.93902e-8im, -0.00273603-2.67627e-7im, -0.00485636-6.33332e-7im] atol=1E-5
end
