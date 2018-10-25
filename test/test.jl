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

    @testset "EulerAngles" begin
        a=rand(3).*360
        @test EulerAngles(a...) isa EulerAngles{Float64}
        @test EulerAngles{Float32}(a...) isa EulerAngles{Float32}
        @test EulerAngles(a[1],a[2],Float32(a[3])) isa EulerAngles{Float64}
        b=EulerAngles(a...)
        @test EulerAngles{Float32}(b) isa EulerAngles{Float32}
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
        end
    end

    @testset "Sequence" begin
        for n=1:3
            a=Pulse(rand(2*n+1)...)
            b=Block([a,a])
            @test Sequence([a],2,[1]) isa Sequence{Float64,n}
            @test Sequence([b],2,[1]) isa Sequence{Float64,n}
            # @test Sequence([a,b],2,[1]) isa Sequence{Float64,n}
            # c=Sequence([a,b],2,[1])
            # @test Sequence{Float32}(c) isa Sequence{Float32,n}
        end
    end

    @testset "HilbertOperator" begin
        @testset "Propagator" begin
            @test_throws ErrorException Propagator(rand(4,4,1))
            @test_throws MethodError Propagator(rand(Char,4,4,1))
            @test Propagator(rand(Complex{Float64},4,4,1)) isa Propagator
        end

        @testset "Hamiltonian" begin
            @test Hamiltonian(rand(4,4,1)) isa Hamiltonian
            @test_throws MethodError Hamiltonian(rand(Char,4,4,1))
            @test Hamiltonian(rand(Complex{Float64},4,4,1)) isa Hamiltonian
        end

        @testset "Math" begin
            a=Propagator(rand(Complex{Float64},4,4,1))
            b=Propagator(rand(Complex{Float64},4,4,1))
            c=Propagator(rand(Complex{Float64},4,4,1))
            mul!(c,a,b)
            @test a.data[:,:,1]*b.data[:,:,1]==c.data[:,:,1]
            mul!(c,a,b,'N','C')
            @test a.data[:,:,1]*b.data[:,:,1]'==c.data[:,:,1]
            mul!(c,a,b,2,0)
            @test 2*a.data[:,:,1]*b.data[:,:,1]==c.data[:,:,1]
            c=a.data.*b.data
        end

        @testset "Convenience functions" begin
            a=Propagator(rand(Complex{Float64},4,4,1))
            b=Propagator(rand(Complex{Float64},4,4,1))
            c=Propagator(rand(Complex{Float64},4,4,1))
            d=Hamiltonian(rand(4,4,1))
            @test copy(a).data==a.data
            @test size(similar(a).data)==size(a.data)
            @test typeof(similar(a).data)==typeof(a.data)
            @test similar(d,Propagator) isa Propagator
            @test eltype(similar(d,Propagator).data)==Complex{Float64}
            @test operator_iter(a)==(4,1)
        end
    end

    @testset "PropagatorDict" begin
        for n=1:3
            a=PropagatorDict{Float64,n,Array{Complex{Float64},2}}()
            b=tuple(rand(n)...)
            @test add_rf!(a,b) isa PropagatorDict
            @test haskey(a,b)
            @test keys(a) isa Base.KeySet
            for n in fieldnames(typeof(a))
                @test haskey(getfield(a,n),b)
            end
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

@testset "sim_mas" begin
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
            temps=[Hamiltonian(rand(T,(2^n,2^n,1))) for j in 1:2]
            for a in (rand(T,(2^n,2^n,1)),)
                b=exp(-2*pi*im*a[:,:,1]*1E-6)
                @test expm_cheby(Hamiltonian(a),1E-6,temps).data≈b atol=1E-4
            end
        end
    end
end

@testset "examples" begin
    @test rfdr()[1:5]≈[0.0+0.0im, -0.000304554-9.93079e-9im, -0.00121739-7.93902e-8im, -0.00273603-2.67627e-7im, -0.00485636-6.33332e-7im] atol=1E-5
    @test sidebands()[1:5]≈[0.5+0.0im, 0.321036-0.02204im, 0.0383774-0.094324im, -0.0511445-0.100016im, -0.0164907-0.0230406im] atol=1E-5
    @test redor()[1:5]≈[0.938156+0.0587413im, 0.774492+0.0491454im, 0.547457+0.0340485im, 0.311615+0.0198252im, 0.115145+0.0072829im] atol=1E-5
end

@testset "cuda_examples" begin
    @test rfdr(GPUBatchedMode)[1:5]≈[0.0+0.0im, -0.000304554-9.93079e-9im, -0.00121739-7.93902e-8im, -0.00273603-2.67627e-7im, -0.00485636-6.33332e-7im] atol=1E-5
    @test rfdr(GPUSingleMode)[1:5]≈[0.0+0.0im, -0.000304554-9.93079e-9im, -0.00121739-7.93902e-8im, -0.00273603-2.67627e-7im, -0.00485636-6.33332e-7im] atol=1E-5
end
