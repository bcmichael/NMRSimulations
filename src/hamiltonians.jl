const wigner2_elements = [x->0.25*(1+cosd(x))^2       x->0.5*sind(x)*(1+cosd(x))      x->sqrt(3/8)*sind(x)^2  x->0.5*sind(x)*(1-cosd(x))      x->0.25*(1-cosd(x))^2;
                          x->-0.5*sind(x)*(1+cosd(x)) x->0.5*(2*cosd(x)^2+cosd(x)-1)  x->sqrt(3/8)*sind(2*x)  x->0.5*(-2*cosd(x)^2+cosd(x)+1) x->0.5*sind(x)*(1-cosd(x));
                          x->sqrt(3/8)*sind(x)^2      x->-sqrt(3/8)*sind(2*x)         x->0.5*(3*cosd(x)^2-1)  x->sqrt(3/8)*sind(2*x)          x->sqrt(3/8)*sind(x)^2;
                          x->-0.5*sind(x)*(1-cosd(x)) x->0.5*(-2*cosd(x)^2+cosd(x)+1) x->-sqrt(3/8)*sind(2*x) x->0.5*(2*cosd(x)^2+cosd(x)-1)  x->0.5*sind(x)*(1+cosd(x));
                          x->0.25*(1-cosd(x))^2       x->-0.5*sind(x)*(1-cosd(x))     x->sqrt(3/8)*sind(x)^2  x->-0.5*sind(x)*(1+cosd(x))     x->0.25*(1+cosd(x))^2]

"""
    Z(T<:AbstractArray{Complex{<:AbstractFloat}})

Generate a Z pauli matrix of type 'T'.
"""
Z(::Type{T}) where {T1<:AbstractFloat, T<:AbstractArray{Complex{T1}}} = T(Complex{T1}.([1 0*im; 0 -1]/2))

"""
    X(T<:AbstractArray{Complex{<:AbstractFloat}})

Generate a X pauli matrix of type 'T'.
"""
X(::Type{T}) where {T1<:AbstractFloat, T<:AbstractArray{Complex{T1}}} = T(Complex{T1}.([0*im 1; 1 0]/2))

"""
    Y(T<:AbstractArray{Complex{<:AbstractFloat}})

Generate a Y pauli matrix of type 'T'.
"""
Y(::Type{T}) where {T1<:AbstractFloat, T<:AbstractArray{Complex{T1}}} = T(Complex{T1}.([0 -1*im; 1*im 0]/2))

"""
    euler_rotation(tensor, angles)

Rotate 'tensor' by 'angles' and return the rotated tensor.
"""
function euler_rotation(tensor::SphericalTensor, angles::EulerAngles{T}) where {T}
    s00 = tensor.s00.*Complex{T}(1)
    s20 = rotate_component2(tensor, 0, angles)
    s21 = rotate_component2(tensor, 1, angles)
    s2m1 = rotate_component2(tensor, -1, angles)
    s22 = rotate_component2(tensor, 2, angles)
    s2m2 = rotate_component2(tensor, -2, angles)
    new_tensor = SphericalTensor(s00, s20, s21, s2m1, s22, s2m2)
    return new_tensor
end

"""
    rotate_component2(tensor, component, angles)

Generate a single second rank tensor 'component' of the result of rotating
'tensor' by 'angles'.
"""
rotate_component2(tensor::SphericalTensor, component, angles::EulerAngles{T}) where {T} =
    tensor.s20.*wigner2(component, 0, angles).+
    tensor.s21.*wigner2(component, 1, angles).+
    tensor.s2m1.*wigner2(component, -1, angles).+
    tensor.s22.*wigner2(component, 2, angles).+
    tensor.s2m2.*wigner2(component, -2, angles)

rotate_component2(tensor::SphericalTensor{A}, component, angles::EulerAngles{T}) where {T,A<:HilbertOperator} =
    A(tensor.s20.data.*wigner2(component, 0, angles).+
    tensor.s21.data.*wigner2(component, 1, angles).+
    tensor.s2m1.data.*wigner2(component, -1, angles).+
    tensor.s22.data.*wigner2(component, 2, angles).+
    tensor.s2m2.data.*wigner2(component, -2, angles))

"""
    wigner2(new, old, angles)

Generate the wigner rotation element connecting the second rank tensor
components 'old' and 'new' when rotating by 'angles'.
"""
wigner2(new, old, angles::EulerAngles{T}) where {T} =
    exp(-im*(old*angles.α+new*angles.γ)*pi/180) * T((wigner2_elements[new+3, old+3](angles.β)::Float64))

"""
    kron_up(input, position, total)

Use a kronecker product with identity matrices to expand a single spin operator
'input' into a product operator basis containing 'total' spins. 'position'
specifies which spin 'input' corresponds to.
"""
function kron_up(input::AbstractArray{Complex{T},2}, position, total) where {T<:AbstractFloat}
    out = kron(Array{T}(I, 2^(position-1), 2^(position-1)), input)
    out = kron(out, Array{T}(I, 2^(total-position), 2^(total-position)))
end

"""
    kron_double(in1, in2, position1, position2, total)

Use a kronecker product with identity matrices to create a product of two single
spin operators ('in1','in2') into a product operator basis containing 'total' spins.
'position1' and 'position2' specify which spins the operators correspond to.
"""
function kron_double(in1::T2, in2::T2, position1, position2, total) where {T1,T2<:AbstractArray{Complex{T1},2}}
    if position1 > position2
        position1, position2 = position2, position1
        in1, in2 = in2, in1
    elseif position1 == position2
        throw(ArgumentError("kron_double must be between operators for different spins"))
    end

    out = kron(Array{T1}(I, 2^(position1-1), 2^(position1-1)), in1)
    out = kron(out, Array{T1}(I, 2^(position2-position1-1), 2^(position2-position1-1)))
    out = kron(out, in2)
    out = kron(out, Array{T1}(I, 2^(total-position2), 2^(total-position2)))
    return out
end

"""
    initial_cs(spins, T1=Array)

Generate the chemical shift hamiltonian for 'spins'.
"""
function initial_cs(spins::Vector{Spin{T}}, ::Type{T1}=Array) where {T,T1<:AbstractArray}
    number = size(spins)[1]
    total_size = 2^number
    cs = SphericalTensor(0, 0, 0, 0, 0, 0) * T1(zeros(Complex{T}, (total_size, total_size)))
    z = Z(T1{Complex{T}})
    for i = 1:number
        iso = spins[i].sigma_iso
        anis = spins[i].anisotropy
        asym = spins[i].asymmetry
        angles = spins[i].angles
        tensor = SphericalTensor{Complex{T}}(iso, anis, 0.0, 0.0, sqrt(2/3)*0.5*anis*asym, sqrt(2/3)*0.5*anis*asym)
        tensor = euler_rotation(tensor, angles)
        cs = cs+tensor*kron_up(z, i, number)
    end
    return cs
end

"""
    pulse_H(γB1, xyz)

Generate the rf hamiltonian for a pulse with the given 'γB1' and set of 'xyz'
operators.
"""
function pulse_H(γB1, xyz)
    H = fill!(similar(xyz[1]), 0)
    for n = 1:length(γB1)
        H .+= xyz[3*n-2].*γB1[n]*1000
    end
    return H
end

"""
    phase_rotator(phases, xyz)

Generate an array that can rotate a hamiltonian or propagator of a pulse with
phase 0 to 'phases'
"""
function phase_rotator(phases::NTuple{N,T}, xyz) where {T,N}
    dim = size(xyz[3])[1]
    rotation_diagonal = zeros(Complex{T}, dim)
    for n = 1:length(phases)
        phase = phases[n]
        z = xyz[3*n]
        for j = 1:dim
            rotation_diagonal[j] += phase*z[j, j]
        end
    end
    rotation_diagonal .= exp.(rotation_diagonal.*im.*pi./180)
    element_wise_rotator = rotation_diagonal.*rotation_diagonal'
    return element_wise_rotator
end

"""
    channel_XYZ(spins, channels, x, y, z)

Generate an x, y, and z operator for each channel.
"""
function channel_XYZ(spins::Vector{Spin{T}}, channels, x::T1, y::T1, z::T1) where {T,T1<:AbstractArray{Complex{T},2}}
    nspins = length(spins)
    out = Vector{T1}()
    for n = 1:3*channels
        push!(out, T1(zeros(Complex{T}, (2^nspins, 2^nspins))))
    end

    for n = 1:nspins
        out[spins[n].channel*3-2] += kron_up(x, n, nspins)
        out[spins[n].channel*3-1] += kron_up(y, n, nspins)
        out[spins[n].channel*3] += kron_up(z, n, nspins)
    end
    return out
end

"""
    dipole_coupling(spins, s1, s2, strength, T1=Array)

Generate a hamiltonian for a dipole coupling between spins 's1' and 's2'.
"""
function dipole_coupling(spins::Vector{Spin{T}}, s1, s2, strength, ::Type{T1}=Array) where {T,T1<:AbstractArray}
    s1 != s2 || throw(ArgumentError("Coupling must be between different spins"))

    if s1 > s2
        s1 , s2 = s2, s1
    end

    x = X(T1{Complex{T}})
    y = Y(T1{Complex{T}})
    z = Z(T1{Complex{T}})

    nspins = length(spins)
    matrix = 2*kron_double(z,z,s1,s2,nspins)
    if spins[s1].channel == spins[s2].channel
        matrix -= kron_double(x, x, s1, s2, nspins) + kron_double(y, y, s1, s2, nspins)
    end

    d=SphericalTensor{T}(0.0, strength, 0.0, 0.0, 0.0, 0.0)*matrix
    return d
end

"""
    j_coupling(spins, s1, s2, iso, aniso, asym, T1=Array)

Generate a hamiltonian for a j coupling between spins 's1' and 's2'.
"""
function j_coupling(spins::Vector{Spin{T}}, s1, s2, iso, aniso, asym, ::Type{T1}=Array) where {T,T1<:AbstractArray}
    s1 != s2 || throw(ArgumentError("Coupling must be between different spins"))

    if s1 > s2
        s1, s2 = s2, s1
    end

    x = X(T1{Complex{T}})
    y = Y(T1{Complex{T}})
    z = Z(T1{Complex{T}})

    nspins = length(spins)
    matrix0 = kron_double(z, z, s1, s2, nspins)
    matrix2 = 2*matrix0
    if spins[s1].channel == spins[s2].channel
        matrix = kron_double(x, x, s1, s2, nspins) + kron_double(y, y, s1, s2, nspins)
        matrix0 += matrix
        matrix2 -= matrix
    end

    j = SphericalTensor{T}(iso, 0, 0, 0, 0, 0)*matrix0+
        SphericalTensor{T}(0, 0.5*aniso, 0, 0, 0.5/sqrt(6)*aniso*asym, 0.5/sqrt(6)*aniso*asym)*matrix2
    return j
end

magic_angle(::Type{T}) where {T<:AbstractFloat} = EulerAngles{T}(0, acosd(sqrt(1/3)), 0)
