

function read_file(file)
    particle = zeros(Float64, (100, 3))
    patch1 = zeros(Float64, (100, 3))
    patch2 = zeros(Float64, (100, 3))

    mu_particle = zeros(Float64, (100, 3))
    mu_patch1 = zeros(Float64, (100, 3))
    mu_patch2 = zeros(Float64, (100, 3))

    open(file) do f

        # line_number
        line = 0
        iter = 1
        i = 1
        # read till end of file
        while !eof(f)

            # read a new / next line for every iteration           
            s = readline(f)
            line += 1
            number, type, x, y, z, mux, muy, muz = split(s, " ")
            type = parse(Int64, type)
            if type == 1
                particle[i, :] = [parse(Float64, x), parse(Float64, y), parse(Float64, z)]
                mu_particle[i, :] = [parse(Float64, mux), parse(Float64, muy), parse(Float64, muz)]
                iter = 1

            elseif type == 2 && iter == 2
                patch1[i, :] = [parse(Float64, x), parse(Float64, y), parse(Float64, z)]
                mu_patch1[i, :] = [parse(Float64, mux), parse(Float64, muy), parse(Float64, muz)]
                iter = 3
                i += 1

            elseif type == 2 && iter == 1
                patch2[i, :] = [parse(Float64, x), parse(Float64, y), parse(Float64, z)]
                mu_patch2[i, :] = [parse(Float64, mux), parse(Float64, muy), parse(Float64, muz)]
                iter = 2

            end
        end

    end

    return particle, patch1, patch2, mu_particle, mu_patch1, mu_patch2
end

particle, patch1, patch2, mp, m1, m2 = read_file("mu_3dtest.txt")

boxL = [20, 20, 20]
Np = len(particles)

D11 = zeros(Float64,(Np,Np,3))
D22 = zeros(Float64,(Np,Np,3))
D12 = zeros(Float64,(Np,Np,3))
D21 = zeros(Float64,(Np,Np,3))

D11_norm = eros(Float64,(Np,Np))
D22_norm = zeros(Float64,(Np,Np))
D12_norm = zeros(Float64,(Np,Np))
D21_norm = zeros(Float64,(Np,Np))

for i = 1:Np
    for j = 1:Np
        if i<j
            dist11 = patch1[i] - patch1[j]
            dist11 = dist11 - boxL * rint(dist11/boxL) 
            D11[i,j] = dist11
            D11[j,i] = -dist11 

            D11_norm[i,j] = sqrt(dist11[1]^2 + dist11[2]^2 + dist11[3]^2)
            D11_norm[j,i] = D11_norm[i,j]
            
            dist22 = patch2[i] - patch2[j]
            dist22 = dist22 - boxL * rint(dist22/boxL) 
            D22[i,j] = dist22
            D22[j,i] = -dist22

            D22_norm[i,j] = sqrt(dist22[1]^2 + dist22[2]^2 + dist22[3]^2)
            D22_norm[j,i] = D22_norm[i,j]

            dist12 = patch1[i] - patch2[j]
            dist12 = dist12 - boxL * rint(dist12/boxL) 
            D12[i,j] = dist12
            D12[j,i] = -dist12

            D12_norm[i,j] = sqrt(dist12[1]^2 + dist12[1]^2 + dist12[3]^2)
            D12_norm[j,i] = D12_norm[i,j]

            dist21 = patch2[i] - patch1[j]
            dist21 = dist21 - boxL * rint(dist21/boxL) 
            D21[i,j] = dist21
            D21[j,i] = -dist21

            D21_norm[i,j] = sqrt(dist21[1]^2 + dist21[1]^2 + dist21[3]^2)
            D21_norm[j,i] = D21_norm[i,j]

        end 
    end
end

function U_dipolar(i, j, Dist, mi, mj)

    dij = Dist[i,j]

    dnorm = sqrt(dij[1]^2 + dij[2]^2 + dij[3]^2)
    Udij = -3 * dot(mi, dij) * dot(mj, dij) / dnorm^5 + dot(mi, mj) / dnorm^3
    return Udij
end

function U_polar(i)
    Udi = 0 
    cut_off = 10 

    for j in 1:Np
        if i!=j
            # patch-i-1-patch-j-1 
            if D11_norm[i,j] < cut_off
                Udi += U_dipolar(i,j,D11,m1[i],m1[j])
            end

            # patch-i-2-patch-j-2
            if D22_norm[i,j] < cut_off
                Udi += U_dipolar(i,j,D22,m2[i],m2[j])
            end

            # patch-i-1-patch-j-2 
            if D12_norm[i,j] < cut_off
                Udi += U_dipolar(i,j,D12,m1[i],m2[j])
            end

            # patch-i-2-patch-j-1 
            if D21_norm[i,j] < cut_off
                Udi += U_dipolar(i,j,D21,m1[j],m2[i])
            end

        end 

    return Udi 
end