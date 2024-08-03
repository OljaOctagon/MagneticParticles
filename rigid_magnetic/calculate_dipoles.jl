

function read_file(file)
    Np = 200
    particle = zeros(Float64, (Np, 3))
    patch1 = zeros(Float64, (Np, 3))
    patch2 = zeros(Float64, (Np, 3))

    mu_particle = zeros(Float64, (Np, 3))
    mu_patch1 = zeros(Float64, (Np, 3))
    mu_patch2 = zeros(Float64, (Np, 3))

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
Np = 200

D11 = zeros(Float64, (Np, Np, 3))
D22 = zeros(Float64, (Np, Np, 3))
D12 = zeros(Float64, (Np, Np, 3))
D21 = zeros(Float64, (Np, Np, 3))

D11_norm = zeros(Float64, (Np, Np))
D22_norm = zeros(Float64, (Np, Np))
D12_norm = zeros(Float64, (Np, Np))
D21_norm = zeros(Float64, (Np, Np))

function dist(p1, p2)

    dist = p1 .- p2
    dist = dist .- boxL .* round.(dist ./ boxL)
    dnorm = sqrt(dist[1]^2 + dist[2]^2 + dist[3]^2)

    return dist, dnorm

end

for i = 1:Np
    for j = 1:Np
        if i < j

            D11[i, j, :], D11_norm[i, j] = dist(patch1[i, :], patch1[j, :])
            D11[j, i, :] = D11[j, i, :]
            D11_norm[j, i] = D11_norm[i, j]

            D22[i, j, :], D22_norm[i, j] = dist(patch2[i, :], patch2[j, :])
            D22[j, i, :] = D22[j, i, :]
            D22_norm[j, i] = D22_norm[i, j]

            D12[i, j, :], D12_norm[i, j] = dist(patch1[i, :], patch2[j, :])
            D12[j, i, :] = D12[j, i, :]
            D12_norm[j, i] = D12_norm[i, j]

            D21[i, j, :], D21_norm[i, j] = dist(patch2[i, :], patch1[j, :])
            D21[j, i, :] = D21[j, i, :]
            D21_norm[j, i] = D21_norm[i, j]

        end
    end
end

function Ud_ij(i, j, dij, dij_norm, mi, mj)

    Udij = -3 * dot(mi, dij) * dot(mj, dij) / dij_norm^5 + dot(mi, mj) / dij_norm^3
    return Udij

end

function Ud_i(i)

    Udi = 0
    cut_off = 10

    for j in 1:Np
        if i != j
            # patch-i-1-patch-j-1 
            if D11_norm[i, j] < cut_off
                Udi += Ud_ij(i, j, D11, m1[i, :], m1[j, :])
            end

            # patch-i-2-patch-j-2
            if D22_norm[i, j] < cut_off
                Udi += Ud_ij(i, j, D22, m2[i], m2[j])
            end

            # patch-i-1-patch-j-2 
            if D12_norm[i, j] < cut_off
                Udi += Ud_ij(i, j, D12, m1[i], m2[j])
            end

            # patch-i-2-patch-j-1 
            if D21_norm[i, j] < cut_off
                Udi += Ud_ij(i, j, D21, m1[j], m2[i])
            end

        end
    end
    return Udi
end

U = zeros(Float64, Np)
for i in 1:Np
    U[i] = Ud_i(i)

end

print(U)