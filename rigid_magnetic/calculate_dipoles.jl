using LinearAlgebra
using ArgParse

# read file for 2 patch systems 
function read_file_two_patch(file, Np)

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

        # skip first nine lines
        for i in 1:9
            readline(f)
        end

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

# read file for one patch systems 
function read_file_one_patch(file, Np)

    particle = zeros(Float64, (Np, 3))
    patch1 = zeros(Float64, (Np, 3))

    mu_particle = zeros(Float64, (Np, 3))
    mu_patch1 = zeros(Float64, (Np, 3))

    open(file) do f

        # line_number
        line = 0
        i = 1
        # skip first nine lines
        for i in 1:9
            s = readline(f)
        end

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


            elseif type == 2
                patch1[i, :] = [parse(Float64, x), parse(Float64, y), parse(Float64, z)]
                mu_patch1[i, :] = [parse(Float64, mux), parse(Float64, muy), parse(Float64, muz)]
                i += 1

            end
        end

    end

    return particle, patch1, mu_particle, mu_patch1
end

# calculate pair distances between patches 
function dist(p1, p2, boxL, ndim)

    if ndim == 3
        Lbc = boxL
    end

    if ndim == 2
        Lbc = [boxL[1], boxL[2], 1000]
    end

    dist = p1 .- p2
    dist = dist .- boxL .* round.(dist ./ Lbc)
    dnorm = sqrt(dist[1]^2 + dist[2]^2 + dist[3]^2)

    return dist, dnorm

end


# calculate dipole-dipole pair potential 
function Ud_ij(dij, dij_norm, mi, mj)
    Udij = (-3 * dot(mi, dij) * dot(mj, dij) / dij_norm^5) + (dot(mi, mj) / dij_norm^3)
    return Udij

end

# write file 
function write_file(U, Np, file)

    open("DIPOLE_" * file, "w") do f
        for i in 1:Np
            println(f, U[i])
        end
    end
end


function dipole_two_patch(Np, boxL, file, ndim)

    particle, patch1, patch2, mp, m1, m2 = read_file_two_patch(file, Np)

    D11 = zeros(Float64, (Np, Np, 3))
    D22 = zeros(Float64, (Np, Np, 3))
    D12 = zeros(Float64, (Np, Np, 3))
    D21 = zeros(Float64, (Np, Np, 3))

    D11_norm = zeros(Float64, (Np, Np))
    D22_norm = zeros(Float64, (Np, Np))
    D12_norm = zeros(Float64, (Np, Np))
    D21_norm = zeros(Float64, (Np, Np))

    for i = 1:Np
        for j = 1:Np
            if i < j

                D11[i, j, :], D11_norm[i, j] = dist(patch1[i, :], patch1[j, :], boxL, ndim)
                D11[j, i, :] = D11[j, i, :]
                D11_norm[j, i] = D11_norm[i, j]

                D22[i, j, :], D22_norm[i, j] = dist(patch2[i, :], patch2[j, :], boxL, ndim)
                D22[j, i, :] = D22[j, i, :]
                D22_norm[j, i] = D22_norm[i, j]

                D12[i, j, :], D12_norm[i, j] = dist(patch1[i, :], patch2[j, :], boxL, ndim)
                D12[j, i, :] = D12[j, i, :]
                D12_norm[j, i] = D12_norm[i, j]

                D21[i, j, :], D21_norm[i, j] = dist(patch2[i, :], patch1[j, :], boxL, ndim)
                D21[j, i, :] = D21[j, i, :]
                D21_norm[j, i] = D21_norm[i, j]

            end
        end
    end

    function Ud_i_two_patch(i)

        Udi = 0
        cut_off = 10

        for j in 1:Np
            if i != j
                # patch-i-1-patch-j-1 
                if D11_norm[i, j] < cut_off
                    Udi += Ud_ij(D11[i, j, :], D11_norm[i, j], m1[i, :], m1[j, :])
                end

                # patch-i-2-patch-j-2
                if D22_norm[i, j] < cut_off
                    Udi += Ud_ij(D22[i, j, :], D22_norm[i, j], m2[i, :], m2[j, :])
                end

                # patch-i-1-patch-j-2 
                if D12_norm[i, j] < cut_off
                    Udi += Ud_ij(D12[i, j, :], D12_norm[i, j], m1[i, :], m2[j, :])
                end

                # patch-i-2-patch-j-1 
                if D21_norm[i, j] < cut_off
                    Udi += Ud_ij(D21[i, j, :], D21_norm[i, j], m1[j, :], m2[i, :])
                end

            end
        end
        return Udi
    end

    U = zeros(Float64, Np)
    for i in 1:Np
        U[i] = Ud_i_two_patch(i)

    end

    write_file(U, Np, file)
end

function dipole_one_patch(Np, boxL, file, ndim)

    particle, patch1, mp, m1 = read_file_one_patch(file, Np)
    D11 = zeros(Float64, (Np, Np, 3))
    D11_norm = zeros(Float64, (Np, Np))
    print
    for i = 1:Np
        for j = 1:Np
            if i < j

                D11[i, j, :], D11_norm[i, j] = dist(patch1[i, :], patch1[j, :], boxL, ndim)
                D11[j, i, :] = -D11[j, i, :]
                D11_norm[j, i] = D11_norm[i, j]

            end
        end
    end

    function Ud_i_one_patch(i)

        Udi = 0
        cut_off = 10

        for j in 1:Np
            if i != j
                # patch-i-1-patch-j-1 
                if D11_norm[i, j] < cut_off
                    Udi += Ud_ij(D11[i, j, :], D11_norm[i, j], m1[i, :], m1[j, :])
                end
            end
        end
        return Udi
    end

    U = zeros(Float64, Np)
    for i in 1:Np
        U[i] = Ud_i_one_patch(i)
    end

    write_file(U, Np, file)
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "arg1"
        help = "file input"
        required = true
    end
    return parse_args(s)
end


function main()

    args = parse_commandline()
    file = args["arg1"]
    Np = 200
    boxL = [20, 20, 20]
    npatch = 2
    ndim = 3

    if npatch == 1
        dipole_one_patch(Np, boxL, file, ndim)
    end

    if npatch == 2
        dipole_two_patch(Np, boxL, file, ndim)
    end

end

main()

