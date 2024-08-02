

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

boxL = [20, 20, 20]

particle = particle * boxL
patch1 = patch1 * boxL
patch2 = patch2 * boxL

# THIS is all wrong! needs to be a matrix between patches 1 among each other and patches 2 among each other.
# cutoff can be though of at this point 
dist_patch12 = patch1 - patch2
dist_patch12 = dist_patch12 - boxL * rint(dist_patch12 / boxL)

function U_dipolar(i, j, patch1, patch2, mu_patch1, mu_patch2, dist_patch12)
    rnorm = sqrt(dist_patch12[1]^2 + dist_patch12[2]^2 + dist_patch12[3]^2)
    Udij = -3 * dot(mu_patch1, dist_patch12) * dot(mu_patch2, dist_patch12) / rnorm^5 - dot(mu_patch1, mu_patch2) / rnorm^3

end


particle, patch1, patch2, mu_particle, mu_patch1, mu_patch2 = read_file("mu_3dtest.txt")

