

function read_file(file)
    particles = zeros(Float64, (100, 3))
    patch1 = zeros(Float64, (100, 3))
    patch2 = zeros(Float64, (100, 3))

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
            number, type, x, y, z = split(s, " ")
            type = parse(Int64, type)
            if type == 1
                particles[i, :] = [parse(Float64, x), parse(Float64, y), parse(Float64, z)]
                iter = 1

            elseif type == 2 && iter == 2
                patch1[i, :] = [parse(Float64, x), parse(Float64, y), parse(Float64, z)]
                iter = 3
                i += 1

            elseif type == 2 && iter == 1
                patch2[i, :] = [parse(Float64, x), parse(Float64, y), parse(Float64, z)]
                iter = 2

            end
        end

    end

    return particles, patch1, patch2
end


particles, patch1, patch2 = read_file("test.lammpstrj")

