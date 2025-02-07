# diffusion_2D_perf_multixpu_hide_comm.jl

const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using ImplicitGlobalGrid
using Printf

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

macro qx(ix, iy) esc(:(-D_dx * (C[$ix+1, $iy+1] - C[$ix, $iy+1]))) end
macro qy(ix, iy) esc(:(-D_dy * (C[$ix+1, $iy+1] - C[$ix+1, $iy]))) end

@parallel_indices (ix, iy) function compute!(C2, C, D_dx, D_dy, dt, _dx, _dy, size_C1_2, size_C2_2)
    if (ix <= size_C1_2 && iy <= size_C2_2)
        C2[ix+1, iy+1] = C[ix+1, iy+1] - dt * ((@qx(ix + 1, iy) - @qx(ix, iy)) * _dx + (@qy(ix, iy + 1) - @qy(ix, iy)) * _dy)
    end
    return nothing
end

@views function diffusion_2D(hide_comm_config::String; nx=64, ny=64, ttot=1.0)
    me, dims = init_global_grid(nx, ny, 1, quiet=true)
    dx, dy  = 10.0 / nx_g(), 10.0 / ny_g()
    dt      = min(dx, dy)^2 / 1.0 / 4.1
    nt      = cld(ttot, dt)

    C  = @zeros(nx, ny)
    C2 = copy(C)
    C .= Data.Array([exp(-(x_g(ix, dx, C) + dx / 2 - 5.0)^2 -
                         (y_g(iy, dy, C) + dy / 2 - 5.0)^2) for ix = 1:size(C, 1), iy = 1:size(C, 2)])

    t_tic = Base.time()
    if hide_comm_config == "no-hidecomm"
        for it = 1:nt
            @parallel compute!(C2, C, 1.0 / dx, 1.0 / dy, dt, 1.0 / dx, 1.0 / dy, size(C, 1) - 2, size(C, 2) - 2)
            C, C2 = C2, C
            update_halo!(C)
        end
    else
        tuple_val = eval(Meta.parse(hide_comm_config))
        for it = 1:nt
            @hide_communication (tuple_val[1], tuple_val[2]) begin
                @parallel compute!(C2, C, 1.0 / dx, 1.0 / dy, dt, 1.0 / dx, 1.0 / dy, size(C, 1) - 2, size(C, 2) - 2)
                C, C2 = C2, C
                update_halo!(C)
            end
        end
    end
    t_toc = Base.time() - t_tic

    finalize_global_grid()
    println(t_toc)  # Ensure only this line is parsed as Float64
end

if length(ARGS) == 3
    hide_comm_config = ARGS[1]
    nx = parse(Int, ARGS[2])
    ny = parse(Int, ARGS[2])
    ttot = parse(Float64, ARGS[3])
    diffusion_2D(hide_comm_config; nx=nx, ny=ny, ttot=ttot)
end
