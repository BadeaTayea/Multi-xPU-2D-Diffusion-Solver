# diffusion_2D_perf_multixpu.jl

const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using ImplicitGlobalGrid
using Plots, Printf
using JLD2  # For saving results

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
    return
end

@views function diffusion_2D(; do_visu=false)
    # Physics
    Lx, Ly = 10.0, 10.0
    D      = 1.0
    ttot   = 1.0
    # Numerics (updated)
    nx, ny = 64, 64
    nout   = 20
    # Initialize global grid
    me, dims = init_global_grid(nx, ny, 1)
    dx, dy  = Lx / nx_g(), Ly / ny_g()
    dt      = min(dx, dy)^2 / D / 4.1
    nt      = cld(ttot, dt)
    D_dx, D_dy = D / dx, D / dy
    _dx, _dy   = 1.0 / dx, 1.0 / dy

    # Use global coordinates for initial conditions
    C = @zeros(nx, ny)
    C .= Data.Array([exp(-(x_g(ix, dx, C) + dx / 2 - Lx / 2)^2 -
                         (y_g(iy, dy, C) + dy / 2 - Ly / 2)^2) for ix = 1:size(C, 1), iy = 1:size(C, 2)])
    C2 = copy(C)

    t_tic = 0.0
    niter = 0

    # Visualization setup
    if do_visu
        if (me == 0) ENV["GKSwstype"] = "nul"; if !isdir("viz2D_mxpu_out") mkdir("viz2D_mxpu_out") end; loadpath = "./viz2D_mxpu_out/"; anim = Animation(loadpath, String[]); println("Animation directory: $(anim.dir)") end
        nx_v, ny_v = (nx - 2) * dims[1], (ny - 2) * dims[2]
        if (nx_v * ny_v * sizeof(Data.Number) > 0.8 * Sys.free_memory()) error("Not enough memory for visualization.") end
        C_v   = zeros(nx_v, ny_v)
        C_inn = zeros(nx - 2, ny - 2)
        xi_g, yi_g = LinRange(dx + dx / 2, Lx - dx - dx / 2, nx_v), LinRange(dy + dy / 2, Ly - dy - dy / 2, ny_v)
    else
        nx_v, ny_v = (nx - 2) * dims[1], (ny - 2) * dims[2]
        C_v = zeros(nx_v, ny_v)  # Ensure initialization even without visualization
    end

    # Time loop
    for it = 1:nt
        if it == 11 t_tic = Base.time(); niter = 0 end
        @hide_communication (8, 2) begin
            @parallel compute!(C2, C, D_dx, D_dy, dt, _dx, _dy, size(C, 1) - 2, size(C, 2) - 2)
            C, C2 = C2, C
            update_halo!(C)
        end
        niter += 1

        if do_visu && (it % nout == 0)
            C_inn .= Array(C)[2:end-1, 2:end-1]; gather!(C_inn, C_v)
            if me == 0
                opts = (aspect_ratio=1, xlims=(xi_g[1], xi_g[end]), ylims=(yi_g[1], yi_g[end]), clims=(0.0, 1.0), c=:turbo, xlabel="Lx", ylabel="Ly", title="time = $(round(it * dt, sigdigits=3))")
                heatmap(xi_g, yi_g, Array(C_v)'; opts...); frame(anim)
            end
        end
    end

    finalize_global_grid()
    t_toc = Base.time() - t_tic
    A_eff = 2 / 1e9 * nx * ny * sizeof(Float64)
    t_it  = t_toc / niter
    T_eff = A_eff / t_it
    @printf("Time = %1.3f sec, T_eff = %1.2f GB/s (niter = %d)\n", t_toc, round(T_eff, sigdigits=3), niter)
    if (do_visu && me == 0) gif(anim, "diffusion_2D_mxpu.gif", fps=5) end

    # Save final result as an Array to avoid CUDA-related loading issues
    if me == 0
        @save "C_output_multixpu.jld2" C_v=Array(C_v)
    end
end

diffusion_2D(; do_visu=true)
