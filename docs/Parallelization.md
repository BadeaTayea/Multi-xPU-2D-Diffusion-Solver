The **parallelization strategy** followed in this project leverages a **multi-xPU approach** using a combination of distributed computing with `ImplicitGlobalGrid.jl` and GPU acceleration using `ParallelStencil.jl`. Key techniques employed are discussed below:



**1. Initialization for Multi-xPU Execution**

The solver begins by defining whether to use GPU or multi-threaded CPU parallelization:

```julia
const USE_GPU = true
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
```

- **`@init_parallel_stencil(CUDA, Float64, 2)`** initializes `ParallelStencil.jl` for 2D parallel computations on **CUDA GPUs** (if `USE_GPU = true`) or **multi-threaded CPUs** (if `false`).

- This initialization also configures the low-level parallel backend and vectorizes operations, ensuring the solver can efficiently scale on multiple GPUs or CPU threads. The **2D grid dimension** is declared here to inform the parallel backend of how data is partitioned and processed.

---

**2. Domain Partitioning and Grid Initialization**

Domain decomposition and partitioning across processors (xPUs) are critical for distributed memory parallelization. The grid is globally partitioned using:

```julia
me, dims = init_global_grid(nx, ny, 1)
```

- **`init_global_grid(nx, ny, 1)`** splits the 2D grid among processes (MPI-like behavior) in both dimensions, where `nx` and `ny` are the grid dimensions.
  - `me` identifies the current processor’s rank.
  - `dims` specifies how the domain is decomposed along the x and y dimensions.

Global grid-based decomposition ensures that each subdomain handles only a portion of the full grid (`nx_g()` and `ny_g()` give local dimensions).

---

**3. Setting the Numerical Scheme and Initial Conditions**

The **numerical discretization** follows the finite difference method (FDM), with the 2D Laplacian represented by partial derivatives. Important grid parameters include:

```julia
dx, dy  = Lx / nx_g(), Ly / ny_g()
dt      = min(dx, dy)^2 / D / 4.1
D_dx, D_dy = D / dx, D / dy
_dx, _dy   = 1.0 / dx, 1.0 / dy
```

- The spatial step sizes `dx` and `dy` are determined by dividing the domain length by the grid points in each direction.
- The **time step `dt`** is computed using the **stability criterion for explicit methods**, ensuring the numerical scheme remains stable.
- **Finite differences for derivatives** are precomputed using `D_dx` and `D_dy` to optimize flux calculations during iterations.

Initial conditions are set using the **global grid coordinates**:
```julia
C .= Data.Array([exp(-(x_g(ix, dx, C) + dx / 2 - Lx / 2)^2 -
                     (y_g(iy, dy, C) + dy / 2 - Ly / 2)^2) for ix = 1:size(C, 1), iy = 1:size(C, 2)])
```

- **`x_g` and `y_g`** provide the global coordinates for each grid point.
- A **Gaussian initial condition** is set, representing a concentrated heat distribution at the center of the domain.

This global coordinate mapping ensures that each subdomain initializes its local grid based on its global location, supporting distributed computations.

---

**4. Kernel Implementation: Parallelized Finite Difference Updates**

The core computation is performed using the kernel function:

```julia
@parallel_indices (ix, iy) function compute!(C2, C, D_dx, D_dy, dt, _dx, _dy, size_C1_2, size_C2_2)
    if (ix <= size_C1_2 && iy <= size_C2_2)
        C2[ix+1, iy+1] = C[ix+1, iy+1] - dt * ((@qx(ix + 1, iy) - @qx(ix, iy)) * _dx +
                                               (@qy(ix, iy + 1) - @qy(ix, iy)) * _dy)
    end
    return
end
```

This function updates the grid in parallel across the x and y dimensions:
- **Parallel execution:** The **`@parallel_indices (ix, iy)`** macro parallelizes the nested loops, ensuring grid updates occur concurrently on available GPUs or threads.
- **Flux macros:** The finite difference fluxes are efficiently computed using the macros:
  ```julia
  macro qx(ix, iy) esc(:(-D_dx * (C[$ix+1, $iy+1] - C[$ix, $iy+1]))) end
  macro qy(ix, iy) esc(:(-D_dy * (C[$ix+1, $iy+1] - C[$ix+1, $iy]))) end
  ```
  These macros perform partial derivative calculations using **precomputed differences** to optimize performance and reduce memory accesses.

- **Flux divergence:** The grid is updated using the flux divergence formula:
  ```julia
  C2[ix+1, iy+1] = C[ix+1, iy+1] - dt * (qx_divergence + qy_divergence)
  ```

This approach **minimizes overhead** by directly computing flux differences without looping or nested conditions.

---

**5. Halo Exchange and Communication Overlap**

Communication between neighboring subdomains occurs via **halo updates**. To ensure consistency across subdomains, boundary data is exchanged after each iteration:

```julia
@hide_communication (8, 2) begin
    @parallel compute!(C2, C, D_dx, D_dy, dt, _dx, _dy, size(C, 1) - 2, size(C, 2) - 2)
    C, C2 = C2, C
    update_halo!(C)
end
```

- The `@hide_communication (8, 2)` macro overlaps communication (halo exchange) with computation, reducing idle times.
- The **`update_halo!(C)`** function ensures that boundary grid values are exchanged with neighboring subdomains.

This communication model reduces synchronization overhead and maximizes parallel efficiency.

---

**6. Visualization and Gathering Results**

After each time step, the visualization (if enabled) consolidates subdomain results:

```julia
C_inn .= Array(C)[2:end-1, 2:end-1]
gather!(C_inn, C_v)
```

- **`gather!` collects local subdomain data** into the global grid (`C_v`) for visualization.
- **The master process (`me == 0`)** handles visualization tasks, ensuring computational processes are not blocked:
  ```julia
  if me == 0
      heatmap(xi_g, yi_g, Array(C_v)'; opts...)
      frame(anim)
  end
  ```

This decoupling of visualization from computation ensures the solver remains efficient during large-scale simulations.

---

**7. Performance Monitoring: Effective Bandwidth**

To measure parallel performance, the script computes the **effective memory bandwidth (`T_eff`)**:

```julia
A_eff = 2 / 1e9 * nx * ny * sizeof(Float64)
t_it  = t_toc / niter
T_eff = A_eff / t_it
@printf("Time = %1.3f sec, T_eff = %1.2f GB/s (niter = %d)\n", t_toc, round(T_eff, sigdigits=3), niter)
```

- **`A_eff`** measures the memory transfer requirements per iteration (two arrays per grid point).
- **`T_eff` evaluates how efficiently memory is utilized**, serving as an indicator for bottlenecks or opportunities for optimization.

---

**8. Final Result Saving**

The solver saves the final result using **JLD2**:
```julia
if me == 0
    @save "C_output_multixpu.jld2" C_v=Array(C_v)
end
```

This ensures that distributed subdomains are properly gathered and stored for post-processing.

