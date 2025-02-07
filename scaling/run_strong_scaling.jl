# run_strong_scaling.jl

using Printf
using DelimitedFiles
using Plots

nx_values = 16 .* 2 .^ (1:8)
T_eff_results = Float64[]

for nx in nx_values
    println("Running for nx = $nx...")
    cmd = `srun -n 1 julia --project=. diffusion_2D_perf_multixpu_benchmarking.jl $nx $nx 0.5`
    output = read(cmd, String)
    
    # Extract lines and filter out empty or irrelevant ones
    lines = split(output, "\n")
    lines = filter(!isempty, lines)  # Remove empty lines

    # Parse T_eff from the last meaningful line
    T_eff = parse(Float64, lines[end])
    push!(T_eff_results, T_eff)
end

# Save results to file
writedlm("strong_scaling_results.txt", hcat(nx_values, T_eff_results))

# Plot
plot(nx_values, T_eff_results, xlabel="Grid size (nx = ny)", ylabel="T_eff (GB/s)", title="Strong Scaling Benchmark", marker=:o, legend=false)
savefig("strong_scaling_plot.png")
