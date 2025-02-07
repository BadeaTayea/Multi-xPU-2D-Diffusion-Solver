# run_hide_comm_benchmarking.jl

using Printf
using DelimitedFiles
using Plots

configs = ["no-hidecomm", "(2,2)", "(8,2)", "(16,4)", "(16,16)"]
np = 64  # Using 64 MPI processes
optimal_nx = 64  
ttot = 0.5  # Adjust to ensure time loop is short

exec_times = Float64[]
T_base = nothing  # Placeholder for normalization

# First execution for baseline
config = configs[1]
println("Running for configuration: $config...")

cmd = `srun -n $np julia --project=. diffusion_2D_perf_multixpu_hide_comm.jl $config $optimal_nx $ttot`
output = read(cmd, String)
lines = filter(!isempty, split(output, "\n"))

if isempty(lines)
    error("No output from diffusion_2D_perf_multixpu_hide_comm.jl. Check the script.")
end

T_base = try
    parse(Float64, lines[end])
catch e
    error("Failed to parse execution time from output: ", output, "\nError: ", e)
end

println("Baseline execution time: $T_base seconds")
push!(exec_times, T_base)

# Remaining configurations
for config in configs[2:end]
    println("Running for configuration: $config...")
    
    cmd = `srun -n $np julia --project=. diffusion_2D_perf_multixpu_hide_comm.jl $config $optimal_nx $ttot`
    output = read(cmd, String)
    lines = filter(!isempty, split(output, "\n"))

    if isempty(lines)
        error("No output from diffusion_2D_perf_multixpu_hide_comm.jl. Check the script.")
    end

    exec_time = try
        parse(Float64, lines[end])
    catch e
        error("Failed to parse execution time from output: ", output, "\nError: ", e)
    end

    push!(exec_times, exec_time)
end

# Normalize execution times
normalized_times = exec_times ./ T_base

# Save results
writedlm("hide_comm_results.txt", hcat(1:length(configs), normalized_times))

# Plot
plot(1:length(configs), normalized_times, xlabel="Configuration Index", ylabel="Normalized Execution Time", 
     title="Hide Communication Benchmark", xticks=(1:5, configs), marker=:o, legend=false)
savefig("hide_comm_plot.png")
