using Printf
using DelimitedFiles
using Plots

np_values = [1, 4, 16, 25, 50]  # MPI process counts for weak scaling
optimal_nx = 64 #  
optimal_ny = 64 # 

exec_times = Float64[]  # Stores execution times for each configuration
normalized_times = Float64[]  # Stores normalized execution times

# Baseline run for np = 1
np = np_values[1]
println("Running baseline for np = $np...")

# Compute total grid size for baseline
total_nx = round(Int, optimal_nx * sqrt(np))
total_ny = round(Int, optimal_ny * sqrt(np))

cmd = `srun -n $np julia --project=. diffusion_2D_perf_multixpu_weak_scaling.jl $total_nx $total_ny 0.5`
output = read(cmd, String)

# Parse execution times
lines = filter(!isempty, split(output, "\n"))
exec_times_per_rank = try
    parse.(Float64, lines)
catch e
    error("Failed to parse execution times from output: ", output, "\nError: ", e)
end

T_base = maximum(exec_times_per_rank)
println("Baseline execution time: $T_base seconds")

# Store baseline results
push!(exec_times, T_base)
push!(normalized_times, 1.0)  # Normalized time is 1.0 for baseline

# Remaining runs
for np in np_values[2:end]
    println("Running for np = $np...")
    
    total_nx = round(Int, optimal_nx * sqrt(np))
    total_ny = round(Int, optimal_ny * sqrt(np))
    
    cmd = `srun -n $np julia --project=. diffusion_2D_perf_multixpu_weak_scaling.jl $total_nx $total_ny 0.5`
    output = read(cmd, String)
    
    # Extract execution times
    lines = filter(!isempty, split(output, "\n"))
    exec_times_per_rank = try
        parse.(Float64, lines)
    catch e
        error("Failed to parse execution times from output: ", output, "\nError: ", e)
    end
    
    exec_time = maximum(exec_times_per_rank)
    push!(exec_times, exec_time)
    
    # Normalize execution time
    normalized_time = exec_time / T_base
    push!(normalized_times, normalized_time)
end

# Save results to file
writedlm("weak_scaling_results.txt", hcat(np_values, normalized_times))

# Generate and save plot
plot(
    np_values, 
    normalized_times, 
    xlabel="Number of Processes (np)", 
    ylabel="Normalized Time", 
    title="Weak Scaling Benchmark", 
    marker=:o, 
    legend=false
)
savefig("weak_scaling_plot.png")
