using JLD2
using Plots

# Load the arrays from files
C_cpu = load("C_output_gpu_false.jld2", "C_final")  # CPU output
C_gpu = load("C_output_gpu_true.jld2", "C_final")   # GPU output

# Debugging prints for types and sizes
# Uncomment the lines below to inspect the loaded data types and sizes if needed.
# println("C_gpu type: ", typeof(C_gpu))
# println("C_gpu size: ", size(C_gpu))
# println("C_cpu type: ", typeof(C_cpu))
# println("C_cpu size: ", size(C_cpu))

# Calculate the maximum absolute difference
max_diff = maximum(abs.(C_cpu .- C_gpu))
println("Maximum difference between CPU and GPU outputs: ", max_diff)

# Optional: Visualize the difference
heatmap(abs.(C_cpu .- C_gpu), 
        title="Absolute Difference between GPU and CPU Outputs", 
        color=:turbo, xlabel="x", ylabel="y")

# Save the difference plot
savefig("difference_heatmap.png")
