using JLD2, LinearAlgebra

# Load outputs
multixpu_data = JLD2.jldopen("C_output_multixpu.jld2", "r") do file
    file["C_v"]
end

single_xpu_data = JLD2.jldopen("C_output_gpu_true.jld2", "r") do file
    file["C_final"]
end

# Extract inner points from single xPU data 
single_xpu_inner = single_xpu_data[2:end-1, 2:end-1]

# Ensure dimensions match
if size(multixpu_data) != size(single_xpu_inner)
    println("Dimension mismatch: Multi-xPU size: ", size(multixpu_data), " vs Single-xPU inner size: ", size(single_xpu_inner))
    exit(1)
end

# Compute the normalized difference
diff = norm(multixpu_data .- single_xpu_inner) / norm(single_xpu_inner)
println("Normalized difference between Multi-xPU and Single xPU outputs: ", diff)

println("Multi-xPU data range: ", minimum(multixpu_data), " to ", maximum(multixpu_data))
println("Single xPU data range (inner): ", minimum(single_xpu_inner), " to ", maximum(single_xpu_inner))

println("Sample Multi-xPU: ", multixpu_data[1:5, 1:5])
println("Sample Single xPU: ", single_xpu_inner[1:5, 1:5])
