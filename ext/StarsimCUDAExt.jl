"""
    StarsimCUDAExt

Extension providing NVIDIA GPU support via CUDA.jl.
"""
module StarsimCUDAExt

using Starsim
using CUDA
using Distributions

const GPU_THREADS = 256
const GPUVector = CuVector

gpu_zeros(::Type{T}, n) where {T} = CUDA.zeros(T, n)
gpu_fill!(x, v) = CUDA.fill!(x, v)
gpu_synchronize() = CUDA.synchronize()
gpu_device() = CUDA.device()
@inline _gpu_thread_index() = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

macro gpu_launch(groups, call)
    groups_expr = esc(groups)
    call_expr = esc(call)
    return :(@cuda threads=GPU_THREADS blocks=$groups_expr $call_expr)
end

include("StarsimGPUCommon.jl")

Starsim._to_gpu_backend(sim::Starsim.Sim, ::Val{:cuda}) = _to_gpu_impl(sim)
Starsim._run_gpu_backend!(sim::Starsim.Sim, ::Val{:cuda}; verbose::Int=1, cache_edges::Bool=false) = _run_gpu_impl!(sim; verbose=verbose, cache_edges=cache_edges)

end # module StarsimCUDAExt
