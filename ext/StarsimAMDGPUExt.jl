"""
    StarsimAMDGPUExt

Extension providing AMD GPU support via AMDGPU.jl.
"""
module StarsimAMDGPUExt

using Starsim
using AMDGPU
using Distributions

const GPU_THREADS = 256
const GPUVector = ROCVector

gpu_zeros(::Type{T}, n) where {T} = AMDGPU.zeros(T, n)
gpu_fill!(x, v) = AMDGPU.fill!(x, v)
gpu_synchronize() = AMDGPU.synchronize()
gpu_device() = AMDGPU.device()
@inline _gpu_thread_index() = (workgroupIdx().x - Int32(1)) * workgroupDim().x + workitemIdx().x

macro gpu_launch(groups, call)
    groups_expr = esc(groups)
    call_expr = esc(call)
    return :(@roc groupsize=GPU_THREADS gridsize=GPU_THREADS*$groups_expr $call_expr)
end

include("StarsimGPUCommon.jl")

Starsim._to_gpu_backend(sim::Starsim.Sim, ::Val{:amdgpu}) = _to_gpu_impl(sim)
Starsim._run_gpu_backend!(sim::Starsim.Sim, ::Val{:amdgpu}; verbose::Int=1, cache_edges::Bool=false) = _run_gpu_impl!(sim; verbose=verbose, cache_edges=cache_edges)

end # module StarsimAMDGPUExt
