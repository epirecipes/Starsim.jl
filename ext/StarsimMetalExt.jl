"""
    StarsimMetalExt

Extension providing Apple Silicon GPU support via Metal.jl.
"""
module StarsimMetalExt

using Starsim
using Metal
using Distributions

const GPU_THREADS = 256
const GPUVector = MtlVector

gpu_zeros(::Type{T}, n) where {T} = Metal.zeros(T, n)
gpu_fill!(x, v) = Metal.fill!(x, v)
gpu_synchronize() = Metal.synchronize()
gpu_device() = Metal.current_device()
@inline _gpu_thread_index() = thread_position_in_grid_1d()

@inline function _gpu_kernel_launch(groups::Integer, kernel::F, args::Vararg{Any,N}) where {F,N}
    Metal.@metal threads=GPU_THREADS groups=groups kernel(args...)
end

macro gpu_launch(groups, call)
    Meta.isexpr(call, :call) || error("@gpu_launch second argument must be a function call")
    kernel = call.args[1]
    kargs = call.args[2:end]
    return esc(:( $_gpu_kernel_launch($groups, $kernel, $(kargs...)) ))
end

include("StarsimGPUCommon.jl")

Starsim._to_gpu_backend(sim::Starsim.Sim, ::Val{:metal}) = _to_gpu_impl(sim)
Starsim._run_gpu_backend!(sim::Starsim.Sim, ::Val{:metal}; verbose::Int=1, cache_edges::Bool=false) = _run_gpu_impl!(sim; verbose=verbose, cache_edges=cache_edges)

end # module StarsimMetalExt
