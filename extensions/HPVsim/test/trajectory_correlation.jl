# HPVsim trajectory correlation: run Julia, load Python trajectories, compute Pearson r.
using HPVsim
using Starsim
using JSON
using Statistics
using Printf

n_seeds = 10
n_agents = 5000

# Run Julia sims
julia_results = Dict{String, Vector{Vector{Float64}}}()

for seed in 1:n_seeds
    sim = HPVSim(
        n_agents=n_agents,
        genotypes=[:hpv16],
        start=2000.0,
        stop=2050.0,
        dt=1.0,
        rand_seed=seed,
        verbose=0,
    )
    Starsim.run!(sim)

    # Extract disease results
    for (dname, disease) in sim.diseases
        md = Starsim.module_data(disease)
        for (rname, result) in md.results
            key = string(rname)
            vals = Float64.(result.values)
            if !haskey(julia_results, key)
                julia_results[key] = Vector{Vector{Float64}}()
            end
            push!(julia_results[key], vals)
        end
    end

    # Extract sim-level results
    for (rname, result) in sim.results.data
        key = string(rname)
        vals = Float64.(result.values)
        if !haskey(julia_results, key)
            julia_results[key] = Vector{Vector{Float64}}()
        end
        push!(julia_results[key], vals)
    end
    println("  Seed $seed: $(length(julia_results)) series")
end

# Compute Julia means
julia_means = Dict{String, Vector{Float64}}()
for (k, arrays) in julia_results
    if length(arrays) == n_seeds
        mat = hcat(arrays...)
        julia_means[k] = vec(mean(mat, dims=2))
    end
end

# Load Python means
py_data = JSON.parsefile(joinpath(@__DIR__, "python_trajectories.json"))

# Map Julia keys to Python keys
key_mapping = Dict(
    "prevalence" => "hpv_prevalence",
    "n_susceptible" => "n_susceptible",
    "n_infected" => "n_infected",
    "new_infections" => "hpv_incidence",
    "n_alive" => "n_alive",
)

# Pearson correlation
function pearson_r(x::Vector{Float64}, y::Vector{Float64})
    n = min(length(x), length(y))
    x, y = x[1:n], y[1:n]
    mx, my = mean(x), mean(y)
    sx, sy = std(x), std(y)
    (sx == 0 || sy == 0) && return NaN
    return cor(x, y)
end

println("\n" * "="^75)
println("HPVsim Trajectory Correlation: Julia vs Python (10 seeds, 5000 agents)")
println("="^75)
@printf("%-25s | %-8s | %-12s | %-12s | %s\n",
        "Variable", "Pearson r", "Julia range", "Python range", "Status")
println("-"^75)

global matched = 0
global total = 0

for (jkey, pkey) in sort(collect(key_mapping), by=x->x[1])
    if haskey(julia_means, jkey) && haskey(py_data, pkey)
        jmean = julia_means[jkey]
        pmean = Float64.(py_data[pkey]["mean"])

        r = pearson_r(jmean, pmean)
        jrange = @sprintf("%.1f–%.1f", minimum(jmean), maximum(jmean))
        prange = @sprintf("%.1f–%.1f", minimum(pmean), maximum(pmean))
        status = r > 0.9 ? "✓ STRONG" : r > 0.7 ? "~ MODERATE" : "✗ WEAK"
        if r > 0.7; global matched += 1; end
        global total += 1
        @printf("%-25s | %8.4f | %-12s | %-12s | %s\n",
                "$jkey → $pkey", r, jrange, prange, status)
    end
end

# Also check ALL Python keys against ALL Julia keys for best matches
println("\n--- Auto-matched trajectories (best correlation) ---")
@printf("%-25s | %-25s | %-8s | %s\n", "Julia key", "Python key", "Pearson r", "Status")
println("-"^75)

for jkey in sort(collect(keys(julia_means)))
    best_r = -1.0
    best_pkey = ""
    jmean = julia_means[jkey]
    for (pkey, pinfo) in py_data
        pmean = Float64.(pinfo["mean"])
        r = pearson_r(jmean, pmean)
        if !isnan(r) && r > best_r
            best_r = r
            best_pkey = pkey
        end
    end
    status = best_r > 0.9 ? "✓ STRONG" : best_r > 0.7 ? "~ MOD" : "✗ WEAK"
    @printf("%-25s | %-25s | %8.4f | %s\n", jkey, best_pkey, best_r, status)
end

println("\n$matched/$total mapped variables have r > 0.7")
