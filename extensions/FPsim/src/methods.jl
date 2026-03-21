"""
Contraceptive method definitions for FPsim.
"""

"""
    Method

Represents a single contraceptive method with efficacy and discontinuation data.
"""
struct Method
    name::Symbol
    label::String
    efficacy::Float64          # Typical-use efficacy (1 - failure rate per year)
    discontinuation::Float64   # Annual discontinuation rate
    modern::Bool               # Whether this is a modern method
end

"""Default contraceptive methods (generic profile)."""
const DEFAULT_METHODS = [
    Method(:none, "None", 0.0, 0.0, false),
    Method(:pill, "Pill", 0.91, 0.30, true),
    Method(:iud, "IUD", 0.99, 0.10, true),
    Method(:injectable, "Injectable", 0.94, 0.25, true),
    Method(:implant, "Implant", 0.995, 0.05, true),
    Method(:condom, "Condom", 0.82, 0.40, true),
    Method(:withdrawal, "Withdrawal", 0.78, 0.50, false),
    Method(:traditional, "Traditional", 0.70, 0.40, false),
]

# ============================================================================
# Method mix — probability distribution over methods for new users
# ============================================================================

"""
    MethodMix

Stores method mixing probabilities for initiating and switching contraception.
"""
struct MethodMix
    method_names::Vector{Symbol}
    mix_probs::Vector{Float64}    # Probability of each method for new users (sums to 1)
    switch_prob::Float64          # Probability of switching (vs stopping) on discontinuation
end

"""Default method mix (roughly based on sub-Saharan Africa patterns)."""
const DEFAULT_METHOD_MIX = MethodMix(
    [:pill, :iud, :injectable, :implant, :condom, :withdrawal, :traditional],
    [0.10, 0.05, 0.35, 0.30, 0.10, 0.05, 0.05],
    0.7,
)

# ============================================================================
# Helpers
# ============================================================================

"""Find a method by name; returns first method (:none) if not found."""
function method_by_name(methods::Vector{Method}, name::Symbol)
    idx = findfirst(m -> m.name == name, methods)
    idx === nothing && return methods[1]
    return methods[idx]
end

"""Find 1-based index of a method by name; 0 if not found."""
function method_index(methods::Vector{Method}, name::Symbol)
    idx = findfirst(m -> m.name == name, methods)
    return idx === nothing ? 0 : idx
end

"""
    sample_method(rng, methods, mix) → Int

Sample a method index (1-based) from the method mix.
"""
function sample_method(rng::AbstractRNG, methods::Vector{Method}, mix::MethodMix)
    r = rand(rng)
    cum = 0.0
    for (i, name) in enumerate(mix.method_names)
        cum += mix.mix_probs[i]
        if r < cum
            idx = findfirst(m -> m.name == name, methods)
            return idx === nothing ? 1 : idx
        end
    end
    return 1
end

# ============================================================================
# CSV loading
# ============================================================================

"""Load methods from shared/methods.csv."""
function load_methods_from_csv(path::AbstractString)
    df = CSV.read(path, DataFrame)
    methods = Method[Method(:none, "None", 0.0, 0.0, false)]
    for row in eachrow(df)
        row.name == "none" && continue
        mod_flag = row.modern isa Bool ? row.modern : (uppercase(string(row.modern)) == "TRUE")
        push!(methods, Method(
            Symbol(row.name),
            string(row.label),
            Float64(row.efficacy),
            0.20,
            mod_flag,
        ))
    end
    return methods
end

"""Load method mix from a mix.csv file."""
function load_method_mix_from_csv(path::AbstractString, methods::Vector{Method})
    df = CSV.read(path, DataFrame)
    label_to_name = Dict{String, Symbol}()
    for m in methods
        label_to_name[m.label] = m.name
        label_to_name[string(m.name)] = m.name
    end
    # Also map common CSV labels
    extra_map = Dict(
        "Implants" => :impl, "Injectables" => :inj, "Condoms" => :cond,
        "IUDs" => :iud, "Pill" => :pill, "BTL" => :btl,
        "Withdrawal" => :wdraw, "Other traditional" => :othtrad,
        "Other modern" => :othmod,
    )
    for (k, v) in extra_map
        label_to_name[k] = v
    end

    names = Symbol[]
    probs = Float64[]
    for row in eachrow(df)
        label = string(row.method)
        name = get(label_to_name, label, nothing)
        if name !== nothing && name != :none
            push!(names, name)
            push!(probs, Float64(row.perc) / 100.0)
        end
    end
    s = sum(probs)
    s > 0 && (probs ./= s)
    return MethodMix(names, probs, 0.7)
end

"""
    load_methods(; location::Symbol=:generic) → Vector{Method}

Load contraceptive methods for a location.
"""
function load_methods(; location::Symbol=:generic)
    if location == :generic
        return copy(DEFAULT_METHODS)
    else
        path = joinpath(DATA_DIR, "shared", "methods.csv")
        isfile(path) || return copy(DEFAULT_METHODS)
        return load_methods_from_csv(path)
    end
end

"""
    load_method_mix(; location::Symbol=:generic, methods) → MethodMix

Load method mix for a location.
"""
function load_method_mix(; location::Symbol=:generic,
                          methods::Vector{Method}=DEFAULT_METHODS)
    if location == :generic
        return DEFAULT_METHOD_MIX
    else
        path = joinpath(DATA_DIR, string(location), "mix.csv")
        isfile(path) || return DEFAULT_METHOD_MIX
        return load_method_mix_from_csv(path, methods)
    end
end
