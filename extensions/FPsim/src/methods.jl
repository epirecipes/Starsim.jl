"""
Contraceptive method definitions for FPsim.
Matches Python fpsim method switching model with duration-based use,
age-specific switching matrices, and postpartum pathways.
"""

# ============================================================================
# Age bins for method switching (matches Python fpd.method_age_map)
# ============================================================================
const METHOD_AGE_BINS = OrderedDict{String, Tuple{Float64, Float64}}(
    "<18"   => (0.0,  18.0),
    "18-20" => (18.0, 20.0),
    "20-25" => (20.0, 25.0),
    "25-35" => (25.0, 35.0),
    ">35"   => (35.0, 100.0),
)

"""Map an age to a method age bin key."""
function method_age_bin(age::Real)
    for (key, (lo, hi)) in METHOD_AGE_BINS
        if age >= lo && age < hi
            return key
        end
    end
    return ">35"
end

"""Map an age to a 0-based age bin index (for contra use coefficient lookup)."""
function method_age_bin_idx(age::Real)
    i = 0
    for (_, (lo, hi)) in METHOD_AGE_BINS
        if age >= lo && age < hi
            return i
        end
        i += 1
    end
    return length(METHOD_AGE_BINS) - 1
end

# ============================================================================
# Method struct
# ============================================================================

"""
    Method

Represents a single contraceptive method with efficacy, duration distribution,
and switching data. Matches Python fpsim Method class.
"""
struct Method
    name::Symbol
    label::String
    csv_name::String           # Name used in CSV data files
    efficacy::Float64          # Typical-use efficacy (1 - failure rate per year)
    modern::Bool               # Whether this is a modern method
    # Duration distribution parameters
    dur_dist::Symbol           # :lnorm, :weibull, :gamma, :llogis, :exponential, :fixed
    dur_par1::Float64          # Distribution parameter 1 (meanlog/shape/rate)
    dur_par2::Float64          # Distribution parameter 2 (sdlog/scale/rate)
    dur_age_factors::Vector{Float64}  # Age-group adjustment factors for duration
end

"""Default contraceptive methods (generic profile, no switching data)."""
const DEFAULT_METHODS = [
    Method(:none,       "None",        "None",            0.0,   false, :lnorm, 2.115, 0.255, Float64[]),
    Method(:pill,       "Pill",        "Pill",            0.945, true,  :lnorm, 2.359, 0.265, Float64[]),
    Method(:iud,        "IUDs",        "IUD",             0.986, true,  :weibull, 0.479, 3.775, Float64[]),
    Method(:inj,        "Injectables", "Injectable",      0.983, true,  :lnorm, 2.964, 0.157, Float64[]),
    Method(:cond,       "Condoms",     "Condom",          0.946, true,  :lnorm, 2.967, 0.613, Float64[]),
    Method(:btl,        "BTL",         "F.sterilization", 0.995, true,  :fixed, 1000.0, 0.0,  Float64[]),
    Method(:wdraw,      "Withdrawal",  "Withdrawal",      0.866, false, :llogis, 0.764, 2.502, Float64[]),
    Method(:impl,       "Implants",    "Implant",         0.994, true,  :weibull, 0.295, 4.137, Float64[]),
    Method(:othtrad,    "Other traditional", "Other.trad", 0.861, false, :gamma, 0.346, -2.928, Float64[]),
    Method(:othmod,     "Other modern", "Other.mod",      0.88,  true,  :lnorm, 1.786, 0.422, Float64[]),
]

# ============================================================================
# Method switching matrix
# ============================================================================

"""
    MethodSwitchMatrix

Stores the complete method switching probability matrix.
Structure mirrors Python: mcp[pp_state][age_group][from_method] = prob_array

- pp=0: non-postpartum switching (age_group → from_method → prob_array)
- pp=1: 1-month postpartum initiation (age_group → prob_array, no from_method)
- pp=6: 6-month postpartum switching (age_group → from_method → prob_array)

Probability arrays have length n_methods (excluding none), indexed same as method_idx.
"""
struct MethodSwitchMatrix
    method_idx::Vector{Int}   # 1-based method indices in prob array order
    # pp=0: Dict{age_group => Dict{from_method_name => Vector{Float64}}}
    pp0::Dict{String, Dict{Symbol, Vector{Float64}}}
    # pp=1: Dict{age_group => Vector{Float64}}
    pp1::Dict{String, Vector{Float64}}
    # pp=6: Dict{age_group => Dict{from_method_name => Vector{Float64}}}
    pp6::Dict{String, Dict{Symbol, Vector{Float64}}}
end

"""
    ContraUseCoefs

Logistic regression coefficients for probability of contraceptive use.
Matches Python SimpleChoice.get_prob_use().
"""
struct ContraUseCoefs
    intercept::Float64
    age_factors::Vector{Float64}       # Per age bin (0-indexed in Python, length 5)
    fp_ever_user::Float64              # Coefficient for ever-used
    age_ever_user_factors::Vector{Float64}  # Interaction: age × ever-used
end

# ============================================================================
# Method mix — probability distribution over methods for new users
# ============================================================================

"""
    MethodMix

Stores method mixing probabilities for initiating and switching contraception.
"""
struct MethodMix
    method_names::Vector{Symbol}
    mix_probs::Vector{Float64}
    switch_prob::Float64
end

"""Default method mix (roughly based on sub-Saharan Africa patterns)."""
const DEFAULT_METHOD_MIX = MethodMix(
    [:pill, :iud, :inj, :impl, :cond, :wdraw, :othtrad],
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

"""
    sample_from_probs(rng, probs, method_idx_map) → Int

Sample a method using a probability vector and index mapping.
Returns 1-based method index. Used with switching matrix probabilities.
"""
function sample_from_probs(rng::AbstractRNG, probs::Vector{Float64}, method_idx_map::Vector{Int})
    length(probs) == 0 && return 1
    r = rand(rng)
    cum = 0.0
    for i in eachindex(probs)
        cum += probs[i]
        if r < cum
            return method_idx_map[i]
        end
    end
    return method_idx_map[end]
end

"""
    sample_duration(rng, method, age) → Float64

Sample a duration of use (in months) from a method's duration distribution,
with age-group adjustment. Matches Python set_dur_method().
"""
function sample_duration(rng::AbstractRNG, method::Method, age::Real)
    dist = method.dur_dist
    par1 = method.dur_par1
    par2 = method.dur_par2

    # Apply age factor to par1 (scale parameter)
    if !isempty(method.dur_age_factors)
        ai = method_age_bin_idx(age)  # 0-based
        if ai >= 0 && ai < length(method.dur_age_factors)
            par1 += method.dur_age_factors[ai + 1]
        end
    end

    dur = if dist == :lnorm
        # LogNormal: par1=meanlog, par2=sdlog → duration in months
        exp(par1 + par2 * randn(rng))
    elseif dist == :weibull
        # Weibull: par1=shape (log space), par2=scale (log space)
        shape = exp(par1)
        scale = exp(par2)
        d = Weibull(shape, scale)
        rand(rng, d)
    elseif dist == :gamma
        # Gamma: par1=shape (log space), par2=rate (log space)
        shape = exp(par1)
        rate = exp(par2)
        d = Gamma(shape, 1.0 / rate)
        rand(rng, d)
    elseif dist == :llogis
        # Log-logistic (Fisk): par1=shape (log space), par2=scale (log space)
        shape = exp(par1)
        scale = exp(par2)
        # Use inverse CDF: F^{-1}(u) = scale * (u/(1-u))^(1/shape)
        u = rand(rng)
        u = clamp(u, 1e-10, 1.0 - 1e-10)
        scale * (u / (1.0 - u))^(1.0 / shape)
    elseif dist == :fixed
        par1
    else
        12.0  # fallback: 1 year
    end

    return clamp(dur, 1.0, 240.0)  # Clip to [1 month, 20 years]
end

"""
    choose_method_switching(rng, switch_matrix, methods, age, from_method_name, pp_state; jitter=1e-4, method_weights=nothing) → Int

Choose a new method using the switching matrix. Returns 1-based method index.
Matches Python SimpleChoice.choose_method() logic.
"""
function choose_method_switching(rng::AbstractRNG, sm::MethodSwitchMatrix,
                                  methods::Vector{Method}, age::Real,
                                  from_method_name::Symbol, pp_state::Int;
                                  jitter::Float64=1e-4,
                                  method_weights::Union{Nothing, Vector{Float64}}=nothing)
    age_key = method_age_bin(age)

    # Get the probability array
    probs = if pp_state == 1
        # Postpartum 1 month: no from_method dimension
        get(sm.pp1, age_key, nothing)
    elseif pp_state == 6
        age_dict = get(sm.pp6, age_key, nothing)
        age_dict === nothing ? nothing : get(age_dict, from_method_name, nothing)
    else
        age_dict = get(sm.pp0, age_key, nothing)
        age_dict === nothing ? nothing : get(age_dict, from_method_name, nothing)
    end

    probs === nothing && return 1  # fallback to :none

    # Apply jitter to zeros, then apply weights and renormalize
    adjusted = copy(probs)
    for i in eachindex(adjusted)
        if adjusted[i] <= 0.0
            adjusted[i] = abs(jitter * randn(rng))
        end
    end
    if method_weights !== nothing && length(method_weights) == length(adjusted)
        adjusted .*= method_weights
    end
    s = sum(adjusted)
    if s > 0
        adjusted ./= s
    else
        adjusted .= 1.0 / length(adjusted)
    end

    return sample_from_probs(rng, adjusted, sm.method_idx)
end

"""
    compute_prob_use(coefs, age, ever_used; intercept_offset=0.0) → Float64

Compute probability of contraceptive use via logistic regression.
Matches Python SimpleChoice.get_prob_use().
"""
function compute_prob_use(coefs::ContraUseCoefs, age::Real, ever_used::Bool;
                          intercept_offset::Float64=0.0)
    ai = method_age_bin_idx(age)  # 0-based index (0=<18, 1=18-20, 2=20-25, 3=25-35, 4=>35)

    rhs = coefs.intercept
    # Add age factor (5 elements: <18(0), 18-20, 20-25, 25-35, >35)
    if ai >= 0 && ai < length(coefs.age_factors)
        rhs += coefs.age_factors[ai + 1]
    end
    # Add ever-used factor
    if ever_used
        rhs += coefs.fp_ever_user
        # Interaction: applied for ai > 1 (20-25, 25-35, >35)
        # age_ever_user_factors[0..3] = (18-20, 20-25, 25-35, >35)
        # Python accesses [ai-1] with 0-based indexing = Julia ai (1-based)
        if ai > 1 && ai <= length(coefs.age_ever_user_factors)
            rhs += coefs.age_ever_user_factors[ai]
        end
    end

    # Calibration intercept offset
    rhs += intercept_offset

    return 1.0 / (1.0 + exp(-rhs))
end

# ============================================================================
# CSV loading
# ============================================================================

"""Load methods from shared/methods.csv with duration parameters from location data."""
function load_methods_from_csv(shared_path::AbstractString, loc_dir::AbstractString)
    df = CSV.read(shared_path, DataFrame)
    dur_df = nothing
    dur_path = joinpath(loc_dir, "method_time_coefficients.csv")
    if isfile(dur_path)
        dur_df = CSV.read(dur_path, DataFrame; stringtype=String)
    end

    methods = Method[]
    for row in eachrow(df)
        name = Symbol(row.name)
        label = string(row.label)
        csv_name = string(row.csv_name)
        efficacy = Float64(row.efficacy)
        mod_flag = row.modern isa Bool ? row.modern : (uppercase(string(row.modern)) == "TRUE")

        # Duration distribution from method_time_coefficients.csv
        dur_dist = :lnorm
        dur_par1 = 2.0
        dur_par2 = 0.5
        dur_age_factors = Float64[]

        if name == :none
            dur_dist = :lnorm
            dur_par1 = 2.115
            dur_par2 = 0.255
        elseif name == :btl
            dur_dist = :fixed
            dur_par1 = 1000.0
            dur_par2 = 0.0
        end

        if dur_df !== nothing && name != :none && name != :btl
            mdf = dur_df[dur_df.method .== csv_name, :]
            if nrow(mdf) > 0
                func = string(mdf.functionform[1])
                if func in ["lognormal", "lnorm"]
                    dur_dist = :lnorm
                    idx1 = findfirst(mdf.coef .== "meanlog")
                    idx2 = findfirst(mdf.coef .== "sdlog")
                    idx1 !== nothing && (dur_par1 = Float64(mdf.estimate[idx1]))
                    idx2 !== nothing && (dur_par2 = Float64(mdf.estimate[idx2]))
                elseif func == "weibull"
                    dur_dist = :weibull
                    idx1 = findfirst(mdf.coef .== "shape")
                    idx2 = findfirst(mdf.coef .== "scale")
                    idx1 !== nothing && (dur_par1 = Float64(mdf.estimate[idx1]))
                    idx2 !== nothing && (dur_par2 = Float64(mdf.estimate[idx2]))
                elseif func == "gamma"
                    dur_dist = :gamma
                    idx1 = findfirst(mdf.coef .== "shape")
                    idx2 = findfirst(mdf.coef .== "rate")
                    idx1 !== nothing && (dur_par1 = Float64(mdf.estimate[idx1]))
                    idx2 !== nothing && (dur_par2 = Float64(mdf.estimate[idx2]))
                elseif func == "llogis"
                    dur_dist = :llogis
                    idx1 = findfirst(mdf.coef .== "shape")
                    idx2 = findfirst(mdf.coef .== "scale")
                    idx1 !== nothing && (dur_par1 = Float64(mdf.estimate[idx1]))
                    idx2 !== nothing && (dur_par2 = Float64(mdf.estimate[idx2]))
                end

                # Extract age factors
                age_coef_names = ["age_grp_fact(0,18]", "age_grp_fact(18,20]",
                                  "age_grp_fact(20,25]", "age_grp_fact(25,35]",
                                  "age_grp_fact(35,50]"]
                for acn in age_coef_names
                    aidx = findfirst(mdf.coef .== acn)
                    if aidx !== nothing
                        push!(dur_age_factors, Float64(mdf.estimate[aidx]))
                    end
                end
            end
        end

        push!(methods, Method(name, label, csv_name, efficacy, mod_flag,
                              dur_dist, dur_par1, dur_par2, dur_age_factors))
    end
    return methods
end

"""Load method switching matrix from location CSV data."""
function load_switch_matrix(loc_dir::AbstractString, methods::Vector{Method})
    path = joinpath(loc_dir, "method_mix_matrix_switch.csv")
    !isfile(path) && return nothing

    df = CSV.read(path, DataFrame; stringtype=String)

    # Identify method columns (everything except standard cols)
    standard_cols = Set(["postpartum", "From", "n", "age_grp", "group"])
    method_columns = [string(c) for c in names(df) if !(string(c) in standard_cols)]

    # Build method_idx: 1-based indices mapping CSV column order → method index
    csv_to_idx = Dict{String, Int}()
    for (i, m) in enumerate(methods)
        m.name == :none && continue
        csv_to_idx[m.csv_name] = i
    end
    method_idx_arr = Int[get(csv_to_idx, col, 1) for col in method_columns]

    # CSV column name → method name mapping
    csv_to_name = Dict{String, Symbol}()
    for m in methods
        csv_to_name[m.csv_name] = m.name
    end

    pp0 = Dict{String, Dict{Symbol, Vector{Float64}}}()
    pp1 = Dict{String, Vector{Float64}}()
    pp6 = Dict{String, Dict{Symbol, Vector{Float64}}}()

    for row in eachrow(df)
        pp = Int(row.postpartum)
        age_grp = string(row.age_grp)
        from_raw = string(row.From)

        # Extract probability values for method columns
        probs = Float64[Float64(row[Symbol(col)]) for col in method_columns]
        # Ensure non-negative and normalize
        probs = max.(probs, 0.0)
        s = sum(probs)
        s > 0 && (probs ./= s)

        if pp == 1
            pp1[age_grp] = probs
        else
            from_name = if from_raw == "Birth"
                :none
            elseif from_raw == "None"
                :none
            else
                get(csv_to_name, from_raw, :none)
            end

            target = pp == 6 ? pp6 : pp0
            if !haskey(target, age_grp)
                target[age_grp] = Dict{Symbol, Vector{Float64}}()
            end
            target[age_grp][from_name] = probs
        end
    end

    return MethodSwitchMatrix(method_idx_arr, pp0, pp1, pp6)
end

"""Load contraceptive use coefficients from location CSV files."""
function load_contra_use_coefs(loc_dir::AbstractString)
    files = [
        joinpath(loc_dir, "contra_coef_simple.csv"),
        joinpath(loc_dir, "contra_coef_simple_pp1.csv"),
        joinpath(loc_dir, "contra_coef_simple_pp6.csv"),
    ]

    coefs = ContraUseCoefs[]
    for f in files
        if isfile(f)
            df = CSV.read(f, DataFrame; stringtype=String)
            intercept = 0.0
            age_factors = Float64[]
            fp_ever_user = 0.0
            age_ever_user_factors = Float64[]

            for row in eachrow(df)
                rhs = string(row.rhs)
                est = Float64(row.Estimate)
                if occursin("Intercept", rhs)
                    intercept = est
                elseif occursin("age_grp", rhs) && occursin("prior_userTRUE", rhs)
                    push!(age_ever_user_factors, est)
                elseif occursin("age_grp", rhs)
                    push!(age_factors, est)
                elseif occursin("prior_userTRUE", rhs)
                    fp_ever_user = est
                end
            end

            # Prepend 0 for the <18 age group (reference category)
            pushfirst!(age_factors, 0.0)
            push!(coefs, ContraUseCoefs(intercept, age_factors, fp_ever_user, age_ever_user_factors))
        end
    end

    return coefs
end

"""Load method mix from a mix.csv file."""
function load_method_mix_from_csv(path::AbstractString, methods::Vector{Method})
    df = CSV.read(path, DataFrame)
    label_to_name = Dict{String, Symbol}()
    for m in methods
        label_to_name[m.label] = m.name
        label_to_name[string(m.name)] = m.name
        label_to_name[m.csv_name] = m.name
    end
    extra_map = Dict(
        "Implants" => :impl, "Injectables" => :inj, "Condoms" => :cond,
        "IUDs" => :iud, "Pill" => :pill, "BTL" => :btl,
        "Withdrawal" => :wdraw, "Other traditional" => :othtrad,
        "Other modern" => :othmod,
    )
    for (k, v) in extra_map
        label_to_name[k] = v
    end

    names_out = Symbol[]
    probs = Float64[]
    for row in eachrow(df)
        label = string(row.method)
        name = get(label_to_name, label, nothing)
        if name !== nothing && name != :none
            push!(names_out, name)
            push!(probs, Float64(row.perc) / 100.0)
        end
    end
    s = sum(probs)
    s > 0 && (probs ./= s)
    return MethodMix(names_out, probs, 0.7)
end

"""
    load_methods(; location::Symbol=:generic) → Vector{Method}

Load contraceptive methods for a location.
"""
function load_methods(; location::Symbol=:generic)
    if location == :generic
        return copy(DEFAULT_METHODS)
    else
        shared_path = joinpath(DATA_DIR, "shared", "methods.csv")
        loc_dir = joinpath(DATA_DIR, string(location))
        if isfile(shared_path) && isdir(loc_dir)
            return load_methods_from_csv(shared_path, loc_dir)
        elseif isfile(shared_path)
            return load_methods_from_csv(shared_path, "")
        else
            return copy(DEFAULT_METHODS)
        end
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
