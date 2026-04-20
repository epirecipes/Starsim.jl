# API Reference

## Simulation

### Core

```@docs
Starsim.Sim
Starsim.SimPars
Starsim.Pars
Starsim.demo
```

### Running

```@docs
Starsim.init!
Starsim.run!
Starsim.reset!
```

### Results access

```@docs
Starsim.to_dataframe
Starsim.get_result
Starsim.all_modules
Starsim.summarize
```

### Serialization

```@docs
Starsim.save_sim
Starsim.load_sim
Starsim.to_json
Starsim.shrink!
```

### Comparison

```@docs
Starsim.diff_sims
Starsim.check_sims_match
Starsim.mock_sim
```

## People

```@docs
Starsim.People
Starsim.init_people!
Starsim.add_module_state!
Starsim.grow!
Starsim.request_death!
Starsim.finish_step!
```

## States and UIDs

### UIDs

```@docs
Starsim.UIDs
Starsim.uids_cat
```

### State vectors

```@docs
Starsim.StateVector
Starsim.FloatState
Starsim.IntState
Starsim.BoolState
Starsim.IndexState
```

### State operations

```@docs
Starsim.init_state!
Starsim.init_vals!
Starsim.set_state!
Starsim.mul_state!
Starsim.uids
Starsim.false_uids
Starsim.state_gt
Starsim.state_lt
Starsim.state_gte
Starsim.state_lte
Starsim.state_eq
Starsim.state_neq
Starsim.state_and_cmp
```

## Time

### Duration

```@docs
Starsim.Duration
Starsim.days
Starsim.weeks
Starsim.months
Starsim.years
Starsim.to_years
Starsim.to_days
Starsim.to_dt
```

### Rate

```@docs
Starsim.Rate
Starsim.perday
Starsim.perweek
Starsim.permonth
Starsim.peryear
Starsim.to_peryear
Starsim.to_prob
```

### Timeline

```@docs
Starsim.Timeline
Starsim.now
Starsim.advance!
Starsim.is_done
```

## Diseases

### Data containers

```@docs
Starsim.DiseaseData
Starsim.InfectionData
```

### Disease models

```@docs
Starsim.SIR
Starsim.SIS
Starsim.SEIR
```

### Disease functions

```@docs
Starsim.infect!
Starsim.validate_beta!
```

## Networks

### Edge storage

```@docs
Starsim.Edges
Starsim.add_edges!
Starsim.remove_edges!
Starsim.clear_edges!
Starsim.find_contacts
```

### Network data

```@docs
Starsim.NetworkData
Starsim.network_data
Starsim.network_edges
Starsim.net_beta
```

### Network types

```@docs
Starsim.RandomNet
Starsim.MFNet
Starsim.MaternalNet
Starsim.MixingPool
Starsim.StaticNet
Starsim.MSMNet
Starsim.PrenatalNet
Starsim.PostnatalNet
Starsim.BreastfeedingNet
Starsim.HouseholdNet
```

### Graph interop

```@docs
Starsim.to_graph
Starsim.to_digraph
Starsim.to_adjacency_matrix
Starsim.to_contact_matrix
Starsim.from_graph!
Starsim.contact_degrees
Starsim.network_components
```

### Network functions

```@docs
Starsim.update_edges!
Starsim.form_partnerships!
Starsim.form_msm_partnerships!
Starsim.add_pairs!
```

## Demographics

```@docs
Starsim.Births
Starsim.Deaths
Starsim.Pregnancy
```

## Interventions

### Products

```@docs
Starsim.Vx
Starsim.Dx
Starsim.Tx
Starsim.administer!
```

### Delivery mechanisms

```@docs
Starsim.RoutineDelivery
Starsim.CampaignDelivery
Starsim.FunctionIntervention
```

### Convenience constructors

```@docs
Starsim.routine_vx
Starsim.campaign_vx
Starsim.simple_vx
Starsim.routine_screening
Starsim.campaign_screening
```

### Intervention data

```@docs
Starsim.InterventionData
Starsim.intervention_data
```

## Connectors

```@docs
Starsim.ConnectorData
Starsim.connector_data
Starsim.Seasonality
Starsim.seasonal_factor
Starsim.CoinfectionConnector
```

## Analyzers

```@docs
Starsim.AnalyzerData
Starsim.analyzer_data
Starsim.FunctionAnalyzer
Starsim.Snapshot
Starsim.InfectionLog
Starsim.TransmissionEvent
```

## Distributions

### Base types

```@docs
Starsim.StarsimDist
Starsim.BernoulliDist
Starsim.ConstantDist
Starsim.ChoiceDist
```

### Distribution operations

```@docs
Starsim.init_dist!
Starsim.jump_dt!
Starsim.rvs
Starsim.set_slots!
```

### CRN multi-random

```@docs
Starsim.MultiRandom
Starsim.combine_rvs
Starsim.multi_rvs
```

### Distribution constructors

```@docs
Starsim.bernoulli
Starsim.ss_random
Starsim.ss_normal
Starsim.ss_lognormal
Starsim.ss_lognormal_im
Starsim.ss_uniform
```

### Distribution container

```@docs
Starsim.DistsContainer
Starsim.register_dist!
Starsim.init_dists!
Starsim.jump_all!
```

## Results

```@docs
Starsim.Result
Starsim.Results
Starsim.scale_results!
```

## Module system

### Module data

```@docs
Starsim.ModuleData
Starsim.module_data
Starsim.module_name
Starsim.module_pars
Starsim.module_results
Starsim.module_states
Starsim.module_timeline
```

### Definition helpers

```@docs
Starsim.define_states!
Starsim.define_results!
Starsim.define_pars!
```

### Lifecycle methods

```@docs
Starsim.init_pre!
Starsim.init_post!
Starsim.init_results!
Starsim.start_step!
Starsim.step!
Starsim.step_state!
Starsim.step_die!
Starsim.update_results!
Starsim.finalize!
```

## Integration loop

```@docs
Starsim.Loop
Starsim.StepEntry
Starsim.build_loop!
Starsim.run_loop!
```

## MultiSim and scenarios

```@docs
Starsim.MultiSim
Starsim.Scenarios
Starsim.ReducedResult
Starsim.reduce!
Starsim.mean!
Starsim.mean_result
Starsim.quantile_result
Starsim.result_keys
```

## Calibration

```@docs
Starsim.CalibPar
Starsim.CalibComponent
Starsim.Calibration
Starsim.apply_pars!
Starsim.compute_objective
Starsim.mse_loss
Starsim.normal_loss
```

## Settings

```@docs
Starsim.Options
Starsim.OPTIONS
Starsim.crn_enabled
Starsim.get_slot_scale
```

## Utilities

```@docs
Starsim.rate_prob
Starsim.prob_rate
Starsim.time_prob
Starsim.standardize_netkey
Starsim.warn_starsim
```

## Abstract type hierarchy

```@docs
Starsim.AbstractModule
Starsim.AbstractInfection
Starsim.AbstractNetwork
Starsim.AbstractDemographics
Starsim.AbstractRoute
Starsim.AbstractAnalyzer
Starsim.AbstractConnector
Starsim.AbstractStarsimDist
```

## Extension stubs

These functions are defined in Starsim and implemented by package extensions.

### ForwardDiff extension

```@docs
Starsim.sensitivity
Starsim.sensitivity_timeseries
Starsim.gradient_objective
```

### Optimization extension

```@docs
Starsim.build_optproblem
Starsim.run_optimization!
```

### Makie extension

```@docs
Starsim.plot_sim
Starsim.plot_disease
Starsim.plot_comparison
```

### Enzyme extension

```@docs
Starsim.enzyme_sensitivity
```

### GPU extension (Metal / CUDA / AMDGPU)

```@docs
Starsim.to_gpu
Starsim.to_cpu
```

GPU execution is selected via the `backend` keyword to `run!`:

```julia
run!(sim; backend=:metal)   # Apple Silicon (requires `using Metal`)
run!(sim; backend=:cuda)    # NVIDIA (requires `using CUDA`)
run!(sim; backend=:amdgpu)  # AMD (requires `using AMDGPU`)
run!(sim; backend=:gpu)     # auto-select if exactly one GPU package is loaded
```

For repeated runs over a static network, pass `cache_edges=true` to
`run_gpu!` to upload edges once instead of per-step.

### Catlab extension

```@docs
Starsim.EpiNet
Starsim.OpenEpiNet
Starsim.EpiSharer
Starsim.compose_epi
Starsim.to_sim
Starsim.epi_uwd
```
