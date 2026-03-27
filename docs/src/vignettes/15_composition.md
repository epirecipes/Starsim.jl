# Category-Theory Composition with Catlab.jl
Simon Frost

- [Overview](#overview)
- [Setup](#setup)
- [Epidemiological networks (EpiNet)](#epidemiological-networks-epinet)
- [EpiNet to simulation](#epinet-to-simulation)
- [Module composition with
  EpiSharers](#module-composition-with-episharers)
- [Undirected wiring diagrams](#undirected-wiring-diagrams)
- [Composing a simulation](#composing-a-simulation)
- [Verification: composed matches
  manual](#verification-composed-matches-manual)
- [Multi-module composition](#multi-module-composition)
- [Open epidemiological networks](#open-epidemiological-networks)
- [Why category theory?](#why-category-theory)

## Overview

One challenge in agent-based disease modeling is composing multiple
modules (diseases, networks, demographics) into a coherent simulation
without introducing subtle bugs. Starsim.jl provides a
**category-theory-based composition** framework via the `Catlab.jl`
extension, allowing you to formally specify how modules share state and
then automatically assemble a simulation.

This approach uses **undirected wiring diagrams** (UWDs) from applied
category theory: each module is a “box” with ports representing shared
agent states (e.g., `:alive`, `:susceptible`), and junctions wire
compatible ports together.

## Setup

``` julia
using Starsim
using Catlab
```

## Epidemiological networks (EpiNet)

An `EpiNet` is an ACSet (attributed C-set) that formally represents the
transition structure of a disease model. Each state (S, I, R, etc.) is
an object, and each transition (infection, recovery) is a morphism from
source to target state.

``` julia
# Define an SIR transition structure
sir_net = EpiNet(
    [:S, :I, :R],
    [:infection => (:S => :I), :recovery => (:I => :R)]
)
println(sir_net)
```

    StarsimCatlabExt.EpiNetACSet{Symbol}:
      S = 1:3
      T = 1:2
      Name = 1:0
      src : T → S = [1, 2]
      tgt : T → S = [2, 3]
      sname : S → Name = [:S, :I, :R]
      tname : T → Name = [:infection, :recovery]

This is equivalent to the textbook SIR compartmental diagram: **S → I →
R**, where infection moves agents from susceptible to infected, and
recovery moves them from infected to recovered.

``` julia
# SIS has recovery looping back to susceptible
sis_net = EpiNet(
    [:S, :I],
    [:infection => (:S => :I), :recovery => (:I => :S)]
)
println(sis_net)
```

    StarsimCatlabExt.EpiNetACSet{Symbol}:
      S = 1:2
      T = 1:2
      Name = 1:0
      src : T → S = [1, 2]
      tgt : T → S = [2, 1]
      sname : S → Name = [:S, :I]
      tname : T → Name = [:infection, :recovery]

``` julia
# SEIR adds an exposed compartment
seir_net = EpiNet(
    [:S, :E, :I, :R],
    [:exposure => (:S => :E), :progression => (:E => :I), :recovery => (:I => :R)]
)
println(seir_net)
```

    StarsimCatlabExt.EpiNetACSet{Symbol}:
      S = 1:4
      T = 1:3
      Name = 1:0
      src : T → S = [1, 2, 3]
      tgt : T → S = [2, 3, 4]
      sname : S → Name = [:S, :E, :I, :R]
      tname : T → Name = [:exposure, :progression, :recovery]

## EpiNet to simulation

An `EpiNet` can be directly converted to a Starsim simulation via
`to_sim()`, which auto-detects the disease pattern:

``` julia
sim = to_sim(sir_net;
    n_agents = 1000,
    beta = 0.5 / 10,
    dur_inf = 4.0,
    networks = RandomNet(n_contacts=10),
    start = 0.0,
    stop = 40.0,
    rand_seed = 42,
)
run!(sim; verbose=0)
prev = get_result(sim, :sir, :prevalence)
println("Peak prevalence from EpiNet-based sim: $(round(maximum(prev), digits=3))")
```

    Peak prevalence from EpiNet-based sim: 0.112

## Module composition with EpiSharers

For more complex models, `EpiSharer` wraps a Starsim module and declares
which agent states it shares as “ports”. When two sharers declare the
same port (e.g., `:alive`), composing them ensures they operate on the
same population.

``` julia
# Create modules
n_contacts = 10
beta = 0.5 / n_contacts
sir = SIR(beta=beta, dur_inf=4.0, init_prev=0.01)
net = RandomNet(n_contacts=n_contacts)

# Wrap as sharers
disease_sharer = EpiSharer(:sir_disease, sir)
network_sharer = EpiSharer(:contact_net, net)

println(disease_sharer)
println(network_sharer)
```

    EpiSharer(:sir_disease, 1 module, ports=[:alive, :susceptible], category=:disease)
    EpiSharer(:contact_net, 1 module, ports=[:alive], category=:network)

Disease sharers automatically expose `:alive` and `:susceptible` ports.
Network sharers expose `:alive`.

## Undirected wiring diagrams

The wiring diagram specifies how sharers connect via shared junctions:

``` julia
# Build a UWD connecting our sharers
uwd = epi_uwd([disease_sharer, network_sharer])
println("Boxes: ", nparts(uwd, :Box))
println("Junctions: ", nparts(uwd, :Junction))
```

    Boxes: 2
    Junctions: 2

You can also build UWDs by creating multiple sharers:

``` julia
# Create sharers with specific configurations
disease_sharer_2 = EpiSharer(:sir_disease_2, SIR(beta=beta, dur_inf=4.0))
network_sharer_2 = EpiSharer(:contact_net_2, RandomNet(n_contacts=n_contacts))

# Build UWD from sharers
uwd_manual = epi_uwd([disease_sharer_2, network_sharer_2])
println("Manual UWD boxes: ", nparts(uwd_manual, :Box))
```

    Manual UWD boxes: 2

## Composing a simulation

`compose_epi` takes a vector of sharers and assembles them into a
complete simulation:

``` julia
sim_composed = compose_epi(
    [disease_sharer, network_sharer];
    n_agents = 1000,
    start = 0.0,
    stop = 40.0,
    rand_seed = 42,
)
run!(sim_composed; verbose=0)
prev_composed = get_result(sim_composed, :sir, :prevalence)
println("Peak prevalence (composed): $(round(maximum(prev_composed), digits=3))")
```

    Peak prevalence (composed): 0.112

## Verification: composed matches manual

The key guarantee of categorical composition is that the composed model
produces identical results to a manually assembled simulation:

``` julia
# Manual assembly
sim_manual = Sim(
    n_agents = 1000,
    diseases = SIR(beta=beta, dur_inf=4.0, init_prev=0.01),
    networks = RandomNet(n_contacts=n_contacts),
    start = 0.0,
    stop = 40.0,
    rand_seed = 42,
)
run!(sim_manual; verbose=0)
prev_manual = get_result(sim_manual, :sir, :prevalence)

max_diff = maximum(abs.(prev_composed .- prev_manual))
println("Maximum difference: $max_diff")
println("Match: $(max_diff < 1e-10)")
```

    Maximum difference: 0.0
    Match: true

The composed and manual simulations are **identical** — same disease
dynamics, same random number streams, same results. This is the
functoriality guarantee: composition preserves behavior.

## Multi-module composition

Composition becomes most useful when assembling complex models with many
interacting components:

``` julia
# Three-component model: disease + network + demographics
sir2 = SIR(beta=beta, dur_inf=4.0, init_prev=0.01)
net2 = RandomNet(n_contacts=n_contacts)
births = Births(birth_rate=20.0)  # per 1000 per year
deaths = Deaths(death_rate=15.0)

disease_s = EpiSharer(:disease, sir2)
network_s = EpiSharer(:network, net2)
demog_s = EpiSharer(:demographics, [births, deaths])

sim3 = compose_epi(
    [disease_s, network_s, demog_s];
    n_agents = 1000,
    start = 0.0,
    stop = 40.0,
    rand_seed = 42,
)
run!(sim3; verbose=0)
println("Final population: ", get_result(sim3, :n_alive)[end])
```

    Final population: 1221.0

## Open epidemiological networks

For hierarchical composition, `OpenEpiNet` creates structured cospans
that expose specific disease states as ports for gluing:

``` julia
net_data = EpiNet(
    [:S, :I, :R],
    [:infection => (:S => :I), :recovery => (:I => :R)]
)
# Expose S (index 1) as an external port
open_net = OpenEpiNet(net_data, [[1]])
println("OpenEpiNet legs: ", length(open_net.legs))
```

    OpenEpiNet legs: 1

This enables compositional design where disease models can be connected
at shared states — for example, two diseases sharing a susceptible
population through a connector.

## Why category theory?

The categorical approach provides:

1.  **Formal correctness**: Shared states are identified at composition
    time — no silent mismatches or forgotten connections.
2.  **Modularity**: Each component is self-contained with explicit
    interfaces.
3.  **Compositionality**: Complex models are built by composing simpler
    ones, with guarantees that the composition preserves each
    component’s behavior.
4.  **Visual reasoning**: Wiring diagrams give an intuitive graphical
    representation of model structure.

This follows the same philosophy as
[AlgebraicJulia](https://www.algebraicjulia.org/) and the applied
category theory approach to scientific modeling.
