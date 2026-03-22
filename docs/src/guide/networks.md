# Networks

## Built-in network types

### RandomNet

Random contact network with `n_contacts` effective contacts per agent.

```julia
net = RandomNet(n_contacts=10)
```

Edges are regenerated every timestep. Creates `n_contacts÷2` edges per agent; transmission runs bidirectionally.

### StaticNet

Fixed network from an Erdős-Rényi graph or custom generator.

```julia
net = StaticNet(n_contacts=6)
```

### MFNet — Male-Female

Heterosexual partnership network.

```julia
net = MFNet(mean_dur=5.0, participation_rate=0.5)
```

### MSMNet — Men who have sex with men

```julia
net = MSMNet(participation_rate=0.3)
```

### MixingPool

Well-mixed (mass action) contacts.

```julia
net = MixingPool(contact_rate=10.0)
```

### MaternalNet

Mother-child network for vertical transmission. Managed by the Pregnancy module.

## Graphs.jl interop

Convert edges to a Graphs.jl graph for analysis:

```julia
g = to_graph(network_edges(net))
deg = degree(g)
components = connected_components(g)
```

## Matrix representation

Get a sparse adjacency matrix:

```julia
A = to_adjacency_matrix(network_edges(net); weighted=true)
```
