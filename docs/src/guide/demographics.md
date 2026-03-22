# Demographics

## Births

Add births with a crude birth rate (per 1000 per year):

```julia
births = Births(birth_rate=20.0)
```

New agents are added via `grow!` with age 0 and randomly assigned sex.

## Deaths

Add background mortality with a crude death rate (per 1000 per year):

```julia
deaths = Deaths(death_rate=10.0)
```

Deaths are processed via `request_death!` and `step_die!` in the integration loop.

## Combined

```julia
sim = Sim(
    n_agents = 5000,
    networks = RandomNet(n_contacts=10),
    diseases = SIR(beta=0.05, dur_inf=10.0),
    demographics = [Births(birth_rate=20.0), Deaths(death_rate=10.0)],
    dt = 1.0,
    stop = 365.0,
)
run!(sim)
```
