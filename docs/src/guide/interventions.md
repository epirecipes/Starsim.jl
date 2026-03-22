# Interventions

## Products

Products define what happens when an intervention is applied:

```julia
vx = Vx(efficacy=0.9)   # Vaccine: reduces susceptibility by 90%
dx = Dx(sensitivity=0.8)  # Diagnostic test
tx = Tx(efficacy=0.95)    # Treatment
```

## Delivery mechanisms

### RoutineDelivery

Ongoing delivery to a fraction of eligible agents each timestep:

```julia
routine = RoutineDelivery(product=vx, prob=0.05, disease_name=:sir)
```

### CampaignDelivery

One-time delivery at specific years:

```julia
campaign = CampaignDelivery(product=vx, prob=0.8, years=[100.0], disease_name=:sir)
```

## Example

```julia
sim = Sim(
    n_agents = 5000,
    networks = RandomNet(n_contacts=10),
    diseases = SIR(beta=0.1, dur_inf=10.0, init_prev=0.01),
    interventions = [RoutineDelivery(product=Vx(efficacy=0.9), prob=0.02, disease_name=:sir)],
    dt = 1.0,
    stop = 365.0,
)
run!(sim)
```
