# JuMP Problem Examples

This folder contains 7 diverse optimization problem examples demonstrating different problem types, decision variable structures, and parameter sweeps.

## List of Examples

| File | Problem Type | Key Swept Parameters | Features |
|---|---|---|---|
| `01_knapsack.jl` | 0-1 Knapsack MILP | `n_items`, `capacity_ratio`, `seed` | Binary selection, capacity constraint |
| `02_facility_location.jl` | Capacitated Facility Location MILP | `n_facilities`, `fixed_cost_scale`, `seed` | Facility opening + routing optimization |
| `03_portfolio_optimization.jl` | Portfolio Risk-Return LP | `risk_budget`, `max_asset_weight`, `seed` | Continuous allocation, risk budget constraint |
| `04_radial_doctor_scheduling.jl` | Spatial Doctor Scheduling | `segments`, `seed` | Multi-segment subproblem parallelization (`solve_subproblems_parallel`) |
| `05_supply_chain_network.jl` | Multi-Echelon Logistics Network | `transport_multiplier`, `n_dcs`, `seed` | Flow conservation, multi-stage network |
| `06_job_shop_scheduling.jl` | Disjunctive Job Shop Scheduling | `n_jobs`, `n_machines`, `time_variance`, `seed` | Makespan minimization, disjunctive machine overlap constraints |
| `07_multi_objective_blend.jl` | Material Blending Multi-Objective LP | `quality_target`, `emission_weight`, `seed` | Multi-objective weighted-sum optimization |

## How to Use an Example

To use any example as your active problem:

1. Copy the desired example to `problem.jl` at the project root:
   ```bash
   cp examples/01_knapsack.jl problem.jl
   ```

2. Run the sweep or distributed runner:
   ```bash
   # Run full parameter sweep
   julia --project=. main.jl

   # Run distributed range sweep
   julia --project=. main.jl --range 1:5 --format sqlite --out results.db

   # Run multi-rank MPI execution
   mpirun -np 4 julia --project=. main.jl --use-mpi --format json
   ```
