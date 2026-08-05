# Example 6: Job Shop Scheduling Problem
#
# Schedules N jobs across M machines with processing times and sequence precedence constraints.
include("../core/base_problem.jl")

using JuMP
using Random

mutable struct JobShopProblem <: LPProblem
    n_jobs::Int
    n_machines::Int
    time_variance::Float64
    seed::Int
    params::Dict{String,Any}
    model::Union{Nothing,JuMP.Model}
    variables::Dict{String,Any}
    solve_time_s::Union{Nothing,Float64}
end

JobShopProblem(; n_jobs = 4, n_machines = 3, time_variance = 1.0, seed = 0) = JobShopProblem(
    n_jobs, n_machines, time_variance, seed,
    Dict{String,Any}("n_jobs" => n_jobs, "n_machines" => n_machines, "time_variance" => time_variance, "seed" => seed),
    nothing, Dict{String,Any}(), nothing
)

const PARAM_RANGES = Dict{String,Any}(
    "n_jobs" => [3, 4, 5],
    "n_machines" => [2, 3],
    "time_variance" => [0.8, 1.0, 1.5],
    "seed" => [0, 1],
)

function build(p::JobShopProblem)
    rng = MersenneTwister(p.seed)
    J = p.n_jobs
    M_cnt = p.n_machines

    # Random processing times P[j, m]
    P = rand(rng, 1:10, J, M_cnt) .* p.time_variance
    BIG_M = 1000.0

    model = Model()
    @variable(model, start[1:J, 1:M_cnt] >= 0)     # Start time of job j on machine m
    @variable(model, Cmax >= 0)                     # Makespan (completion time)
    @variable(model, z[1:J, 1:J, 1:M_cnt], Bin)     # Precedence binary for machine overlap avoidance

    # Job sequence precedence (Machine m must finish before Machine m+1 starts for job j)
    for j in 1:J, m in 1:(M_cnt - 1)
        @constraint(model, start[j, m + 1] >= start[j, m] + P[j, m])
    end

    # Machine non-overlap disjunctive constraints
    for j1 in 1:J, j2 in 1:J, m in 1:M_cnt
        if j1 < j2
            @constraint(model, start[j1, m] + P[j1, m] <= start[j2, m] + BIG_M * (1 - z[j1, j2, m]))
            @constraint(model, start[j2, m] + P[j2, m] <= start[j1, m] + BIG_M * z[j1, j2, m])
        end
    end

    # Makespan completion bound
    for j in 1:J
        @constraint(model, Cmax >= start[j, M_cnt] + P[j, M_cnt])
    end

    @objective(model, Min, Cmax)

    p.variables = Dict{String,Any}("start" => start, "Cmax" => Cmax)
    p.model = model
    return model
end

function result(p::JobShopProblem)
    row = invoke(result, Tuple{LPProblem}, p)
    cmax_val = try value(p.variables["Cmax"]) catch; 0.0 end
    row["makespan"] = cmax_val
    return row
end

const Problem = JobShopProblem
