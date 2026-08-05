# Example 2: Capacitated Facility Location Problem
#
# Decides which facilities to open and how to route customer demand to minimize total cost.
include("../core/base_problem.jl")

using JuMP
using Random

mutable struct FacilityLocationProblem <: LPProblem
    n_facilities::Int
    n_customers::Int
    fixed_cost_scale::Float64
    seed::Int
    params::Dict{String,Any}
    model::Union{Nothing,JuMP.Model}
    variables::Dict{String,Any}
    solve_time_s::Union{Nothing,Float64}
end

FacilityLocationProblem(; n_facilities = 5, n_customers = 15, fixed_cost_scale = 100.0, seed = 0) = FacilityLocationProblem(
    n_facilities, n_customers, fixed_cost_scale, seed,
    Dict{String,Any}("n_facilities" => n_facilities, "n_customers" => n_customers, "fixed_cost_scale" => fixed_cost_scale, "seed" => seed),
    nothing, Dict{String,Any}(), nothing
)

const PARAM_RANGES = Dict{String,Any}(
    "n_facilities" => [3, 5, 8],
    "fixed_cost_scale" => [50.0, 100.0, 200.0],
    "seed" => [0, 1],
)

function build(p::FacilityLocationProblem)
    rng = MersenneTwister(p.seed)
    F = p.n_facilities
    C = p.n_customers

    fixed_costs = rand(rng, 100:500, F) .* p.fixed_cost_scale
    capacities = rand(rng, 30:80, F)
    demands = rand(rng, 5:15, C)
    
    # 2D coordinates for transport distance calculation
    f_pos = rand(rng, F, 2) .* 100.0
    c_pos = rand(rng, C, 2) .* 100.0
    shipping_costs = [sqrt(sum(abs2, f_pos[i, :] .- c_pos[j, :])) for i in 1:F, j in 1:C]

    model = Model()
    @variable(model, y[1:F], Bin)              # Open facility i
    @variable(model, x[1:F, 1:C] >= 0)          # Fraction of customer j demand served by facility i

    # Satisfy customer demand
    for j in 1:C
        @constraint(model, sum(x[i, j] for i in 1:F) == 1.0)
    end

    # Respect facility capacity limits
    for i in 1:F
        @constraint(model, sum(demands[j] * x[i, j] for j in 1:C) <= capacities[i] * y[i])
    end

    @objective(model, Min, sum(fixed_costs[i] * y[i] for i in 1:F) + sum(shipping_costs[i, j] * demands[j] * x[i, j] for i in 1:F, j in 1:C))

    p.variables = Dict{String,Any}("y" => y, "x" => x, "fixed_costs" => fixed_costs)
    p.model = model
    return model
end

function result(p::FacilityLocationProblem)
    row = invoke(result, Tuple{LPProblem}, p)
    y_val = try value.(p.variables["y"]) catch; zeros(p.n_facilities) end
    row["open_facilities"] = count(v -> v > 0.5, y_val)
    return row
end

const Problem = FacilityLocationProblem
