# Example 1: 0-1 Knapsack Problem
#
# Selects a subset of items to maximize total value while respecting capacity.
include("../core/base_problem.jl")

using JuMP
using Random

mutable struct KnapsackProblem <: LPProblem
    n_items::Int
    capacity_ratio::Float64
    seed::Int
    params::Dict{String,Any}
    model::Union{Nothing,JuMP.Model}
    variables::Dict{String,Any}
    solve_time_s::Union{Nothing,Float64}
end

KnapsackProblem(; n_items = 10, capacity_ratio = 0.5, seed = 0) = KnapsackProblem(
    n_items, capacity_ratio, seed,
    Dict{String,Any}("n_items" => n_items, "capacity_ratio" => capacity_ratio, "seed" => seed),
    nothing, Dict{String,Any}(), nothing
)

const PARAM_RANGES = Dict{String,Any}(
    "n_items" => [10, 20, 50],
    "capacity_ratio" => [0.3, 0.5, 0.7],
    "seed" => [0, 1, 2],
)

function build(p::KnapsackProblem)
    rng = MersenneTwister(p.seed)
    weights = rand(rng, 10:100, p.n_items)
    values = rand(rng, 20:200, p.n_items)
    capacity = round(Int, sum(weights) * p.capacity_ratio)

    model = Model()
    @variable(model, x[1:p.n_items], Bin)

    @constraint(model, capacity_constraint, sum(weights[i] * x[i] for i in 1:p.n_items) <= capacity)
    @objective(model, Max, sum(values[i] * x[i] for i in 1:p.n_items))

    p.variables = Dict{String,Any}("x" => x, "weights" => weights, "values" => values, "capacity" => capacity)
    p.model = model
    return model
end

function result(p::KnapsackProblem)
    row = invoke(result, Tuple{LPProblem}, p)
    x_val = try value.(p.variables["x"]) catch; zeros(p.n_items) end
    row["selected_items"] = count(v -> v > 0.5, x_val)
    row["total_weight"] = sum(p.variables["weights"][i] * (x_val[i] > 0.5 ? 1 : 0) for i in 1:p.n_items)
    return row
end

const Problem = KnapsackProblem
