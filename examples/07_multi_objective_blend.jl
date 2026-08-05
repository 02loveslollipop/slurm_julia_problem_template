# Example 7: Multi-Objective Material Blending Problem
#
# Blends raw ingredients to meet minimum quality specifications while minimizing cost and emissions.
include("../core/base_problem.jl")

using JuMP
using Random

mutable struct BlendingProblem <: LPProblem
    n_ingredients::Int
    quality_target::Float64
    emission_weight::Float64
    seed::Int
    params::Dict{String,Any}
    model::Union{Nothing,JuMP.Model}
    variables::Dict{String,Any}
    solve_time_s::Union{Nothing,Float64}
end

BlendingProblem(; n_ingredients = 6, quality_target = 75.0, emission_weight = 10.0, seed = 0) = BlendingProblem(
    n_ingredients, quality_target, emission_weight, seed,
    Dict{String,Any}("n_ingredients" => n_ingredients, "quality_target" => quality_target, "emission_weight" => emission_weight, "seed" => seed),
    nothing, Dict{String,Any}(), nothing
)

const PARAM_RANGES = Dict{String,Any}(
    "quality_target" => [60.0, 75.0, 85.0],
    "emission_weight" => [0.0, 10.0, 50.0],
    "seed" => [0, 1],
)

function build(p::BlendingProblem)
    rng = MersenneTwister(p.seed)
    N = p.n_ingredients

    costs = rand(rng, 10.0:50.0, N)
    qualities = rand(rng, 40.0:95.0, N)
    emissions = rand(rng, 1.0:15.0, N)
    max_fractions = rand(rng, 0.3:0.6, N)

    model = Model()
    @variable(model, 0 <= x[i=1:N] <= max_fractions[i])

    # Blend ratio sums to 1.0
    @constraint(model, sum(x[i] for i in 1:N) == 1.0)

    # Minimum blend quality specification
    @constraint(model, sum(qualities[i] * x[i] for i in 1:N) >= p.quality_target)

    # Multi-objective weighted sum: Cost + emission_weight * Emissions
    @objective(model, Min, sum(costs[i] * x[i] for i in 1:N) + p.emission_weight * sum(emissions[i] * x[i] for i in 1:N))

    p.variables = Dict{String,Any}("x" => x, "costs" => costs, "emissions" => emissions)
    p.model = model
    return model
end

function result(p::BlendingProblem)
    row = invoke(result, Tuple{LPProblem}, p)
    x_val = try value.(p.variables["x"]) catch; zeros(p.n_ingredients) end
    row["ingredients_used"] = count(v -> v > 1e-4, x_val)
    row["total_emissions"] = sum(p.variables["emissions"][i] * x_val[i] for i in 1:p.n_ingredients)
    return row
end

const Problem = BlendingProblem
