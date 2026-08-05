# Example 5: Multi-Echelon Supply Chain Logistics Network
#
# Optimizes flow from Plants -> Distribution Centers (DCs) -> Retailers to minimize total cost.
include("../core/base_problem.jl")

using JuMP
using Random

mutable struct SupplyChainProblem <: LPProblem
    n_plants::Int
    n_dcs::Int
    n_retailers::Int
    transport_multiplier::Float64
    seed::Int
    params::Dict{String,Any}
    model::Union{Nothing,JuMP.Model}
    variables::Dict{String,Any}
    solve_time_s::Union{Nothing,Float64}
end

SupplyChainProblem(; n_plants = 3, n_dcs = 5, n_retailers = 15, transport_multiplier = 1.0, seed = 0) = SupplyChainProblem(
    n_plants, n_dcs, n_retailers, transport_multiplier, seed,
    Dict{String,Any}("n_plants" => n_plants, "n_dcs" => n_dcs, "n_retailers" => n_retailers, "transport_multiplier" => transport_multiplier, "seed" => seed),
    nothing, Dict{String,Any}(), nothing
)

const PARAM_RANGES = Dict{String,Any}(
    "transport_multiplier" => [0.8, 1.0, 1.5],
    "n_dcs" => [3, 5, 8],
    "seed" => [0, 1],
)

function build(p::SupplyChainProblem)
    rng = MersenneTwister(p.seed)
    P, D, R = p.n_plants, p.n_dcs, p.n_retailers

    plant_cap = rand(rng, 200:500, P)
    dc_cap = rand(rng, 100:300, D)
    retailer_demand = rand(rng, 10:40, R)
    
    prod_cost = rand(rng, 10:20, P)
    dc_fixed_cost = rand(rng, 500:1500, D)

    p2d_cost = rand(rng, 2:8, P, D) .* p.transport_multiplier
    d2r_cost = rand(rng, 1:5, D, R) .* p.transport_multiplier

    model = Model()
    @variable(model, y[1:D], Bin)                  # Open DC j
    @variable(model, flow_p2d[1:P, 1:D] >= 0)      # Plant i -> DC j
    @variable(model, flow_d2r[1:D, 1:R] >= 0)      # DC j -> Retailer k

    # Plant capacity bounds
    for i in 1:P
        @constraint(model, sum(flow_p2d[i, j] for j in 1:D) <= plant_cap[i])
    end

    # DC throughput & capacity bounds
    for j in 1:D
        @constraint(model, sum(flow_p2d[i, j] for i in 1:P) == sum(flow_d2r[j, k] for k in 1:R))
        @constraint(model, sum(flow_d2r[j, k] for k in 1:R) <= dc_cap[j] * y[j])
    end

    # Retailer demand satisfaction
    for k in 1:R
        @constraint(model, sum(flow_d2r[j, k] for j in 1:D) >= retailer_demand[k])
    end

    @objective(model, Min,
        sum(prod_cost[i] * flow_p2d[i, j] for i in 1:P, j in 1:D) +
        sum(dc_fixed_cost[j] * y[j] for j in 1:D) +
        sum(p2d_cost[i, j] * flow_p2d[i, j] for i in 1:P, j in 1:D) +
        sum(d2r_cost[j, k] * flow_d2r[j, k] for j in 1:D, k in 1:R)
    )

    p.variables = Dict{String,Any}("y" => y, "p2d" => flow_p2d, "d2r" => flow_d2r)
    p.model = model
    return model
end

function result(p::SupplyChainProblem)
    row = invoke(result, Tuple{LPProblem}, p)
    y_val = try value.(p.variables["y"]) catch; zeros(p.n_dcs) end
    row["open_dcs"] = count(v -> v > 0.5, y_val)
    return row
end

const Problem = SupplyChainProblem
