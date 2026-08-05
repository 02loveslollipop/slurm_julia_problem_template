# Example 3: Markowitz Portfolio Optimization (LP approximation / risk-bounded return)
#
# Allocates capital across assets to maximize expected return subject to risk/diversification bounds.
include("../core/base_problem.jl")

using JuMP
using Random

mutable struct PortfolioProblem <: LPProblem
    n_assets::Int
    risk_budget::Float64
    max_asset_weight::Float64
    seed::Int
    params::Dict{String,Any}
    model::Union{Nothing,JuMP.Model}
    variables::Dict{String,Any}
    solve_time_s::Union{Nothing,Float64}
end

PortfolioProblem(; n_assets = 10, risk_budget = 0.15, max_asset_weight = 0.25, seed = 0) = PortfolioProblem(
    n_assets, risk_budget, max_asset_weight, seed,
    Dict{String,Any}("n_assets" => n_assets, "risk_budget" => risk_budget, "max_asset_weight" => max_asset_weight, "seed" => seed),
    nothing, Dict{String,Any}(), nothing
)

const PARAM_RANGES = Dict{String,Any}(
    "risk_budget" => [0.05, 0.10, 0.20],
    "max_asset_weight" => [0.15, 0.25, 0.50],
    "seed" => [0, 1, 2],
)

function build(p::PortfolioProblem)
    rng = MersenneTwister(p.seed)
    N = p.n_assets

    expected_returns = rand(rng, N) .* 0.20 .+ 0.02
    volatilities = rand(rng, N) .* 0.30 .+ 0.05

    model = Model()
    @variable(model, 0 <= w[1:N] <= p.max_asset_weight)

    # Full allocation of capital
    @constraint(model, sum(w[i] for i in 1:N) == 1.0)

    # Total portfolio risk limit (weighted average volatility bound)
    @constraint(model, sum(volatilities[i] * w[i] for i in 1:N) <= p.risk_budget)

    @objective(model, Max, sum(expected_returns[i] * w[i] for i in 1:N))

    p.variables = Dict{String,Any}("w" => w, "returns" => expected_returns, "vols" => volatilities)
    p.model = model
    return model
end

function result(p::PortfolioProblem)
    row = invoke(result, Tuple{LPProblem}, p)
    w_val = try value.(p.variables["w"]) catch; zeros(p.n_assets) end
    row["allocated_assets"] = count(v -> v > 1e-4, w_val)
    row["max_weight_held"] = isempty(w_val) ? 0.0 : maximum(w_val)
    return row
end

const Problem = PortfolioProblem
