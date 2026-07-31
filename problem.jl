# Define your optimization problem here.
#
# This is the ONLY file you should need to edit to adapt this template to a
# new JuMP problem. main.jl discovers your problem through the `Problem`
# alias and the `PARAM_RANGES` constant at the bottom of this file and never
# needs to change.
#
# Replace the example below (a tiny 0/1 knapsack) with your own model:
# declare which constructor kwargs should be swept over in PARAM_RANGES,
# build the JuMP model in build(), and optionally extend result() with
# extra fields.
#
# For problems with independent sub-components, use
# `core.base_problem.solve_subproblems_parallel` to solve them concurrently
# via @async tasks (see the docstring for details).
include("core/base_problem.jl")

using JuMP
using Random

mutable struct MyProblem <: LPProblem
    n_items::Int
    capacity::Int
    seed::Int
    params::Dict{String,Any}
    model::Union{Nothing,JuMP.Model}
    variables::Dict{String,Any}
    solve_time_s::Union{Nothing,Float64}
end

MyProblem(; n_items = 10, capacity = 20, seed = 0) = MyProblem(
    n_items, capacity, seed,
    Dict{String,Any}("n_items" => n_items, "capacity" => capacity, "seed" => seed),
    nothing, Dict{String,Any}(), nothing,
)

# Cartesian product swept by `julia main.jl` with no --params override.
const PARAM_RANGES = Dict{String,Any}(
    "n_items" => [5, 10, 15, 20],   # 5, 10, 15, 20
    "capacity" => [20, 40],
)

function build(p::MyProblem)
    rng = MersenneTwister(p.seed)
    weights = rand(rng, 1:10, p.n_items)
    values = rand(rng, 1:10, p.n_items)

    model = Model()
    x = @variable(model, [1:p.n_items], Bin)
    @objective(model, Max, sum(values[i] * x[i] for i in 1:p.n_items))
    @constraint(model, sum(weights[i] * x[i] for i in 1:p.n_items) <= p.capacity)

    p.variables["x"] = x
    return model
end

function result(p::MyProblem)
    row = invoke(result, Tuple{LPProblem}, p)
    row["n_selected"] = count(v -> value(v) > 0.5, p.variables["x"])
    return row
end

# The following commented example shows how to use solve_subproblems_parallel
# when a problem decomposes into independent sub-problems (e.g. geographic
# segments, time windows, asset classes). Each sub-problem is a plain Julia
# function (closures work too, unlike Python's pickling) that receives its
# data, builds a model, solves it, and returns results.
#
#   function _sub_worker(data, time_limit, gap_rel)
#       model = Model()
#       # ... build model from data ...
#       set_optimizer(model, default_solver(time_limit=time_limit, gap_rel=gap_rel))
#       optimize!(model)
#       return Dict("objective" => objective_value(model))
#   end
#
#   tasks = [(_sub_worker, (data_i, 30, 0.05)) for data_i in all_segments]
#   results = solve_subproblems_parallel(tasks)
#   # "scheduler" = the @sync task runner; "runners" = the spawned tasks

# main.jl imports this name -- keep it pointing at your problem class.
const Problem = MyProblem
