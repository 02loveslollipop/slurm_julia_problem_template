# Starter shell for JuMP optimization problems.
#
# This is the primary file you edit to adapt this template to a new JuMP problem.
# main.jl discovers your problem via `Problem` and `PARAM_RANGES` defined here.

include("core/base_problem.jl")

using JuMP

"""
    MyProblem <: LPProblem

Template optimization problem struct. Every non-bookkeeping field must be a
constructor keyword argument.
"""
mutable struct MyProblem <: LPProblem
    scale::Float64
    seed::Int
    params::Dict{String,Any}
    model::Union{Nothing,JuMP.Model}
    variables::Dict{String,Any}
    solve_time_s::Union{Nothing,Float64}
end

MyProblem(; scale = 1.0, seed = 0) = MyProblem(
    scale, seed,
    Dict{String,Any}("scale" => scale, "seed" => seed),
    nothing, Dict{String,Any}(), nothing
)

# Parameter grid swept by `julia main.jl`
const PARAM_RANGES = Dict{String,Any}(
    "scale" => [1.0, 2.0, 5.0],
    "seed" => [0, 1, 2],
)

function build(p::MyProblem)
    """Construct and return the JuMP model."""
    model = Model()
    @variable(model, x >= 0)
    @variable(model, y >= 0)
    
    @constraint(model, x + y <= 10.0 * p.scale)
    @constraint(model, 2x + y <= 15.0 * p.scale)
    
    @objective(model, Max, 3x + 5y)

    p.variables = Dict{String,Any}("x" => x, "y" => y)
    p.model = model
    return model
end

function result(p::MyProblem)
    """Summary result row for output logging."""
    row = invoke(result, Tuple{LPProblem}, p)
    row["x_val"] = try value(p.variables["x"]) catch; 0.0 end
    row["y_val"] = try value(p.variables["y"]) catch; 0.0 end
    return row
end

# Alias exported for main.jl
const Problem = MyProblem
