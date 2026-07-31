# Base interface for JuMP problems.
#
# main.jl depends only on this interface -- never on a specific problem's
# internals -- so main.jl should not need to change when you swap in a new
# problem via problem.jl.
using JuMP
using Logging
import Gurobi
import HiGHS
import MathOptInterface as MOI

"""
    abstract type LPProblem end

Subclass this in problem.jl to define a new optimization problem.

Override `PARAM_RANGES` (a Dict of constructor kwarg name => Vector of
values) to declare which parameters main.jl sweeps over, and implement
`build(p)`. Extend `result(p)` (calling `invoke(result, Tuple{LPProblem}, p)`)
to add problem-specific fields to the output row.
"""
abstract type LPProblem end

# Fallback for problems that forget to implement build() -- mirrors the
# abstract-method contract of the Python template.
build(::LPProblem) = error("build() not implemented for the problem type")

function configure_logging(level = "info")
    """Set up logging for the given --log level (info, debug, silent)."""
    if level == "silent"
        global_logger(NullLogger())
    elseif level == "debug"
        global_logger(ConsoleLogger(stdout, Logging.Debug))
    else
        global_logger(ConsoleLogger(stdout, Logging.Info))
    end
end

function _gurobi_env_params()
    """WLS license credentials from GRB_* env vars, as Gurobi Env params."""
    params = Dict{String,Any}()
    haskey(ENV, "GRB_WLSACCESSID") && (params["WLSACCESSID"] = ENV["GRB_WLSACCESSID"])
    haskey(ENV, "GRB_WLSSECRET") && (params["WLSSECRET"] = ENV["GRB_WLSSECRET"])
    haskey(ENV, "GRB_LICENSEID") && (params["LICENSEID"] = parse(Int, ENV["GRB_LICENSEID"]))
    return params
end

function _gurobi_license_present()
    """True if Gurobi has real license credentials configured (WLS env vars
    or a license file), as opposed to a size-limited trial license."""
    !isempty(_gurobi_env_params()) && return true
    lic_file = get(ENV, "GRB_LICENSE_FILE", "")
    candidates = String[
        lic_file,
        joinpath(homedir(), "gurobi.lic"),
        joinpath(pwd(), "gurobi.lic"),
    ]
    return any(c -> !isempty(c) && isfile(c), candidates)
end

function gurobi_available()
    """True if Gurobi.jl is installed and a real Gurobi license is configured."""
    Base.find_package("Gurobi") === nothing && return false
    return _gurobi_license_present()
end

function _gurobi_optimizer_factory()
    """Optimizer factory for Gurobi with WLS credentials from the env."""
    return () -> Gurobi.Optimizer(Gurobi.Env(_gurobi_env_params()))
end

function default_solver(; msg = false, time_limit = nothing, gap_rel = nothing, threads = nothing)
    """Gurobi if installed and licensed (preferred), else HiGHS."""
    if gurobi_available()
        @debug "Using Gurobi (WLS license)"
        return _gurobi_optimizer_factory()
    end
    @debug "Using HiGHS"
    return HiGHS.Optimizer
end

function _try_set_attribute(model, attr, value)
    """Best-effort solver attribute set: some solvers don't support every
    standard MOI attribute, and the sweep shouldn't die because of it."""
    value === nothing && return
    try
        set_attribute(model, attr, value)
    catch e
        @warn "Could not set attribute $attr = $value: $e"
    end
end

function solve!(p::LPProblem; solver = nothing, msg = false,
                time_limit = nothing, gap_rel = nothing, threads = nothing)
    """Build and solve the problem, storing timing on p. Returns the model."""
    model = build(p)
    @info "Problem defined: $(nameof(typeof(p))) with params=$(p.params)"
    factory = solver === nothing ?
              default_solver(msg = msg, time_limit = time_limit, gap_rel = gap_rel, threads = threads) :
              solver
    @debug "Solving with $factory (msg=$msg, time_limit=$time_limit, gap_rel=$gap_rel, threads=$threads)"
    set_optimizer(model, factory)
    _try_set_attribute(model, MOI.TimeLimitSec(), time_limit)
    _try_set_attribute(model, MOI.RelativeGapTolerance(), gap_rel)
    _try_set_attribute(model, MOI.NumberOfThreads(), threads)
    _try_set_attribute(model, MOI.Silent(), !msg)
    elapsed = @elapsed optimize!(model)
    p.model = model
    p.solve_time_s = elapsed
    @debug "Solve finished: status=$(termination_status(model)), objective=$(objective_value(model)), time=$(elapsed)s"
    return model
end

function result(p::LPProblem)
    """Summary row. Extend via `invoke(result, Tuple{LPProblem}, p)`."""
    p.model === nothing && error("solve!() must be called before result()")
    objective = try
        objective_value(p.model)
    catch
        "NA"
    end
    row = Dict{String,Any}(
        "status" => string(termination_status(p.model)),
        "objective" => objective,
        "solve_time_s" => p.solve_time_s,
    )
    return merge!(row, p.params)
end

function _run_task(task)
    """Unpack and call a task; `(fn, args)` tuples call fn(args...)."""
    fn, args = task isa Tuple ? task : (task, ())
    @debug "Running subproblem task: $(nameof(fn))"
    return isempty(args) ? fn() : fn(args...)
end

function solve_subproblems_parallel(subproblem_tasks; n_workers = nothing)
    """Solve independent sub-problems concurrently via @async tasks.

    Parameters
    ----------
    subproblem_tasks : Vector
        Each element is a function or a `(fn, args)` tuple that builds,
        solves, and returns a result dict. Unlike the Python template
        (multiprocessing), closures work here -- Julia tasks need no
        pickling.
    n_workers : Union{Nothing,Int}
        Maximum number of concurrent tasks. Defaults to
        `min(Threads.nthreads(), length(tasks))`.

    Returns
    -------
    Vector{Any}
        Results in the same order as `subproblem_tasks`. The first
        exception raised by any task is rethrown after all tasks finish.

    Notes
    -----
    A semaphore caps the number of concurrently running tasks; results are
    placed back into a pre-allocated vector by index so ordering is
    preserved regardless of completion order.
    """
    isempty(subproblem_tasks) && return Any[]
    n = something(n_workers, min(Threads.nthreads(), length(subproblem_tasks)))
    results = Vector{Any}(undef, length(subproblem_tasks))
    sem = Channel{Nothing}(n)
    for _ in 1:n
        put!(sem, nothing)
    end
    results_ch = Channel{Any}(length(subproblem_tasks))
    errs_ch = Channel{Any}(length(subproblem_tasks))
    @sync for (i, task) in enumerate(subproblem_tasks)
        @async begin
            take!(sem)
            try
                put!(results_ch, (i, _run_task(task)))
            catch e
                put!(errs_ch, (i, e))
            finally
                put!(sem, nothing)
            end
        end
    end
    if !isempty(errs_ch)
        _, e = take!(errs_ch)
        throw(e)
    end
    for _ in subproblem_tasks
        i, r = take!(results_ch)
        results[i] = r
    end
    return results
end
