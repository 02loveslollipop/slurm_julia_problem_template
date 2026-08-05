# Base interface for JuMP problems.
#
# main.jl depends only on this interface -- never on a specific problem's
# internals -- so main.jl should not need to change when you swap in a new
# problem via problem.jl.
using JuMP
using Logging
using SQLite
using Tables
import Gurobi
import HiGHS
import MPI
import MathOptInterface as MOI
import Dates
import DBInterface
import JSON

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
    wls_id = strip(get(ENV, "GRB_WLSACCESSID", ""))
    wls_sec = strip(get(ENV, "GRB_WLSSECRET", ""))
    lic_str = strip(get(ENV, "GRB_LICENSEID", ""))

    !isempty(wls_id) && (params["WLSACCESSID"] = wls_id)
    !isempty(wls_sec) && (params["WLSSECRET"] = wls_sec)
    if !isempty(lic_str)
        lic_id = tryparse(Int, lic_str)
        lic_id !== nothing && (params["LICENSEID"] = lic_id)
    end
    return params
end

function _gurobi_license_present()
    """True if Gurobi has real license credentials configured (WLS env vars
    or a license file), as opposed to a size-limited trial license."""
    params = _gurobi_env_params()
    if haskey(params, "WLSACCESSID") && haskey(params, "WLSSECRET") && haskey(params, "LICENSEID")
        return true
    end
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

function solve_model!(model; solver = nothing, msg = false,
                      time_limit = nothing, gap_rel = nothing, threads = nothing)
    """Set the optimizer, apply limits, optimize!, and return elapsed seconds.

    Shared by solve!() and by sub-problem workers (e.g. segments), so every
    model in the project gets the same solver strategy and limits.
    """
    factory = solver === nothing ?
              default_solver(msg = msg, time_limit = time_limit, gap_rel = gap_rel, threads = threads) :
              solver
    set_optimizer(model, factory)
    _try_set_attribute(model, MOI.TimeLimitSec(), time_limit)
    _try_set_attribute(model, MOI.RelativeGapTolerance(), gap_rel)
    _try_set_attribute(model, MOI.NumberOfThreads(), threads)
    _try_set_attribute(model, MOI.Silent(), !msg)
    return @elapsed optimize!(model)
end

function solve!(p::LPProblem; solver = nothing, msg = false,
                time_limit = nothing, gap_rel = nothing, threads = nothing)
    """Build and solve the problem, storing timing on p. Returns the model."""
    model = build(p)
    @info "Problem defined: $(nameof(typeof(p))) with params=$(p.params)"
    elapsed = solve_model!(model; solver = solver, msg = msg, time_limit = time_limit,
                           gap_rel = gap_rel, threads = threads)
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

"""
    total_combinations(ranges::Dict{String,Any})

Returns total count of parameter combinations in ranges grid.
"""
function total_combinations(ranges::Dict{String,Any})
    isempty(ranges) && return 1
    return prod(length(v) for v in values(ranges))
end

"""
    combo_at(ranges::Dict{String,Any}, idx::Int)

Calculate 1-indexed 1D permutation `idx` (1..N_total) directly to a parameter Dict.
"""
function combo_at(ranges::Dict{String,Any}, idx::Int)
    isempty(ranges) && return Dict{String,Any}()
    rkeys = sort(collect(keys(ranges)))
    dims = Tuple(length(ranges[k]) for k in rkeys)
    N_total = prod(dims)
    (1 <= idx <= N_total) || error("Index $idx out of bounds (1..$N_total)")
    
    ci = CartesianIndices(dims)[idx]
    combo = Dict{String,Any}()
    for (d, k) in enumerate(rkeys)
        combo[k] = ranges[k][ci[d]]
    end
    return combo
end

"""
    partition_range(total_len::Int, rank::Int, num_ranks::Int)

Computes 1-indexed contiguous block `start_idx:end_idx` for 0-indexed rank.
"""
function partition_range(total_len::Int, rank::Int, num_ranks::Int)
    (total_len <= 0 || num_ranks <= 0) && return 1:0
    num_ranks == 1 && return 1:total_len
    
    items_per_rank = ceil(Int, total_len / num_ranks)
    start_idx = rank * items_per_rank + 1
    end_idx = min((rank + 1) * items_per_rank, total_len)
    
    start_idx > end_idx ? (1:0) : (start_idx:end_idx)
end

"""
    detect_environment_rank()

Detect rank and size from HPC environment variables (Slurm, OpenMPI, PMI).
Returns `(rank, num_ranks, source_name)`.
"""
function detect_environment_rank()
    env_pairs = [
        ("SLURM_PROCID", "SLURM_NTASKS", "Slurm"),
        ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "OpenMPI"),
        ("PMI_RANK", "PMI_SIZE", "PMI"),
        ("MPI_RANK", "MPI_SIZE", "MPI"),
    ]
    for (r_var, n_var, name) in env_pairs
        if haskey(ENV, r_var) && haskey(ENV, n_var)
            r = tryparse(Int, ENV[r_var])
            n = tryparse(Int, ENV[n_var])
            if r !== nothing && n !== nothing && n > 0
                return (r, n, name)
            end
        end
    end
    return (0, 1, "standalone")
end

"""
    init_mpi_context()

Initializes MPI if not already initialized and returns `(comm, rank, size)`.
"""
function init_mpi_context()
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    return comm, MPI.Comm_rank(comm), MPI.Comm_size(comm)
end

"""
    init_sqlite_db(path::String)

Initializes SQLite database with WAL mode and creates results table.
"""
function init_sqlite_db(path::String)
    db = SQLite.DB(path)
    SQLite.execute(db, "PRAGMA journal_mode=WAL;")
    SQLite.execute(db, "PRAGMA busy_timeout=5000;")
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rank INTEGER,
            rep INTEGER,
            params TEXT,
            status TEXT,
            objective TEXT,
            solve_time_s REAL,
            timestamp TEXT,
            details TEXT
        );
    """)
    return db
end

"""
    write_sqlite_row(db, rank::Int, rep::Int, row::Dict{String,Any})

Writes a single result row into SQLite DB.
"""
function write_sqlite_row(db, rank::Int, rep::Int, row::Dict{String,Any})
    status = string(get(row, "status", get(row, "seg0_status", "UNKNOWN")))
    obj = string(get(row, "objective", get(row, "total_doctors", "NA")))
    solve_time = Float64(get(row, "solve_time_s", 0.0))
    params_json = JSON.json(row)
    ts = string(Dates.now())
    
    DBInterface.execute(db, """
        INSERT INTO results (rank, rep, params, status, objective, solve_time_s, timestamp, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """, (rank, rep, params_json, status, obj, solve_time, ts, params_json))
end

"""
    compute_summary(rows::Vector{Dict{String,Any}})

Compute summary statistics for a set of result rows.
"""
function compute_summary(rows::Vector{Dict{String,Any}})
    total = length(rows)
    total == 0 && return Dict{String,Any}("total_runs" => 0)
    
    solve_times = [Float64(get(r, "solve_time_s", 0.0)) for r in rows]
    successful = count(r -> get(r, "status", "") == "OPTIMAL" || get(r, "seg0_status", "") == "OPTIMAL", rows)
    
    return Dict{String,Any}(
        "total_runs" => total,
        "successful_runs" => successful,
        "success_rate" => round(successful / total, digits=4),
        "min_solve_time_s" => minimum(solve_times),
        "max_solve_time_s" => maximum(solve_times),
        "avg_solve_time_s" => round(sum(solve_times) / total, digits=4),
    )
end

"""
    write_json_results(path::String, rows::Vector{Dict{String,Any}}, summary::Dict{String,Any})

Writes rows and summary metadata to a JSON file.
"""
function write_json_results(path::String, rows::Vector{Dict{String,Any}}, summary::Dict{String,Any})
    data = Dict{String,Any}(
        "summary" => summary,
        "results" => rows
    )
    open(path, "w") do io
        write(io, JSON.json(data, 2))
    end
end

"""
    write_jsonl_results(path::String, rows::Vector{Dict{String,Any}})

Writes rows to a JSON Lines file.
"""
function write_jsonl_results(path::String, rows::Vector{Dict{String,Any}})
    open(path, "w") do io
        for r in rows
            println(io, JSON.json(r))
        end
    end
end
