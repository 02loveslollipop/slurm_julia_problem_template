# Example 4: Doctor Scheduling with Radial Map Segmentation
#
# Splits a 100x100 map into N radial segments (pizza slices from center)
# and solves each segment concurrently via solve_subproblems_parallel.
include("../core/base_problem.jl")

using JuMP
using Random

const MAP_SIZE = 100.0
const M_USERS = 30
const N_DOCTORS_MAX = 20
const K_INTERVALS = 144
const DT_MIN = 10
const VELOCITY_KMH = 30

mutable struct RadialDoctorProblem <: LPProblem
    segments::Int
    seed::Int
    m::Int
    n::Int
    K::Int
    params::Dict{String,Any}
    model::Union{Nothing,JuMP.Model}
    variables::Dict{String,Any}
    solve_time_s::Union{Nothing,Float64}
    segment_tasks::Vector{Any}
    segment_results::Vector{Dict{String,Any}}
end

RadialDoctorProblem(; segments = 1, seed = 0, m = nothing, n = nothing, K = nothing) = RadialDoctorProblem(
    segments, seed,
    m === nothing ? M_USERS : m,
    n === nothing ? N_DOCTORS_MAX : n,
    K === nothing ? K_INTERVALS : K,
    Dict{String,Any}("segments" => segments, "seed" => seed),
    nothing, Dict{String,Any}(), nothing, Any[], Dict{String,Any}[],
)

const PARAM_RANGES = Dict{String,Any}(
    "segments" => [1, 2, 4, 8],
    "seed" => collect(0:4),
)

function _generate_patients(p, rng)
    m, K = p.m, p.K
    r = rand(rng, m, 2) .* MAP_SIZE
    a = Dict{Int,Int}(i => rand(rng, 1:(K - 10)) for i in 1:m)
    b = Dict{Int,Int}()
    c = Dict{Int,Int}()
    tau = Dict{Int,Vector{Int}}()
    for i in 1:m
        ventana = min(K - a[i] - 1, rand(rng, 4:20))
        b[i] = a[i] + ventana + 1
        c[i] = rand(rng, 1:max(2, b[i] - a[i]))
        tau[i] = [k for k in 1:K if a[i] <= k <= b[i] - c[i] + 1]
    end
    return r, a, b, c, tau
end

function _assign_segments(p, r)
    cx = cy = MAP_SIZE / 2.0
    angles = atan.(r[:, 2] .- cy, r[:, 1] .- cx)
    angles = angles .+ 2pi .* (angles .< 0)
    seg_size = 2pi / p.segments
    segs = floor.(Int, angles ./ seg_size)
    return clamp.(segs, 0, p.segments - 1)
end

function build(p::RadialDoctorProblem)
    rng = MersenneTwister(p.seed)
    r, a, b, c, tau = _generate_patients(p, rng)
    seg_assignments = _assign_segments(p, r)

    p.segment_tasks = Vector{Any}(undef, p.segments)
    for seg in 0:(p.segments - 1)
        orig_ids = [i for i in 1:p.m if seg_assignments[i] == seg]
        p.segment_tasks[seg + 1] = Dict{String,Any}(
            "seg" => seg,
            "params" => Dict{String,Any}(
                "orig_ids" => orig_ids,
                "r" => r, "a" => a, "b" => b, "c" => c, "tau" => tau,
            ),
        )
    end

    p.variables = Dict{String,Any}()
    p.model = Model()
    return p.model
end

_value(v::JuMP.VariableRef) = try value(v) catch; 0.0 end

function _segment_worker(task)
    seg = task["seg"]
    params = task["params"]
    dt = task["dt"]
    vel = task["vel"]
    K = task["K"]
    n = task["n"]
    time_limit = get(task, "time_limit", nothing)
    gap_rel = get(task, "gap_rel", nothing)
    threads = get(task, "threads", nothing)
    msg = get(task, "msg", false)

    orig_ids = params["orig_ids"]
    m_local = length(orig_ids)
    n_local = min(n, m_local)
    if n_local < 1
        return Dict{String,Any}("seg" => seg, "doctors" => 0, "status" => "NoPatients", "time" => 0.0)
    end

    remap = Dict{Int,Int}(orig => new for (new, orig) in enumerate(sort(orig_ids)))

    r_full = params["r"]
    r_local = [r_full[oi, :] for oi in orig_ids]
    dist = [sqrt(sum(abs2, r_local[i] .- r_local[j])) for i in 1:m_local for j in 1:m_local]
    dist = reshape(dist, m_local, m_local)
    delta_arr = ceil.(Int, dist ./ vel .* 60 ./ dt)
    delta_sub = Dict{Tuple{Int,Int},Int}((i, j) => delta_arr[i, j] for i in 1:m_local for j in 1:m_local)

    a_full, b_full, c_full, tau_full = params["a"], params["b"], params["c"], params["tau"]
    tau_sub = Dict(remap[oi] => tau_full[oi] for oi in orig_ids)
    c_sub = Dict(remap[oi] => c_full[oi] for oi in orig_ids)

    model = Model()
    y = @variable(model, [1:n_local], Bin)
    x = Dict{Tuple{Int,Int,Int},JuMP.VariableRef}()
    for li in 1:m_local, j in 1:n_local, k in tau_sub[li]
        x[li, j, k] = @variable(model, binary = true)
    end

    @objective(model, Min, sum(y[j] for j in 1:n_local))

    for li in 1:m_local
        @constraint(model, sum(x[li, j, k] for j in 1:n_local for k in tau_sub[li]) == 1)
    end

    M = 1000
    for li in 1:m_local, j in 1:n_local, k in tau_sub[li]
        conflict = sum(
            init = 0.0,
            (x[eta, j, s] for eta in 1:m_local if eta != li
             for s in (k - c_sub[eta] - delta_sub[eta, li] + 1):k
             if haskey(x, (eta, j, s))),
        )
        @constraint(model, x[li, j, k] <= 1 - conflict / M)
    end

    for j in 1:n_local
        @constraint(model, sum(x[li, j, k] for li in 1:m_local for k in tau_sub[li]) <= M * y[j])
    end

    elapsed = solve_model!(model; msg = msg, time_limit = time_limit, gap_rel = gap_rel, threads = threads)
    doctors = count(j -> _value(y[j]) > 0.5, 1:n_local)
    status = string(termination_status(model))
    return Dict{String,Any}("seg" => seg, "doctors" => doctors, "status" => status, "time" => elapsed)
end

function solve!(p::RadialDoctorProblem; solver = nothing, msg = false,
                time_limit = nothing, gap_rel = nothing, threads = nothing)
    build(p)
    full_tasks = [Dict{String,Any}(
            "seg" => t["seg"], "params" => t["params"],
            "dt" => DT_MIN, "vel" => VELOCITY_KMH, "K" => p.K, "n" => p.n,
            "time_limit" => time_limit, "gap_rel" => gap_rel, "threads" => threads, "msg" => msg,
        ) for t in p.segment_tasks]

    if gurobi_available()
        all_threads = something(threads, Sys.CPU_THREADS)
        for t in full_tasks; t["threads"] = all_threads; end
        p.segment_results = [_segment_worker(t) for t in full_tasks]
    else
        n_workers = min(Threads.nthreads(), p.segments)
        threads_per_worker = something(threads, 2)
        for t in full_tasks; t["threads"] = threads_per_worker; end
        p.segment_results = solve_subproblems_parallel(
            [(_segment_worker, (t,)) for t in full_tasks]; n_workers = n_workers
        )
    end
    return _finalize(p)
end

function _finalize(p::RadialDoctorProblem)
    sort!(p.segment_results; by = r -> r["seg"])
    p.solve_time_s = sum(r["time"] for r in p.segment_results)
    return nothing
end

function result(p::RadialDoctorProblem)
    total_doctors = sum(r["doctors"] for r in p.segment_results)
    row = Dict{String,Any}(
        "segments" => p.segments,
        "total_doctors" => total_doctors,
        "seed" => p.seed,
        "solve_time_s" => p.solve_time_s,
    )
    for r in p.segment_results
        seg = r["seg"]
        row["seg$(seg)_doctors"] = r["doctors"]
        row["seg$(seg)_status"] = r["status"]
    end
    return row
end

const Problem = RadialDoctorProblem
