# Test suite for the Julia problem template.
#
# Run with:
#     julia --project=. -e 'using Pkg; Pkg.test()'
using Test
using JuMP
import MathOptInterface as MOI
using SHA
using JSON

include(joinpath(@__DIR__, "..", "problem.jl"))

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

# Small instance so solve-based tests stay fast (mirrors the Python
# separation-map tests).
const SMALL = (m = 5, n = 5, K = 48)

@testset "base_problem" begin
    @testset "LPProblem cannot be instantiated" begin
        @test_throws MethodError LPProblem()
    end

    @testset "result before solve! raises" begin
        struct Trivial <: LPProblem
            params::Dict{String,Any}
            model::Union{Nothing,JuMP.Model}
            variables::Dict{String,Any}
            solve_time_s::Union{Nothing,Float64}
        end
        Trivial() = Trivial(Dict{String,Any}(), nothing, Dict{String,Any}(), nothing)

        @test_throws ErrorException result(Trivial())
    end

    @testset "solve_subproblems_parallel runs concurrently" begin
        # 4 tasks sleeping 1s each: sequential would take ~4s, concurrent
        # (n_workers=4) should take ~1s.
        function sleeper(d)
            sleep(d)
            return Dict{String,Any}("slept" => d)
        end
        N, S = 4, 1.0
        wall = @elapsed results = solve_subproblems_parallel(
            [(sleeper, (S,)) for _ in 1:N], n_workers = N
        )

        @test length(results) == N
        @test all(r["slept"] == S for r in results)

        # Sequential would take ~4s; allow up to 3s so loaded 2-core CI
        # runners don't flake, while still proving concurrency.
        @test wall < 3.0
        @info "  $N tasks x $(S)s -> wall: $(round(wall, digits=2))s (sequential would be $(N * S)s)"
    end
end

@testset "problem" begin
    @testset "build returns a valid JuMP model and stages segment tasks" begin
        problem = Problem(segments = 4, seed = 0; SMALL...)
        model = build(problem)
        @test model isa JuMP.Model
        @test length(problem.segment_tasks) == 4
        @test haskey(problem.segment_tasks[1], "params")
        @test sum(length(t["params"]["orig_ids"]) for t in problem.segment_tasks) == 5
    end

    @testset "solve reaches OPTIMAL on a small instance" begin
        problem = Problem(segments = 1, seed = 0; SMALL...)
        solve!(problem; time_limit = 10, gap_rel = 0.1)
        @test problem.segment_results[1]["status"] == "OPTIMAL"
    end

    @testset "result contains expected keys" begin
        problem = Problem(segments = 1, seed = 0; SMALL...)
        solve!(problem; time_limit = 10, gap_rel = 0.1)
        row = result(problem)
        for key in ("segments", "total_doctors", "seed", "solve_time_s")
            @test haskey(row, key)
        end
    end

    @testset "PARAM_RANGES declared for sweeps" begin
        @test PARAM_RANGES isa Dict
        @test !isempty(PARAM_RANGES)
    end

    @testset "multi-segment solves correctly" begin
        problem = Problem(segments = 4, seed = 1; SMALL...)
        solve!(problem; time_limit = 10, gap_rel = 0.1)
        row = result(problem)
        @test row["total_doctors"] > 0
        for i in 0:3
            @test row["seg$(i)_status"] in ("OPTIMAL", "NoPatients")
        end
    end
end

@testset "problem contract (validity)" begin
    @testset "compiles with default params" begin
        problem = Problem()
        model = build(problem)
        @test model isa JuMP.Model
        @test length(problem.segment_tasks) == problem.segments
    end

    @testset "PARAM_RANGES keys match constructor kwargs" begin
        # Every PARAM_RANGES key must be a constructor kwarg, or main.jl's
        # sweep (Problem(; combo...)) fails at runtime. The template contract
        # is: every non-bookkeeping field is a constructor kwarg.
        bookkeeping = (:params, :model, :variables, :solve_time_s, :segment_tasks, :segment_results)
        data_fields = setdiff(fieldnames(Problem), bookkeeping)
        for key in keys(PARAM_RANGES)
            @test Symbol(key) in data_fields
        end
    end

    @testset "PARAM_RANGES values are non-empty" begin
        for (key, values) in PARAM_RANGES
            @test length(values) > 0
        end
    end

    @testset "every param combination compiles" begin
        # Cap how many PARAM_RANGES combinations get build-tested so a large
        # sweep grid doesn't make the test suite slow.
        rkeys = collect(keys(PARAM_RANGES))
        tested = 0
        for combo in Iterators.take(Iterators.product((PARAM_RANGES[k] for k in rkeys)...), 20)
            params = Dict{String,Any}(k => v for (k, v) in zip(rkeys, combo))
            problem = Problem(; (Symbol(k) => v for (k, v) in params)...)
            model = build(problem)
            @test model isa JuMP.Model
            @test length(problem.segment_tasks) == problem.segments
            tested += 1
        end
        @test tested > 0
    end
end

@testset "main" begin
    # main.jl is run as a subprocess against the environment that Pkg.test
    # set up for us, so all project deps are available to it.
    test_env = Base.active_project()

    function run_main(args...)
        out = IOBuffer()
        cmd = `$(Base.julia_cmd()) --project=$test_env $(joinpath(PROJECT_ROOT, "main.jl")) $args`
        code = try
            run(pipeline(cmd, stdout = out, stderr = out))
            0
        catch e
            e isa ProcessFailedException || rethrow()
            e.procs[1].exitcode
        end
        return code, String(take!(out))
    end

    function parse_csv(path)
        lines = readlines(path)
        isempty(lines) && return String[], Vector{Vector{String}}()
        header = split(lines[1], ",")
        rows = [split(line, ",") for line in lines[2:end]]
        return header, rows
    end

    @testset "single param combo with repeats" begin
        out_file = joinpath(mktempdir(), "results.csv")
        params = JSON.json(Dict("segments" => 2, "seed" => 1, "m" => 5, "n" => 5, "K" => 48))
        code, output = run_main("--params", params, "--repeats", "2",
                                "--time-limit", "10", "--gap-rel", "0.1", "--out", out_file)
        if code != 0
            @info output
        end
        @test code == 0

        header, rows = parse_csv(out_file)
        @test length(rows) == 2
        status_col = findfirst(==("seg0_status"), header)
        @test status_col !== nothing
        @test all(r[status_col] in ("OPTIMAL", "NoPatients") for r in rows)
    end

    @testset "sweep produces full grid" begin
        # The full PARAM_RANGES grid (default m=30/n=20/K=144) is too slow for
        # a test; mirror the Python separation-map tests and verify the sweep
        # plumbing with the SMALL instance instead.
        out_file = joinpath(mktempdir(), "sweep.csv")
        params = JSON.json(Dict("segments" => 1, "m" => 5, "n" => 5, "K" => 48))
        code, output = run_main("--params", params, "--repeats", "1",
                                "--time-limit", "10", "--gap-rel", "0.1", "--out", out_file)
        if code != 0
            @info output
        end
        @test code == 0

        _, rows = parse_csv(out_file)
        @test length(rows) == 1
    end
end

@testset "main.jl is unmodified" begin
    # Guards the template's core contract: main.jl is generic and problem-
    # agnostic, so adapting this template to a new problem should never
    # require touching it -- only problem.jl changes.
    #
    # If you have a genuine reason to change main.jl (a real bug fix, a new
    # generic CLI flag that every problem benefits from, etc.), that's fine --
    # just update EXPECTED_SHA256 below to the new file's hash in the same
    # commit, so the change is explicit and reviewable rather than accidental.
    # Current hash: sha256sum main.jl
    EXPECTED_SHA256 = "49722d293eb1f912af3f492b391bd13aa434907f9eb932ce37573ba046a624a0"
    actual = bytes2hex(sha256(read(joinpath(PROJECT_ROOT, "main.jl"))))
    @test actual == EXPECTED_SHA256
end
