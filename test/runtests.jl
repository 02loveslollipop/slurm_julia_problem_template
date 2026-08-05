# Test suite for the Julia problem template.
#
# Run with:
#     julia --project=. -e 'using Pkg; Pkg.test()'
using Test
using JuMP
import MathOptInterface as MOI
using SHA
using JSON
using SQLite
using Tables
import DBInterface

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
        @test wall < 3.0
        @info "  $N tasks x $(S)s -> wall: $(round(wall, digits=2))s (sequential would be $(N * S)s)"
    end
end

@testset "distributed_helpers" begin
    @testset "total_combinations and combo_at linear indexing" begin
        ranges = Dict{String,Any}("a" => [10, 20], "b" => [100, 200, 300])
        @test total_combinations(ranges) == 6

        c1 = combo_at(ranges, 1)
        c6 = combo_at(ranges, 6)
        @test c1["a"] in [10, 20] && c1["b"] in [100, 200, 300]
        @test c6["a"] in [10, 20] && c6["b"] in [100, 200, 300]
        @test c1 != c6
        @test_throws ErrorException combo_at(ranges, 0)
        @test_throws ErrorException combo_at(ranges, 7)
    end

    @testset "partition_range" begin
        # 20 items across 4 ranks: 5 items each
        @test partition_range(20, 0, 4) == 1:5
        @test partition_range(20, 1, 4) == 6:10
        @test partition_range(20, 2, 4) == 11:15
        @test partition_range(20, 3, 4) == 16:20

        # 10 items across 1 rank
        @test partition_range(10, 0, 1) == 1:10

        # Extra ranks beyond total items return empty range
        @test partition_range(2, 5, 10) == 1:0
    end

    @testset "detect_environment_rank" begin
        ENV["SLURM_PROCID"] = "2"
        ENV["SLURM_NTASKS"] = "8"
        r, n, src = detect_environment_rank()
        @test r == 2 && n == 8 && src == "Slurm"
        delete!(ENV, "SLURM_PROCID")
        delete!(ENV, "SLURM_NTASKS")

        r_std, n_std, src_std = detect_environment_rank()
        @test r_std == 0 && n_std == 1 && src_std == "standalone"
    end
end

@testset "storage_helpers" begin
    @testset "SQLite database init and row insertion" begin
        db_file = joinpath(mktempdir(), "test.db")
        db = init_sqlite_db(db_file)
        @test isfile(db_file)

        row = Dict{String,Any}("segments" => 1, "seed" => 0, "status" => "OPTIMAL", "total_doctors" => 5, "solve_time_s" => 0.12)
        write_sqlite_row(db, 0, 0, row)

        res = DBInterface.execute(db, "SELECT rank, rep, status, solve_time_s FROM results;") |> Tables.columntable
        @test length(res.rank) == 1
        @test res.rank[1] == 0
        @test res.status[1] == "OPTIMAL"
        @test res.solve_time_s[1] == 0.12
    end

    @testset "compute_summary and JSON result writers" begin
        rows = [
            Dict{String,Any}("status" => "OPTIMAL", "solve_time_s" => 1.0),
            Dict{String,Any}("status" => "OPTIMAL", "solve_time_s" => 3.0),
        ]
        summary = compute_summary(rows)
        @test summary["total_runs"] == 2
        @test summary["successful_runs"] == 2
        @test summary["avg_solve_time_s"] == 2.0
        @test summary["min_solve_time_s"] == 1.0
        @test summary["max_solve_time_s"] == 3.0

        json_file = joinpath(mktempdir(), "test.json")
        write_json_results(json_file, rows, summary)
        @test isfile(json_file)

        parsed = JSON.parsefile(json_file)
        @test parsed["summary"]["total_runs"] == 2
        @test length(parsed["results"]) == 2
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

@testset "main e2e" begin
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

    @testset "range partitioning with SQLite output" begin
        db_file = joinpath(mktempdir(), "range_test.db")
        params = JSON.json(Dict("segments" => 1, "m" => 5, "n" => 5, "K" => 48))
        code, output = run_main("--params", params, "--range", "1:2", "--format", "sqlite",
                                "--time-limit", "10", "--gap-rel", "0.1", "--out", db_file)
        if code != 0
            @info output
        end
        @test code == 0
        @test isfile(db_file)

        db = SQLite.DB(db_file)
        res = DBInterface.execute(db, "SELECT COUNT(*) as cnt FROM results;") |> Tables.columntable
        @test res.cnt[1] == 1
    end

    @testset "rank partitioning with JSON output" begin
        json_file1 = joinpath(mktempdir(), "rank0.json")
        json_file2 = joinpath(mktempdir(), "rank1.json")
        params = JSON.json(Dict("segments" => 1, "m" => 5, "n" => 5, "K" => 48))

        code1, out1 = run_main("--params", params, "--rank", "0", "--num-ranks", "2", "--format", "json",
                               "--time-limit", "10", "--gap-rel", "0.1", "--out", json_file1)
        code2, out2 = run_main("--params", params, "--rank", "1", "--num-ranks", "2", "--format", "json",
                               "--time-limit", "10", "--gap-rel", "0.1", "--out", json_file2)

        @test code1 == 0
        @test isfile(json_file1)
        p1 = JSON.parsefile(json_file1)
        @test p1["results"][1]["rank"] == 0

        @test code2 == 0
        @test isfile(json_file2)
        p2 = JSON.parsefile(json_file2)
        @test p2["results"][1]["rank"] == 1
    end

    @testset "CSV format output" begin
        csv_file = joinpath(mktempdir(), "results.csv")
        params = JSON.json(Dict("segments" => 1, "m" => 5, "n" => 5, "K" => 48))
        code, output = run_main("--params", params, "--format", "csv",
                                "--time-limit", "10", "--gap-rel", "0.1", "--out", csv_file)
        @test code == 0
        @test isfile(csv_file)
        lines = readlines(csv_file)
        @test length(lines) >= 2
    end
end

@testset "main.jl is unmodified" begin
    EXPECTED_SHA256 = "7691c27dbc5cf06f91b2885c04af8f9379b34d3c82974e76aa54ffad2111475b"
    actual = bytes2hex(sha256(read(joinpath(PROJECT_ROOT, "main.jl"))))
    @test actual == EXPECTED_SHA256
end
