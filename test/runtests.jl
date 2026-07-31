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

        @test wall < 2.0
        @info "  $N tasks x $(S)s -> wall: $(round(wall, digits=2))s (sequential would be $(N * S)s)"
    end
end

@testset "problem" begin
    @testset "build returns a valid JuMP model with expected variables" begin
        problem = Problem(n_items = 5, capacity = 10, seed = 1)
        model = build(problem)
        @test model isa JuMP.Model
        @test length(problem.variables["x"]) == 5
    end

    @testset "solve reaches OPTIMAL on a small instance" begin
        problem = Problem(n_items = 5, capacity = 10, seed = 1)
        solve!(problem; time_limit = 10)
        @test termination_status(problem.model) == MOI.OPTIMAL
    end

    @testset "result contains expected keys" begin
        problem = Problem(n_items = 5, capacity = 10, seed = 1)
        solve!(problem; time_limit = 10)
        row = result(problem)
        for key in ("status", "objective", "solve_time_s", "n_items", "capacity", "seed", "n_selected")
            @test haskey(row, key)
        end
    end

    @testset "PARAM_RANGES declared for sweeps" begin
        @test PARAM_RANGES isa Dict
        @test !isempty(PARAM_RANGES)
    end
end

@testset "problem contract (validity)" begin
    @testset "compiles with default params" begin
        problem = Problem()
        model = build(problem)
        @test model isa JuMP.Model
        @test objective_function(model) !== nothing
        @test length(all_variables(model)) > 0
    end

    @testset "PARAM_RANGES keys match constructor kwargs" begin
        # Every PARAM_RANGES key must be a constructor kwarg, or main.jl's
        # sweep (Problem(; combo...)) fails at runtime. The template contract
        # is: every non-bookkeeping field is a constructor kwarg.
        bookkeeping = (:params, :model, :variables, :solve_time_s)
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
            @test objective_function(model) !== nothing
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
        params = JSON.json(Dict("n_items" => 5, "capacity" => 10, "seed" => 1))
        code, output = run_main("--params", params, "--repeats", "2",
                                "--time-limit", "10", "--out", out_file)
        if code != 0
            @info output
        end
        @test code == 0

        header, rows = parse_csv(out_file)
        @test length(rows) == 2
        status_col = findfirst(==("status"), header)
        @test all(r[status_col] == "OPTIMAL" for r in rows)
    end

    @testset "sweep produces full grid" begin
        out_file = joinpath(mktempdir(), "sweep.csv")
        code, output = run_main("--time-limit", "10", "--out", out_file)
        if code != 0
            @info output
        end
        @test code == 0

        _, rows = parse_csv(out_file)
        # PARAM_RANGES in the example problem: 4 n_items values x 2 capacities = 8 combos
        @test length(rows) == 8
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
