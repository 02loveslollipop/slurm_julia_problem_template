# julia-problem-template

Template project for parametrized JuMP linear/MILP optimization problems,
the Julia mirror of the Python `pulp-problem-template` (same architecture,
same CLI, same solver strategy). Copy this directory as the starting point
for a new problem and rename it.

## Structure

- **`main.jl`** -- generic runner. Not meant to be edited: it only depends
  on the problem interface in `core/base_problem.jl` and the `Problem`
  alias + `PARAM_RANGES` exported by `problem.jl`, so it stays correct no
  matter what problem you plug in.
- **`problem.jl`** -- your problem. Define constructor kwargs, their sweep
  ranges (`PARAM_RANGES`), the JuMP model (`build()`), and any extra
  result fields (`result()`). This is the only file you should need to edit.
- **`core/base_problem.jl`** -- the `LPProblem` abstract type and the
  shared machinery: `configure_logging`, the default-solver selection
  (Gurobi with WLS credentials from `GRB_*` env vars if licensed, else
  HiGHS), `solve!`, `result`, and `solve_subproblems_parallel`.
- **`tests/runtests.jl`** -- unit tests for the base layer, the example
  problem, an end-to-end `main.jl` invocation, and a guard test that fails
  if `main.jl` is edited (see Testing below).
- **`.github/workflows/tests.yml`** -- runs the test suite on push/PR
  across Julia 1.10-1.12.
- **`run_job.sh`** / **`CAMBER.md`** -- submission script and instructions
  for running this project as a job on Camber Cloud.
- **`Dockerfile`** -- containerized runner (see Docker below).

## Usage

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # first time only

# Sweep PARAM_RANGES (full cartesian product), 1 repeat each
julia --project=. main.jl

# Single parameter combination, repeated 5x (e.g. to check solve-time variance)
julia --project=. main.jl --params '{"n_items": 8, "capacity": 30}' --repeats 5

# Solver limits
julia --project=. main.jl --time-limit 30 --gap-rel 0.01 --threads 4
```

Results are written to `results.csv` (override with `--out`).

## Docker

```bash
docker build -t julia-problem .

# Sweep PARAM_RANGES, results written inside the container's /app
docker run --rm julia-problem

# Mount a host directory to get results.csv back out
docker run --rm -v "$PWD/out:/app/out" julia-problem --out out/results.csv

# Run the test suite instead of main.jl
docker run --rm --entrypoint julia julia-problem --project=. -e 'using Pkg; Pkg.test()'
```

## Adapting to a new problem

1. Open `problem.jl`. Rename `MyProblem` to something specific if you like
   (just keep the `Problem = MyProblem` alias at the bottom pointing at it).
2. Change the constructor kwargs to accept your problem's parameters
   (every non-bookkeeping field of your struct is a constructor kwarg).
3. Set `PARAM_RANGES` to the subset of those parameters you want `main.jl`
   to sweep over by default.
4. Implement `build(p)`: construct the JuMP model, store any variable
   vector(s) on `p.variables`, return the model.
5. (Optional) Extend `result(p)` to report problem-specific derived values,
   always starting from `row = invoke(result, Tuple{LPProblem}, p)`.
6. Update `tests/runtests.jl` for your new model's expected behavior.

`main.jl` does not change.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

14 tests:

| Testset | Tests | Checks |
|---|---|---|
| `base_problem` | 3 | `LPProblem` can't be instantiated (abstract type); `result()` raises if called before `solve!()`; `solve_subproblems_parallel` truly runs 4 sleeping tasks concurrently. |
| `problem` | 4 | `build()` returns a valid `JuMP.Model` with the expected variables; a small instance solves to `OPTIMAL`; `result()` has all expected keys; `PARAM_RANGES` is a non-empty dict. |
| `problem contract` | 4 | Generic, model-agnostic checks that survive after you swap in your own problem: `Problem()` compiles with default params; every `PARAM_RANGES` key is a real constructor kwarg (so the sweep can't throw); every range value is a non-empty iterable; every point in the sweep grid (capped at 20) builds a valid model. |
| `main` | 2 | End-to-end: running `main.jl` as a subprocess with `--params`/`--repeats` produces the right number of CSV rows; running it with no args sweeps the full `PARAM_RANGES` grid. |
| `main.jl is unmodified` | 1 | Hashes `main.jl` and compares against a pinned `sha256` -- fails if `main.jl` is edited, enforcing that only `problem.jl` (and its tests) change when adapting the template. If you have a real reason to change `main.jl`, update `EXPECTED_SHA256` in that test as part of the same change. |

## Running from a release (no repo needed)

Every push to `master` creates a GitHub Release with a `julia_problem.zip`
containing `problem.jl`, `main.jl`, `Project.toml`, `run_job.sh`, and the
`core/` module. The cluster can fetch and run with one command:

```bash
wget https://github.com/02loveslollipop/slurm_julia_problem_template/releases/latest/download/julia_problem.zip
unzip julia_problem.zip && bash run_job.sh --time-limit 120 --gap-rel 0.05
```

Override the release URL to pin a specific version instead of `latest`.

## Running on Camber Cloud

See [`CAMBER.md`](./CAMBER.md) for how to upload this project to Camber
Stash and submit it as a job (test on a small node first, then scale up).
