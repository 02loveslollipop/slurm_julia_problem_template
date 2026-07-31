# Running on Camber Cloud

This project runs as a plain Camber job: upload the code to Stash, then
submit a job whose command installs the Julia project and runs `main.jl`.
`run_job.sh` does exactly that and forwards any arguments through to
`main.jl`, so it works unchanged regardless of what you pass on `--cmd`.

## Prerequisites

- Camber CLI installed and logged in (`camber login`).
- `camber me` prints your username and Stash root
  (`stash://<your-username>/`).
- Julia 1.10+ available on the job node (`julia --version`). If the node
  image lacks Julia, install it in the job command (e.g. via `juliaup`)
  before calling `run_job.sh`.

## 1. Upload the project to Stash

```bash
camber stash mkdir stash://<your-username>/<project-name>
camber stash cp ./main.jl ./problem.jl ./Project.toml ./run_job.sh \
  stash://<your-username>/<project-name>/
camber stash cp -r ./core stash://<your-username>/<project-name>/
```

Re-run the relevant `cp` command whenever you change `problem.jl` (or any
other file) -- Stash does not auto-sync from your local disk.

## 2. Submit a small test job first

Validate the pipeline (dependency install, solver availability, a quick
solve) on the smallest node before spending credit on a bigger one:

```bash
camber job create --engine base --size xxsmall --num-nodes 1 \
  --cmd "bash run_job.sh --params '{\"n_items\": 5, \"capacity\": 10}' --repeats 1" \
  --path stash://<your-username>/<project-name>/
```

Check status and logs:

```bash
camber job get <job_id>
camber job logs <job_id>
```

`--params` overrides `PARAM_RANGES` with a single combination -- use it
for a fast correctness check, same as `julia main.jl --params ...` locally.

## 3. Run the real sweep on a bigger node

Once the test job completes cleanly, drop `--params` to run the full
`PARAM_RANGES` sweep defined in `problem.jl`, and size the node to your
workload (more cores -> pass `--threads` through to the solver too if your
problem benefits from it):

```bash
camber job create --engine base --size large --num-nodes 1 \
  --cmd "bash run_job.sh --time-limit 60 --gap-rel 0.01 --threads 16" \
  --path stash://<your-username>/<project-name>/
```

Node sizes accepted by `--size` (smallest to largest as of this writing):
`xxsmall`, `xsmall`, `small`, `medium`, `large`. Run
`camber job create --help` to confirm current values and
`camber` docs / pricing for current specs, since these can change.

## 4. Retrieve results

`main.jl` writes `results.csv` (or whatever `--out` you passed) into the
job's working directory, which is the Stash path passed via `--path` -- it
syncs back to Stash automatically once the job finishes:

```bash
camber stash ls stash://<your-username>/<project-name>/ -r
camber stash cp stash://<your-username>/<project-name>/results.csv ./
```

## Why HiGHS, not Gurobi

Camber job nodes don't have a Gurobi license available, so
`core/base_problem.jl` picks HiGHS (open source, ships as a self-contained
binary artifact) by default -- no extra setup needed on the node. Gurobi is
used only when the node has a real license configured (`GRB_*` WLS env
vars or a license file) *and* the full Gurobi optimizer with
`libgurobi*.so` is installed (`GUROBI_HOME` set).
