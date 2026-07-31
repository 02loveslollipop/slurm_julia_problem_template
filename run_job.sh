#!/usr/bin/env bash
# Submission script for Camber Cloud, invoked as e.g.:
#   camber job create --cmd "bash run_job.sh --time-limit 30" ...
# Installs Julia project dependencies (Gurobi + WLS license, HiGHS as
# fallback) and runs main.jl, forwarding any extra args.
set -euo pipefail

echo "== Installing dependencies =="
julia --project=. -e 'using Pkg; Pkg.instantiate()'

echo "== Configuring Gurobi WLS license =="
# Credentials are injected at runtime, never committed: locally from the
# gitignored .env file (if present), on Camber/Docker from job env vars.
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi
# Gurobi is used only when all three GRB_* vars are set.
export GRB_WLSACCESSID="${GRB_WLSACCESSID:-}"
export GRB_WLSSECRET="${GRB_WLSSECRET:-}"
export GRB_LICENSEID="${GRB_LICENSEID:-}"
if [ -z "$GRB_WLSSECRET" ]; then
  echo "WARNING: GRB_* WLS vars not set -- falling back to HiGHS"
fi

echo "== Configuring Gurobi library path =="
# Gurobi.jl needs GUROBI_HOME pointing at the directory containing
# libgurobi*.so. On Camber the full Gurobi tarball is not present by
# default, so skip if we can't find it -- HiGHS is the fallback solver.
if [ -z "${GUROBI_HOME:-}" ]; then
  for candidate in "$HOME"/.local/opt/gurobi*/linux64 /opt/gurobi*/linux64; do
    if [ -d "$candidate" ] && ls "$candidate"/lib/libgurobi*.so >/dev/null 2>&1; then
      GUROBI_HOME="$candidate"
      break
    fi
  done
fi
export GUROBI_HOME="${GUROBI_HOME:-}"

echo "== Verifying a solver is available =="
julia --project=. -e 'println("Julia ", VERSION); println("HiGHS available: ", Base.find_package("HiGHS") !== nothing)'

echo "== Running: julia --project=. main.jl $* =="
julia --project=. main.jl "$@"

echo "== Done =="
