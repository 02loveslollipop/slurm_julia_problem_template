FROM julia:1.12

WORKDIR /app

# Gurobi WLS credentials are NOT baked into the image. Pass them at runtime:
#   docker run -e GRB_WLSACCESSID=... -e GRB_WLSSECRET=... -e GRB_LICENSEID=... IMAGE

# The full Gurobi optimizer (with libgurobi130.so) is downloaded at build
# time so Gurobi.jl can find it; credentials still come from runtime env vars.
# The tarball extracts to gurobi1302 (version with dots stripped).
ARG GUROBI_VERSION=13.0.2
RUN curl -fsSL https://packages.gurobi.com/13.0/gurobi${GUROBI_VERSION}_linux64.tar.gz \
      -o /tmp/gurobi.tar.gz \
 && tar xzf /tmp/gurobi.tar.gz -C /opt \
 && rm /tmp/gurobi.tar.gz
ENV GUROBI_HOME=/opt/gurobi1302/linux64

RUN apt-get update && apt-get install -y --no-install-recommends openmpi-bin libopenmpi-dev && rm -rf /var/lib/apt/lists/*

COPY Project.toml Manifest.toml .
COPY src src
RUN julia --project=. -e 'using Pkg; Pkg.instantiate()'

COPY . .

# Forwards CLI args straight to main.jl, e.g.:
#   docker run --rm -v "$PWD/out:/app/out" IMAGE --time-limit 30 --out out/results.csv
ENTRYPOINT ["julia", "--project=.", "main.jl"]
