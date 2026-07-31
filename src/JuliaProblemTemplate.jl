# Stub module so the project is loadable/precompilable as a package, which
# Pkg.test() requires. The template's files live at the repo root (mirroring
# the Python template's flat layout); this module just re-exposes them.
module JuliaProblemTemplate

include(joinpath(@__DIR__, "..", "problem.jl"))

end
