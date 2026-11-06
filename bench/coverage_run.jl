# Command-line form of the coverage step (v0.3.0: logic in src/post.jl run_coverage; `bin/tcc.jl run.toml coverage`
# with [coverage] delta_cut, kmax does the same).
#   julia --threads=T coverage_run.jl run.toml DELTA_CUT [KMAX]
include(joinpath(@__DIR__, "..", "src", "Tcc.jl"))
using .Tcc, TOML
conf = TOML.parsefile(ARGS[1]); Tcc.expand_plan!(conf)
Tcc.run_coverage(conf; delta_cut=parse(Float64, ARGS[2]), kmax=length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3000)
