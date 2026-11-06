# Command-line form of the targets_pairs step (v0.3.0: src/post.jl run_targets_pairs): the KK target windows of
# selected (sA,sB) pairs, outside the pipeline, so that several nodes can fill one run's targets/ in parallel.
#   julia --project=<env> --threads=T targets_run.jl RUN_TOML "a-b,c-d" [source_delta_max]
# Does not touch manifest.toml.  Give the same source_delta_max as the pipeline run.
include(joinpath(@__DIR__, "..", "src", "Tcc.jl"))
using .Tcc, TOML
t = TOML.parsefile(ARGS[1]); Tcc.expand_plan!(t); c = Tcc.pipeline_config(t)
pairs = [parse.(Int, split(p, "-")) for p in split(ARGS[2], ",")]
length(ARGS) >= 3 ? Tcc.run_targets_pairs(c, pairs; source_delta_max=parse(Float64, ARGS[3])) : Tcc.run_targets_pairs(c, pairs)
