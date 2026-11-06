# Command-line form of the ope3 step (v0.3.0: src/post.jl run_ope3_step; `bin/tcc.jl run.toml ope3` with [ope3]).
#   julia --project=<env> --threads=T ope3_run.jl RUN_TOML OUT_DIR [OPTS_TOML]
# OPTS_TOML keys: src_dmax, pair_dmax, int_ranks, half_lmax, half_nev, half_only, threshold, image_bytes, qg_gate, cache.
include(joinpath(@__DIR__, "..", "src", "Tcc.jl"))
using .Tcc, TOML
length(ARGS) >= 2 || error("usage: ope3_run.jl RUN_TOML OUT_DIR [OPTS_TOML]")
c = Tcc.pipeline_config(TOML.parsefile(ARGS[1])); c["g"] = Tcc.resolved_g(c)
o = length(ARGS) >= 3 ? TOML.parsefile(ARGS[3]) : Dict{String,Any}(); o["out"] = abspath(ARGS[2])
Tcc.run_ope3_step(c, o; run_toml=abspath(ARGS[1]))
