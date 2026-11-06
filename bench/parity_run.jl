# Command-line form of the parity step (v0.3.0: src/post.jl run_parity; `bin/tcc.jl run.toml parity` with [parity]).
#   julia --project=<env> --threads=T parity_run.jl RUN_TOML OUT_DIR [targets] [vacuum_raw=±1]
include(joinpath(@__DIR__, "..", "src", "Tcc.jl"))
using .Tcc, TOML
c = Tcc.pipeline_config(TOML.parsefile(ARGS[1])); c["g"] = Tcc.resolved_g(c)
va = [a for a in ARGS[3:end] if startswith(a, "vacuum_raw=")]
Tcc.run_parity(c, ARGS[2]; targets="targets" in ARGS[3:end], vacuum_raw=isempty(va) ? nothing : parse(Int, split(va[1], "=")[2]))
