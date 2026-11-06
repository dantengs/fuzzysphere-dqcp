# Command-line form of the stageb step (v0.3.0: src/stageb.jl run_stageb; replaces the scripts xy_half_extend.jl,
# xy_half_resume.jl and xy_half_lean.jl).  `bin/tcc.jl run.toml stageb` with [stageb] does the same.
#   julia --project=<env> --threads=T stageb_run.jl RUN_TOML OUT_DIR OPE3_DIR [half|11] [key=value ...]
# key=value: any [stageb] key, e.g. lmax=3 max_nev=384 step=windows windows=0,1 rawtarget=true.
include(joinpath(@__DIR__, "..", "src", "Tcc.jl"))
using .Tcc, TOML
length(ARGS) >= 3 || error("usage: stageb_run.jl RUN_TOML OUT_DIR OPE3_DIR [half|11] [key=value ...]")
c = Tcc.pipeline_config(TOML.parsefile(ARGS[1])); c["g"] = Tcc.resolved_g(c)
o = Dict{String,Any}("out" => ARGS[2], "ope3" => ARGS[3], "sector" => length(ARGS) >= 4 && !occursin('=', ARGS[4]) ? ARGS[4] : "half")
for a in ARGS[4:end]
    occursin('=', a) || continue
    k, v = split(a, '='; limit=2)
    o[k] = something(tryparse(Int, v), tryparse(Float64, v), v == "true" ? true : v == "false" ? false : String(v))
end
Tcc.run_stageb(c, o)
