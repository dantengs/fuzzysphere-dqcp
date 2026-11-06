# Tcc unified entry:  julia --project=<env> --threads=T bin/tcc.jl run.toml [step ...] [--force=stage,...] [--data=DIR]
# Steps: basis tune spectrum targets kk ope summary (pipeline), coverage, targets_pairs, ope3, parity, parity_join,
# fullq, stageb, rawtarget.  Without steps: [plan] steps, else the pipeline stages (src/plan.jl).
#
# Data root.  Every relative path of run.toml (run.out, the out keys of the steps, kk.cache, basis.segment_checkpoint,
# stageb.prelim_top_window, ...) is resolved against the data root, which is the first of
#   1. --data=DIR on the command line,
#   2. the environment variable TCC_DATA,
#   3. [paths] data_root in user.toml at the repository root (copy user.example.toml; user.toml is not tracked by git),
#   4. the current directory.
# Absolute paths in run.toml are used as they are.  The run file itself is read from where it is given.
include(joinpath(@__DIR__, "..", "src", "Tcc.jl"))
using .Tcc, TOML
length(ARGS) >= 1 || error("usage: tcc.jl run.toml [step ...] [--force=stage,...] [--data=DIR]")
force = String[]; steps = String[]; data = nothing
for a in ARGS[2:end]
    if startswith(a, "--force=")
        append!(force, split(a[9:end], ","))
    elseif startswith(a, "--data=")
        global data = a[8:end]
    else
        push!(steps, a)
    end
end

function data_root(arg)
    arg !== nothing && return abspath(expanduser(arg))
    v = strip(get(ENV, "TCC_DATA", ""))
    isempty(v) || return abspath(expanduser(v))
    repo = normpath(joinpath(@__DIR__, ".."))
    user = joinpath(repo, "user.toml")
    if isfile(user)
        r = strip(string(get(get(TOML.parsefile(user), "paths", Dict()), "data_root", "")))
        isempty(r) || return abspath(joinpath(repo, expanduser(r)))
    end
    pwd()
end

config = abspath(ARGS[1])
root = data_root(data)
mkpath(root); cd(root)
println("TCC_DATA_ROOT ", root)
m = Tcc.run_steps(config; steps=isempty(steps) ? nothing : steps, force=force)
println("TCC_DONE version=", Tcc.TCC_VERSION, " stages=", join(sort(collect(keys(get(m, "stages", Dict())))), ","))
