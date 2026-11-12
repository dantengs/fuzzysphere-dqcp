# Tcc unified entry:
#   julia --project=<env> --threads=T bin/tcc.jl run.toml [step ...] [--force=stage,...] [--data=DIR] [--set=key=value ...]
# Steps: basis tune spectrum targets kk ope summary (pipeline), coverage, targets_pairs, ope3, parity, parity_join,
# fullq, stageb, rawtarget.  Without steps: [plan] steps, else the pipeline stages (src/plan.jl).
#
# --set=table.key=value overrides one key of run.toml for this call (value in TOML syntax; a bare word is a string),
# e.g. split the Stage B windows of the largest N over several jobs without editing the plan:
#   bin/tcc.jl N12.toml stageb --set=stageb.step=windows --set=stageb.windows=[0,1]
#   bin/tcc.jl N12.toml stageb --set=stageb.step=windows --set=stageb.windows=[2]
#   bin/tcc.jl N12.toml stageb                                   # then KK and OPE on the finished windows
# The effective plan is written to <data root>/plans/<name>_set_<hash>.toml and run from there, so every output
# records the plan that produced it.
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
length(ARGS) >= 1 || error("usage: tcc.jl run.toml [step ...] [--force=stage,...] [--data=DIR] [--set=key=value ...]")
force = String[]; steps = String[]; data = nothing; sets = Pair{Vector{String},Any}[]
for a in ARGS[2:end]
    if startswith(a, "--force=")
        append!(force, split(a[9:end], ","))
    elseif startswith(a, "--data=")
        global data = a[8:end]
    elseif startswith(a, "--set=")
        kv = a[7:end]; i = findfirst('=', kv)
        i === nothing && error("--set needs table.key=value, got $a")
        key = String.(split(kv[1:i-1], ".")); raw = kv[i+1:end]
        all(!isempty, key) || error("--set: empty key in $a")
        val = try TOML.parse("v = " * raw)["v"] catch; raw end
        push!(sets, key => val)
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
if !isempty(sets)
    t = TOML.parsefile(config)
    for (key, val) in sets
        d = t
        for k in key[1:end-1]
            d = get!(d, k, Dict{String,Any}())
            d isa AbstractDict || error("--set: $(join(key, ".")) is not inside a table")
        end
        d[key[end]] = val
    end
    desc = join(["$(join(k, "."))=$(repr(v))" for (k, v) in sets], " ")
    eff = joinpath(root, "plans", splitext(basename(config))[1] * "_set_" * bytes2hex(Tcc.sha256(desc))[1:8] * ".toml")
    mkpath(dirname(eff))
    open(eff, "w") do io
        println(io, "# ", config, " with --set ", desc)
        TOML.print(io, t)
    end
    println("TCC_PLAN ", eff, "  (", desc, ")")
    config = eff
end
m = Tcc.run_steps(config; steps=isempty(steps) ? nothing : steps, force=force)
println("TCC_DONE version=", Tcc.TCC_VERSION, " stages=", join(sort(collect(keys(get(m, "stages", Dict())))), ","))
