# Calculation plan and step runner (v0.3.0).  One TOML file states WHAT is computed for one N; the same
# code serves every N (the only N-dependent branch of the physics is the odd/even phase of the C2 cross
# density term, src/so5_c2_density_phase.jl; resource switches are chosen by size, src/resources.jl).
#
#   [plan]
#   sectors   = [[0,0],[0,1],[0,2],[1,1],[1,2],[2,2]]   integer (sA,sB) pairs, a <= b in {0,1,2}; or "all"
#   lmax      = 4                       spectrum blocks L = 0..lmax for every sector (KK sources; targets to lmax+1)
#   delta_max = 6.2                     KK target windows cover the sources up to this Delta (targets.source_delta_max)
#   half      = true                    (1/2,1/2) sector in the pipeline OPE stage (ope.half)
#   steps     = ["coverage", "ope3", "parity", "parity_join", "fullq", "stageb"]   default steps of bin/tcc.jl
#
# The plan expands into the stage keys of the pipeline (spectrum.blocks = sectors x 0..lmax in lexicographic
# order, targets.source_delta_max, ope.half); giving the same key twice with different values is an error.
# Without [plan] a run file is read exactly as before (all production run files of v0.2.x work unchanged).
#
# Steps (each one is skipped or resumed from its checkpoints when it already ran):
#   basis tune spectrum targets kk ope summary   the pipeline stages (src/pipeline.jl)
#   coverage    spectrum windows grown until every block reaches [coverage] delta_cut (kmax), then
#               targets -> kk -> ope -> summary (src/post.jl run_coverage)
#   targets_pairs  [targets_pairs] pairs = [[a,b],...]: target windows of these pairs only, outside the manifest
#   ope3        [ope3] three-point insertions (src/ope3.jl)        out default <run.out>_ope3
#   parity      [parity] physical parity (src/post.jl)             out default <run.out>_parity; targets, vacuum_raw
#   parity_join phi x phi towers of ope3 with P (tools/ope3_parity_join.py; [parity_join] python, else the
#               environment variable TCC_PYTHON, else "python" on Windows and "python3" elsewhere)
#   fullq       [fullq] full-Q integer-sector OPE (src/fullq.jl)   out default <run.out>_fullq
#   stageb      [stageb] half / (1,1) sector (src/stageb.jl)       sectors = ["half"] (or ["half","11"]);
#               per-sector tables [stageb.half], [stageb.11] override; out default <run.out>_stageb_<sector>
#   rawtarget   OPE onto the full raw Stage B windows (src/stageb.jl run_rawtarget), per stageb sector
#
# Every step writes plan.toml into its output directory (expanded plan, the step's options, code version);
# the pipeline stages record the plan in the run's manifest.toml.  tools/data_inventory.py builds the table
# of computed sectors and data from these records (and from the files of runs made before v0.3.0).

const PIPELINE_STEPS = ["basis", "tune", "spectrum", "targets", "kk", "ope", "summary"]
const POST_STEPS = ["coverage", "targets_pairs", "ope3", "parity", "parity_join", "fullq", "stageb", "rawtarget"]

function _set_once!(t, sec, key, v)
    d = get!(t, sec, Dict{String,Any}())
    haskey(d, key) && d[key] != v && error("plan gives $sec.$key = $v but the file also sets $(d[key])")
    d[key] = v
end

"""Expand [plan] into the stage keys (in place); returns the plan record (Dict) or nothing."""
function expand_plan!(t::AbstractDict)
    haskey(t, "plan") || return nothing
    p = t["plan"]; rec = Dict{String,Any}()
    if haskey(p, "sectors") || haskey(p, "lmax")
        sec = get(p, "sectors", "all")
        pairs = sec == "all" ? [(a, b) for a in 0:2 for b in a:2] : sort(unique([Tuple(Int.(s)) for s in sec]))
        all(((a, b),) -> 0 <= a <= b <= 2, pairs) || error("plan.sectors: pairs (a,b) with 0 <= a <= b <= 2")
        lmax = Int(get(p, "lmax", 2))
        blocks = [[a, b, L] for (a, b) in pairs for L in 0:lmax]
        old = get(get(t, "spectrum", Dict()), "blocks", nothing)
        old === nothing || [Int.(x) for x in old] == blocks || error("plan.sectors/lmax and spectrum.blocks differ")
        get!(t, "spectrum", Dict{String,Any}())["blocks"] = blocks
        rec["sectors"] = [collect(x) for x in pairs]; rec["lmax"] = lmax
    end
    haskey(p, "delta_max") && (_set_once!(t, "targets", "source_delta_max", Float64(p["delta_max"])); rec["delta_max"] = Float64(p["delta_max"]))
    haskey(p, "half") && (_set_once!(t, "ope", "half", Bool(p["half"])); rec["half"] = Bool(p["half"]))
    haskey(p, "steps") && (rec["steps"] = String.(p["steps"]))
    rec
end

_step_out(t, step, suffix=step) = abspath(get(get(t, step, Dict()), "out", t["run"]["out"] * "_" * suffix))

function _write_plan(od, plan, step, opts)
    mkpath(od)
    open(joinpath(od, "plan.toml"), "w") do io
        TOML.print(io, Dict("step" => step, "plan" => plan === nothing ? Dict{String,Any}() : plan,
            "options" => Dict{String,Any}(string(k) => v for (k, v) in opts),
            "code" => Dict(string(k) => v for (k, v) in pairs(producer_info())), "time" => string(round(Int, time()))))
    end
end

"""Stage B options of one sector: [stageb] keys overridden by [stageb.<sector>]."""
function stageb_options(t, sector)
    sb = get(t, "stageb", Dict{String,Any}())
    o = Dict{String,Any}(k => v for (k, v) in sb if !(v isa AbstractDict) && k != "sectors")
    merge!(o, get(sb, sector, Dict{String,Any}()))
    o["sector"] = sector
    single = length(get(sb, "sectors", ["half"])) == 1
    o["out"] = abspath(get(get(sb, sector, Dict()), "out", single && haskey(sb, "out") ? sb["out"] : t["run"]["out"] * "_stageb_" * sector))
    o["ope3"] = abspath(get(o, "ope3", _step_out(t, "ope3")))
    o
end

"""
    run_steps(config; steps=nothing, force=String[])

Run the given steps (default: plan.steps, else the pipeline stages of run.stages) of one TOML file.
"""
function run_steps(config; steps=nothing, force=String[])
    path = config isa AbstractString ? abspath(config) : ""
    t = config isa AbstractString ? TOML.parsefile(config) : deepcopy(config)
    plan = expand_plan!(t)
    todo = steps !== nothing ? collect(steps) : plan !== nothing && haskey(plan, "steps") ? plan["steps"] : nothing
    todo === nothing && return run_pipeline(t; force)
    for s in todo; s in PIPELINE_STEPS || s in POST_STEPS || error("unknown step $s"); end
    pipe = [s for s in todo if s in PIPELINE_STEPS]
    if "coverage" in todo
        cv = get(t, "coverage", Dict{String,Any}()); haskey(cv, "delta_cut") || error("[coverage] delta_cut is required")
        run_coverage(t; delta_cut=cv["delta_cut"], kmax=Int(get(cv, "kmax", 3000)),
                     stages=isempty(pipe) ? nothing : pipe)
    elseif !isempty(pipe)
        run_pipeline(t; stages=pipe, force)
    end
    out = abspath(t["run"]["out"]); man = joinpath(out, "manifest.toml")
    plan !== nothing && (!isempty(pipe) || "coverage" in todo) && isfile(man) && _manifest_update!(man, m -> (m["plan"] = plan))
    # post steps read the finished run with its final (coverage-grown) windows
    final = isfile(joinpath(out, "coverage_run.toml")) && haskey(t, "coverage") ? TOML.parsefile(joinpath(out, "coverage_run.toml")) : t
    c = pipeline_config(final); c["g"] = resolved_g(c)
    for s in todo
        s in PIPELINE_STEPS || s == "coverage" || println("STEP_BEGIN $s")
        if s == "targets_pairs"
            o = t["targets_pairs"]; run_targets_pairs(c, o["pairs"]; source_delta_max=get(o, "source_delta_max", get(c, "source_delta_max", nothing)))
        elseif s == "ope3"
            o = Dict{String,Any}(get(t, "ope3", Dict())); o["out"] = _step_out(t, "ope3")
            _write_plan(o["out"], plan, s, o); run_ope3_step(c, o; run_toml=path)
        elseif s == "parity"
            o = get(t, "parity", Dict()); od = _step_out(t, "parity")
            _write_plan(od, plan, s, o)
            run_parity(c, od; targets=get(o, "targets", true), vacuum_raw=get(o, "vacuum_raw", nothing))
        elseif s == "parity_join"
            # [parity_join] python, else TCC_PYTHON, else "python" on Windows and "python3" elsewhere
            py = get(get(t, "parity_join", Dict()), "python", get(ENV, "TCC_PYTHON", Sys.iswindows() ? "python" : "python3"))
            run(`$py $(joinpath(dirname(@__DIR__), "tools", "ope3_parity_join.py")) $(_step_out(t, "ope3")) $(_step_out(t, "parity"))`)
        elseif s == "fullq"
            o = Dict{String,Any}(get(t, "fullq", Dict())); o["out"] = _step_out(t, "fullq")
            get!(o, "parity_kk", joinpath(_step_out(t, "parity"), "parity_kk.csv"))
            _write_plan(o["out"], plan, s, o); run_fullq(c, o)
        elseif s in ("stageb", "rawtarget")
            for sector in String.(get(get(t, "stageb", Dict()), "sectors", ["half"]))
                o = stageb_options(t, sector)
                if s == "stageb"
                    _write_plan(o["out"], plan, s, o); run_stageb(c, o)
                else
                    run_rawtarget(c, o)
                end
            end
        end
        s in PIPELINE_STEPS || s == "coverage" || println("STEP_DONE $s")
    end
    isfile(man) ? TOML.parsefile(man) : Dict{String,Any}()
end
