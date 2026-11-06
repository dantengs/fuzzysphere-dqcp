# Inventory of computed sectors and data (v0.3.0), generated from the records that the runs themselves wrote,
# so that the data table of the paper is never copied by hand.
#
#   julia data_inventory.jl INDEX.toml OUT_PREFIX [DATA_ROOT]
#
# Directories in INDEX.toml are relative to DATA_ROOT (default: the current directory).
#
# INDEX.toml lists the output directories of each N (any entry may be omitted):
#   [[N]]
#   N = 5
#   run = "results/main/N5"                  pipeline run (manifest.toml, coverage*.toml, kk/, targets/)
#   ope3 = "results/ope3/N5"
#   parity = "results/parity/N5"
#   fullq = "results/fullq/N5"               directory holding run_manifest.toml of the full-Q step
#   stageb = ["results/stageb/half/N5", "results/stageb/11/N5"]   Stage B outputs (stageB_manifest.toml, input_binding.toml)
#   rawtarget = ["results/stageb/half/N5/rawtarget"]               rawtarget_manifest.toml
#   jobs = ""                                 free text (e.g. your job ids), copied to the table
# Example: configs/inventory_index.example.toml.
# Only text records are read (TOML, CSV); no checkpoint is loaded, and directory listings are not used for
# coverage because checkpoint payloads of finished runs may have been moved to an archive.
# Writes OUT_PREFIX.md (one section per N) and OUT_PREFIX.csv (N, step, item, value, source).
using TOML

rows = Any[]                                     # (N, step, item, value, source)
add!(N, step, item, value, src) = push!(rows, (N, step, item, string(value), src))
readcsv(p) = (L = readlines(p); h = split(L[1], ','); [Dict(zip(strip.(h, '"'), strip.(split(l, ','), '"'))) for l in L[2:end] if !isempty(l)])
pairs_of(blocks) = sort(unique([(b[1], b[2]) for b in blocks]))
lrange(ls) = isempty(ls) ? "-" : (u = sort!(unique(ls)); length(u) == u[end] - u[1] + 1 ? "L$(u[1])–$(u[end])" : "L" * join(u, ","))
fmt(x) = x isa AbstractFloat ? string(round(x; sigdigits=4)) : string(x)

function inventory_run(N, d)
    src(f) = joinpath(d, f)
    cfgp = isfile(src("coverage_run.toml")) ? src("coverage_run.toml") : src("run.toml")
    if isfile(cfgp)
        t = TOML.parsefile(cfgp); sp = get(t, "spectrum", Dict()); tg = get(t, "targets", Dict())
        add!(N, "run", "g", t["run"]["g"], cfgp)
        blocks = get(sp, "blocks", nothing)
        if blocks !== nothing
            for (a, b) in pairs_of(blocks)
                add!(N, "spectrum", "sector ($a,$b) blocks", lrange([x[3] for x in blocks if (x[1], x[2]) == (a, b)]), cfgp)
            end
        end
        add!(N, "spectrum", "nev per X sector", get(sp, "nev", "-"), cfgp)
        isempty(get(sp, "block_nev", Dict())) || add!(N, "spectrum", "block_nev", join(sort(["$k=$v" for (k, v) in sp["block_nev"]]), " "), cfgp)
        isempty(get(sp, "parity_nev", Dict())) || add!(N, "spectrum", "parity_nev", join(sort(["$k=$v" for (k, v) in sp["parity_nev"]]), " "), cfgp)
        if blocks !== nothing
            slmax = max(Int(get(sp, "lmax", 2)), maximum(x[3] for x in blocks))
            add!(N, "targets", "KK target windows (rule of src/targets.jl)", "every sector L0–$(slmax + 1), complete up to the sources with Delta <= $(get(tg, "source_delta_max", "max of window"))", cfgp)
        end
        add!(N, "targets", "initial_nev / max_nev", "$(get(tg, "initial_nev", "-")) / $(get(tg, "max_nev", "-"))", cfgp)
        add!(N, "basis", "segment checkpoint", isempty(get(get(t, "basis", Dict()), "segment_checkpoint", "")) ? "built in run" : t["basis"]["segment_checkpoint"], cfgp)
        add!(N, "ope", "half sector in pipeline OPE", get(get(t, "ope", Dict()), "half", false), cfgp)
    end
    if isfile(src("coverage.toml"))
        cv = TOML.parsefile(src("coverage.toml"))
        add!(N, "coverage", "delta_cut / kmax / extensions", "$(cv["delta_cut"]) / $(cv["kmax"]) / $(length(get(cv, "history", [])))", src("coverage.toml"))
        per = Dict{Tuple{Int,Int},Vector{Any}}()
        for (k, v) in cv["sectors"]
            m = match(r"^s(\d)_s(\d)_L(\d+)_x(-?\d)$", k); m === nothing && continue
            push!(get!(per, (parse(Int, m[1]), parse(Int, m[2])), Any[]), v)
        end
        for ((a, b), vs) in sort(collect(per); by=first)
            tops = [v["complete"] ? Inf : v["delta_top"] for v in vs]
            add!(N, "spectrum", "sector ($a,$b) states / min Delta_top", "$(sum(v["states"] for v in vs)) / $(fmt(minimum(tops)))", src("coverage.toml"))
        end
    end
    if isfile(src("kk/channel_coverage.csv"))
        cc = readcsv(src("kk/channel_coverage.csv"))
        for (a, b) in sort(unique([(r["sA"], r["sB"]) for r in cc]))
            cnt = Dict{String,Int}()
            for r in cc
                (r["sA"], r["sB"]) == (a, b) && parse(Float64, r["threshold"]) == 0.25 && (cnt[r["status"]] = get(cnt, r["status"], 0) + 1)
            end
            add!(N, "kk", "sector ($a,$b) channels at t=0.25 by status", join(sort(["$k=$v" for (k, v) in cnt]), " "), src("kk/channel_coverage.csv"))
        end
    end
    if isfile(src("physics_report.toml"))
        r = TOML.parsefile(src("physics_report.toml"))
        for k in ("F", "alpha", "E0", "vacuum_leakage", "descendant_gap_error", "primary_candidates")
            haskey(r, k) && add!(N, "summary", k, fmt(r[k]), src("physics_report.toml"))
        end
    end
    if isfile(src("manifest.toml"))
        m = TOML.parsefile(src("manifest.toml"))
        for (s, v) in sort(collect(get(m, "stages", Dict())); by=first)
            add!(N, "producer", s, "v$(get(v, "producer_version", "?")) $(first(string(get(v, "producer_commit", "")), 8))", src("manifest.toml"))
        end
        haskey(m, "plan") && add!(N, "plan", "plan (v0.3.0)", repr(m["plan"]), src("manifest.toml"))
    end
end

function inventory_ope3(N, d)
    f = joinpath(d, "ope3_input.toml")
    if isfile(f)
        o = get(TOML.parsefile(f), "opts", Dict())
        for k in ("int_ranks", "src_dmax", "pair_dmax", "half_lmax", "half_nev", "threshold", "half_only")
            haskey(o, k) && add!(N, "ope3", k, o[k], f)
        end
    end
    hb = joinpath(d, "half", "blocks")
    isdir(hb) && add!(N, "ope3", "half (1/2,1/2) windows", lrange([parse(Int, x[2:end]) for x in readdir(hb) if occursin(r"^L\d+$", x)]), hb)
    for c in sort(filter(x -> endswith(x, ".csv"), isdir(d) ? readdir(d) : String[]))
        add!(N, "ope3", "rows $c", count(!isempty, readlines(joinpath(d, c))) - 1, joinpath(d, c))
    end
end

function inventory_parity(N, d)
    f = joinpath(d, "parity_summary.toml"); isfile(f) || return
    s = TOML.parsefile(f)
    add!(N, "parity", "vacuum_raw_sign / max_residual", "$(s["vacuum_raw_sign"]) / $(fmt(s["max_residual"]))", f)
    add!(N, "parity", "KK modes / mixed at 0.25", "$(get(s, "kk_modes", "-")) / $(get(s, "kk_modes_mixed_t0.25", "-"))", f)
    p = joinpath(d, "parity.csv"); isfile(p) || return
    rs = readcsv(p)
    add!(N, "parity", "states with P (blocks + targets)", length(rs), p)
    # parity.csv lists every stored block and target window state (text record; checkpoint payloads may be archived)
    for kind in ("block", "target"), (a, b) in sort(unique([(r["sA"], r["sB"]) for r in rs if r["window"] == kind]))
        sel = [r for r in rs if r["window"] == kind && (r["sA"], r["sB"]) == (a, b)]
        add!(N, kind == "block" ? "spectrum" : "targets", "sector ($a,$b) $(kind) windows at the parity job: L / states / max Delta",
             "$(lrange([parse(Int, r["L"]) for r in sel])) / $(length(sel)) / $(fmt(maximum(parse(Float64, r["Delta"]) for r in sel)))", p)
    end
end

function inventory_fullq(N, d)
    f = joinpath(d, "run_manifest.toml"); isfile(f) || return
    m = TOML.parsefile(f)
    secs = get(m, "sectors_processed", [[0, 1], [0, 2], [1, 2], [0, 0], [1, 1], [2, 2]])  # absent = the original six
    add!(N, "fullq", "sectors (order)", join(["($(s[1]),$(s[2]))" for s in secs], " "), f)
    add!(N, "fullq", "rank_max / source, target Delta max", "$(m["rank_max"]) / $(m["source_Delta_max"]), $(m["target_Delta_max"])", f)
    add!(N, "fullq", "status / rows", "$(m["status"]) / $(get(m, "rows", "-"))", f)
end

function inventory_stageb(N, d)
    f = joinpath(d, "stageB_manifest.toml"); b = joinpath(d, "input_binding.toml")
    bind = isfile(b) ? TOML.parsefile(b) : Dict()
    sector = get(bind, "sector", "?")
    haskey(bind, "raw_dmax") && add!(N, "stageb $sector", "raw windows reach Delta", bind["raw_dmax"], b)
    if isfile(f)
        m = TOML.parsefile(f)
        add!(N, "stageb $sector", "raw windows / KK sources and OPE spins", "L0–$(m["KK_target_spin_max"]) / L0–$(m["spin_max"])", f)
        pc = joinpath(d, "projected_catalog.csv")
        isfile(pc) && add!(N, "stageb $sector", "KK-projected modes (t=0.25)", length(readcsv(pc)), pc)
        add!(N, "stageb $sector", "source / target Delta max", "$(m["projected_source_Delta_max"]) / $(m["projected_target_Delta_max"])", f)
        add!(N, "stageb $sector", "status / rows", "$(m["status"]) / $(m["rows"])", f)
        add!(N, "stageb $sector", "producer", get(m, "version", get(bind, "variant", first(get(bind, "script_sha", "?"), 8))), f)
    else
        w = joinpath(d, "windows")
        add!(N, "stageb $sector", "status", "no stageB_manifest.toml (window jobs only or unfinished); window files: " *
             (isdir(w) ? lrange([parse(Int, m[1]) for m in (match(r"^L(\d+)\.ckpt$", x) for x in readdir(w)) if m !== nothing]) : "-"), d)
    end
end

function inventory_rawtarget(N, d)
    f = joinpath(d, "rawtarget_manifest.toml"); isfile(f) || return
    m = TOML.parsefile(f)
    add!(N, "rawtarget $(m["sector"])", "edges / worst projection regression", "$(m["edges"]) / $(fmt(m["worst_projection_regression_rel"]))", f)
end

length(ARGS) >= 2 || error("usage: data_inventory.jl INDEX.toml OUT_PREFIX [DATA_ROOT]")
# relative directories of the index are read below DATA_ROOT (default: the current directory) and are written
# to the inventory as they stand, so the inventory carries no machine path
const INDEX, OUT = abspath(ARGS[1]), abspath(ARGS[2])
idx = TOML.parsefile(INDEX)
length(ARGS) >= 3 && cd(ARGS[3])
for e in idx["N"]
    N = e["N"]; haskey(e, "jobs") && add!(N, "jobs", "jobs (ledger)", e["jobs"], basename(INDEX))
    haskey(e, "run") && inventory_run(N, e["run"])
    haskey(e, "ope3") && inventory_ope3(N, e["ope3"])
    haskey(e, "parity") && inventory_parity(N, e["parity"])
    haskey(e, "fullq") && inventory_fullq(N, e["fullq"])
    for d in get(e, "stageb", String[]); inventory_stageb(N, d); end
    for d in get(e, "rawtarget", String[]); inventory_rawtarget(N, d); end
end
csvq(x) = occursin(r"[,\"\n]", x) ? "\"" * replace(x, "\"" => "\"\"") * "\"" : x
open(OUT * ".csv", "w") do io
    println(io, "N,step,item,value,source")
    for r in rows; println(io, join(csvq.(string.(r)), ",")); end
end
open(OUT * ".md", "w") do io
    println(io, "# Data inventory\n\nGenerated by `tools/data_inventory.jl` from the run records (", basename(INDEX), ").\n")
    for N in sort(unique(first.(rows)))
        println(io, "## N = $N\n\n| step | item | value |\n|---|---|---|")
        for r in rows
            r[1] == N && println(io, "| ", r[2], " | ", r[3], " | ", replace(r[4], "|" => "\\|"), " |")
        end
        println(io)
    end
end
println("INVENTORY_DONE rows=$(length(rows)) -> $(ARGS[2]).md/.csv")
