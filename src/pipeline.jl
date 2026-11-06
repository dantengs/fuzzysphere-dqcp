# Unified entry (v0.1.7): one TOML file, one command, every stage resumable.
#
#   julia --project=<env> --threads=T Tcc_v0.0.1/bin/tcc.jl run.toml [stage ...]
#
# Stages (in order): basis -> tune -> spectrum -> targets -> kk -> ope -> summary.
# run.g may be a number or "tuned" (the stages after `tune` then use out/tune/selected_g.toml).
# Each stage writes atomic checkpoints under run.out and is skipped when it completed
# with the same configuration; out/manifest.toml records version, configuration digest,
# stage status, wall time and peak RSS.  Physics keys (N, g, blocks, nev, exchange,
# parity, segment basis) are frozen for an output directory: a changed value is refused.
#
# [run]      N, g (number or "tuned"), out, stages (default all except tune unless g = "tuned")
# [tune]     start, bounds, maxpoints, ftol, xtol, exchange
# [basis]    segment_checkpoint ("" = build into out/segment_spaces.ckpt), ph_half ("auto" = largest segment
#            sector >= 2000 configurations, src/resources.jl; v0.1.17 particle-hole half construction, src/ph.jl)
# [spectrum] lmax (2), blocks ("all" or [[sA,sB,L],...]), nev (12), exchange (true),
#            parity ("both"), parity_nev, block_nev, residency ("auto" = by segment basis size), concurrency (1),
#            tol (1e-10), post_bytes, async_io, extend (false; v0.1.14: larger nev/parity_nev/
#            block_nev/blocks on an existing output -> deflated extension, old vectors kept)
# [threads]  native (8), action (nthreads), workers (nthreads), tile (64), build_blas (16),
#            solver_blas (16)
# [targets]  initial_nev (12), max_nev (100), guard (1e-5), independent_check (true)
# [kk]       thresholds, cache (kk_terms disk cache directory), legacy_calibration
# [ope]      edges = [[rank, li, lf], ...] Gamma^2 insertions on (0,0)

const STAGES = ["basis", "tune", "spectrum", "targets", "kk", "ope", "summary"]
const PHYSICS_KEYS = ["N", "g", "blocks", "nev", "exchange", "parity", "parity_nev", "block_nev", "lmax", "tune"]
# v0.1.14: window keys may change for an existing output when spectrum.extend = true (deflated
# extension: stored eigenvectors are kept, only the missing states are computed)
const WINDOW_KEYS = ["blocks", "nev", "parity_nev", "block_nev"]

"""Resolve run.g = "tuned" from the tune stage output."""
function resolved_g(c)
    c["g"] isa Real && return Float64(c["g"])
    f = joinpath(c["out"], "tune", "selected_g.toml")
    isfile(f) || error("run.g = \"tuned\" but the tune stage has not produced $f")
    Float64(TOML.parsefile(f)["g"])
end

"""Flatten a pipeline TOML (Dict) into the cfg dictionary used by the stage functions."""
function pipeline_config(t::AbstractDict)
    run = t["run"]; c = Dict{String,Any}()
    c["N"] = Int(run["N"]); c["out"] = abspath(run["out"])
    c["g"] = run["g"] == "tuned" ? "tuned" : Float64(run["g"])
    c["stages"] = get(run, "stages", c["g"] == "tuned" ? STAGES : filter(!=("tune"), STAGES))
    c["tune"] = get(t, "tune", Dict())
    for sec in ("basis", "spectrum", "threads", "targets")
        for (k, v) in get(t, sec, Dict()); c[k] = v; end
    end
    c["kk"] = get(t, "kk", Dict()); c["ope"] = get(t, "ope", Dict())
    haskey(c["kk"], "cache") && (c["kk_cache"] = c["kk"]["cache"])
    c["lmax"] = Int(get(c, "lmax", 2)); c["nev"] = Int(get(c, "nev", 12))
    blocks = get(c, "blocks", "all")
    c["blocks"] = blocks == "all" ? [[a, b, L] for a in 0:2 for b in a:2 for L in 0:c["lmax"]] : [Int.(x) for x in blocks]
    c["residency_request"] = get(c, "residency", "auto"); c["residency"] = resolve_residency(c)   # v0.3.0: by basis size
    c["action"] = get(c, "action", Threads.nthreads()); c["workers"] = get(c, "workers", Threads.nthreads())
    c["segment_checkpoint"] = get(c, "segment_checkpoint", "")
    haskey(c, "native_dim_max") && (NATIVE_DIM_MAX[] = Int(c["native_dim_max"]))   # v0.1.17 kernel switch
    c
end

_physics(c) = Dict(k => c[k] for k in PHYSICS_KEYS if haskey(c, k))

# v0.1.13: a finished stage is skipped when the inputs that determine its *results* are unchanged
# (physics + the stage's own keys); thread counts, residency or concurrency never force a rerun.
const STAGE_KEYS = Dict(
    "basis" => ["segment_checkpoint"],
    "tune" => ["segment_checkpoint", "tune"],
    "spectrum" => ["segment_checkpoint", "tol"],
    "targets" => ["segment_checkpoint", "tol", "initial_nev", "max_nev", "guard", "independent_check", "target_parity", "pairs"],
    "kk" => ["segment_checkpoint", "tol", "initial_nev", "max_nev", "guard", "target_parity", "pairs", "kk"],
    "ope" => ["segment_checkpoint", "tol", "initial_nev", "max_nev", "target_parity", "pairs", "kk", "ope"],
    "summary" => ["segment_checkpoint", "tol", "initial_nev", "max_nev", "target_parity", "pairs", "kk", "ope"])
_stage_digest_v13(c, s) = _digest(merge(_physics(c), Dict(k => get(c, k, nothing) for k in STAGE_KEYS[s])))
# v0.1.14: basis and tune do not depend on the window keys (an extension never re-runs them)
_stage_digest(c, s) = s in ("basis", "tune") ?
    _digest(merge(filter(p -> !(first(p) in WINDOW_KEYS), _physics(c)), Dict(k => get(c, k, nothing) for k in STAGE_KEYS[s]))) :
    _stage_digest_v13(c, s)
_digest(x) = bytes2hex(sha256(sprint(show, sort(collect(x); by=first))))

# v0.1.16 (V15-7): per-stage provenance.  Every finished stage records the code that produced it
# and an inventory digest of its active outputs (checkpoint payload SHA-256s, other files by content;
# superseded / rejected / stale / abandoned checkpoints and attempt caches excluded).  A stage that
# is otherwise skippable is re-run when its outputs no longer match the inventory.
const STAGE_OUTPUTS = Dict("basis" => ["segment_spaces.ckpt"], "tune" => ["tune"], "spectrum" => ["blocks"],
                           "targets" => ["targets"], "kk" => ["kk"], "ope" => ["ope"],
                           "summary" => ["summary.tsv", "physics_report.toml", "physics_primaries.csv"])
const _INACTIVE = r"\.(k\d+|prepolish|rejected|stale)\.ckpt|\.superseded\.|\.abandoned|\.next(/|$)|(^|/)attempts(/|$)|\.pending-|\.stale(\.\d+)?(/|$)|\.replace-"

function _stage_inventory(out, s)
    lines = String[]
    for top in STAGE_OUTPUTS[s]
        root = joinpath(out, top); ispath(root) || continue
        if isfile(root)
            push!(lines, top * "\t" * file_digest(root)); continue
        end
        for (r, ds, fs) in walkdir(root)
            rel = join(splitpath(relpath(r, out)), '/')
            occursin(_INACTIVE, rel) && continue
            if "payload.jls" in fs && "sha256.txt" in fs
                push!(lines, rel * "\t" * strip(read(joinpath(r, "sha256.txt"), String))); continue
            end
            for f in fs
                fr = rel * "/" * f; occursin(_INACTIVE, fr) && continue
                endswith(f, ".tmp") && continue
                push!(lines, fr * "\t" * file_digest(joinpath(r, f)))
            end
        end
    end
    sort!(lines; by=codeunits)
    (digest=bytes2hex(sha256(join(lines, "\n"))), count=length(lines))
end

# v0.2.8 (audit A4): a finished stage is also bound to the content of the stages it reads.  Its record keeps
# an upstream digest (current output inventories of every earlier stage -- tune only when run.g = "tuned" --
# and the SHA of an external segment checkpoint); a skip requires it to match, so a re-run, extended or
# hand-edited upstream stage (also one filled outside the pipeline, e.g. bench/targets_run.jl) re-runs every
# stage below it.  The stage functions themselves reuse every checkpoint whose identity still matches.
_upstream(c, s) = [u for u in STAGES[1:findfirst(==(s), STAGES)-1] if u != "tune" || c["g"] == "tuned"]

function _upstream_digest(c, s, inv)
    lines = ["$u\t$(inv(u))" for u in _upstream(c, s)]
    seg = c["segment_checkpoint"]
    isempty(seg) || push!(lines, "segment_checkpoint\t" * strip(read(joinpath(seg, "sha256.txt"), String)))
    bytes2hex(sha256(join(lines, "\n")))
end

"""Records written before v0.2.8 carry no upstream digest: accept them only when every finished upstream
stage finished no later than `s` and its outputs still match the inventory it recorded then."""
function _upstream_legacy_ok(m, c, s, inv)
    t = tryparse(Int, string(get(m["stages"][s], "finished", "")))
    for u in _upstream(c, s)
        su = get(m["stages"], u, Dict()); get(su, "status", "") == "done" || continue
        tu = tryparse(Int, string(get(su, "finished", "")))
        t !== nothing && tu !== nothing && tu > t && return false
        haskey(su, "outputs_digest") && su["outputs_digest"] != inv(u) && return false
    end
    true
end

function _manifest_update!(path, f)
    m = isfile(path) ? TOML.parsefile(path) : Dict{String,Any}()
    f(m)
    tmp = path * ".tmp"; open(io -> TOML.print(io, m), tmp, "w"); mv(tmp, path; force=true)
    m
end

"""Spectrum rows of a finished run (all blocks)."""
function run_rows(out)
    rows = NamedTuple[]
    for d in readdir(joinpath(out, "blocks"))
        p = joinpath(out, "blocks", d, "result.ckpt"); isdir(p) && append!(rows, _checked_load(p).rows)
    end
    sort!(rows; by=r -> r.energy)
end

"""Tuning functional F(g) = 2(E_T - E0)/(E_J - E0) - 3 from ground (0,0,L0,C2=0), current (0,1,L1,C2=3), stress (0,0,L2,C2=0)."""
function tuning_F(rows)
    pick(a, b, L, c) = (x = [r for r in rows if (r.sa, r.sb, r.L) == (a, b, L) && abs(r.c2 - c) < 1e-5];
                        isempty(x) ? nothing : x[argmin([r.energy for r in x])])
    g0 = pick(0, 0, 0, 0.0); j = pick(0, 1, 1, 3.0); t = pick(0, 0, 2, 0.0)
    any(isnothing, (g0, j, t)) && return nothing
    (F=2 * (t.energy - g0.energy) / (j.energy - g0.energy) - 3, E0=g0.energy, EJ=j.energy, ET=t.energy)
end

"""
    run_pipeline(config; stages=nothing, force=String[])

`config` is a TOML path or a parsed Dict.  Returns the manifest.
"""
function run_pipeline(config; stages=nothing, force=String[])
    t = config isa AbstractString ? TOML.parsefile(config) : config
    c = pipeline_config(t); out = c["out"]; mkpath(out)
    man = joinpath(out, "manifest.toml")
    phys = _digest(_physics(c)); full = _digest(c)
    _manifest_update!(man, m -> begin
        if haskey(m, "physics_digest") && m["physics_digest"] != phys
            old = get(m, "physics", Dict{String,Any}()); new = Dict(k => string(v) for (k, v) in _physics(c))
            core = [k for k in union(keys(old), keys(new)) if !(k in WINDOW_KEYS)]
            changed = [k for k in union(keys(old), keys(new)) if get(old, k, nothing) != get(new, k, nothing)]
            if get(c, "extend", false) && !isempty(old) && all(get(old, k, nothing) == get(new, k, nothing) for k in core)
                hist = get(m, "window_history", Dict{String,Any}[])
                push!(hist, Dict{String,Any}("from" => Dict(k => get(old, k, "") for k in changed), "to" => Dict(k => get(new, k, "") for k in changed),
                                            "tcc_version" => string(TCC_VERSION), "time" => string(round(Int, time()))))
                m["window_history"] = hist
                println("WINDOW_EXTEND ", join(changed, ","))
            else
                error("$(out) belongs to a different physical configuration (N, g, blocks, nev, exchange, parity, lmax)" *
                      (all(k -> k in WINDOW_KEYS, changed) ? "; only window keys changed: set spectrum.extend = true to extend" : ""))
            end
        end
        m["physics_digest"] = phys; m["config_digest"] = full
        m["tcc_version"] = string(TCC_VERSION); m["julia"] = string(VERSION)
        m["code_tree"] = tree_digest(dirname(@__DIR__))
        m["git_commit"] = get(ENV, "TCC_GIT_COMMIT", git_commit(@__DIR__))
        m["physics"] = Dict(k => string(v) for (k, v) in _physics(c))
        haskey(m, "stages") || (m["stages"] = Dict{String,Any}())
        # v0.1.16: the global fields above describe the latest invocation only; keep the history
        inv = get(m, "invocations", Dict{String,Any}[])
        push!(inv, Dict{String,Any}("tcc_version" => m["tcc_version"], "code_tree" => m["code_tree"],
                                    "git_commit" => m["git_commit"], "time" => string(round(Int, time())),
                                    "stages" => stages === nothing ? "default" : join(stages, ",")))
        m["invocations"] = inv
    end)
    todo = stages === nothing ? [s for s in STAGES if s in c["stages"]] : collect(stages)
    spaces_ref = Ref{Any}(nothing); sha_ref = Ref{String}("")
    function spaces!()
        spaces_ref[] === nothing || return spaces_ref[]
        sp = joinpath(out, "segment_split")
        if get(c, "residency", "all") != "all" && isdir(joinpath(sp, "space_2"))       # v0.1.10: <= 2 spaces resident
            spaces_ref[] = SpaceStore(sp); sha_ref[] = _checked_load(joinpath(sp, "bases")).source_sha
            return spaces_ref[]
        end
        seg = c["segment_checkpoint"]
        if isempty(seg)
            own = joinpath(out, "segment_spaces.ckpt")
            spaces_ref[] = _checked_load(own).spaces; sha_ref[] = _ckpt_sha(own)
        else
            s = load_segment_spaces(seg); spaces_ref[] = s.spaces; sha_ref[] = s.sha
        end
        spaces_ref[]
    end
    invs = Dict{String,String}()                         # v0.2.8: output inventories, at most once per stage
    inv(u) = get!(() -> _stage_inventory(out, u).digest, invs, u)
    for s in todo
        s in STAGES || error("unknown stage $s")
        m = TOML.parsefile(man)
        st = get(m["stages"], s, Dict())
        sd = _stage_digest(c, s); up = _upstream_digest(c, s, inv)
        if get(st, "status", "") == "done" && !(s in force) &&
           (get(st, "stage_digest", get(st, "config_digest", "")) in (sd, _stage_digest_v13(c, s), full))
            if !haskey(st, "outputs_digest") || inv(s) == st["outputs_digest"]
                legacy = !haskey(st, "upstream_digest")
                if legacy ? _upstream_legacy_ok(m, c, s, inv) : st["upstream_digest"] == up
                    legacy && _manifest_update!(man, m -> (m["stages"][s]["upstream_digest"] = up;
                                                           m["stages"][s]["upstream_recorded"] = "backfilled by v$(TCC_VERSION)"))
                    pv = get(st, "producer_version", "")
                    println("STAGE_SKIP $s (done", pv == string(TCC_VERSION) ? "" : isempty(pv) ? ", producer not recorded" : " by v$pv",
                            haskey(st, "outputs_digest") ? "" : ", no output inventory", legacy ? ", upstream digest backfilled" : "", ")"); continue
                end
                println("STAGE_RERUN $s (an upstream stage changed after it finished)")
            else
                println("STAGE_RERUN $s (outputs differ from the inventory recorded when it finished)")
            end
        end
        # v0.3.0: "auto" residency is decided once the segment basis exists (after the basis stage)
        c["residency_request"] == "auto" && (c["residency"] = residency_auto(c))
        println("STAGE_BEGIN $s"); flush(stdout)
        t0 = time(); info = Dict{String,Any}()
        if s == "basis"
            if isempty(c["segment_checkpoint"])
                own = joinpath(out, "segment_spaces.ckpt")
                ph = _use_ph(c, c["N"])
                # v0.1.17b: the segment build is LAPACK-bound; env.sh sets OMP_NUM_THREADS=1, so set BLAS like
                # run_spectrum does (B20, N11 PH build: BLAS 1 128.8 s, 16 52.8 s, 64 83.6 s; N10 direct 10.5 -> 6.0 s)
                isdir(own) || BLAS.set_num_threads(get(c, "build_blas", 16))
                isdir(own) || _atomic_save(own, (meta=(N=c["N"], tcc=string(TCC_VERSION), ph_half=ph), spaces=build_segment_spaces(c["N"]; ph)))
                info["segment_checkpoint"] = own; info["sha256"] = _ckpt_sha(own); info["ph_half"] = ph
            else
                info["segment_checkpoint"] = c["segment_checkpoint"]
                info["sha256"] = strip(read(joinpath(c["segment_checkpoint"], "sha256.txt"), String))
            end
        elseif s == "tune"
            cfg = copy(c); cfg["segment_checkpoint"] = isempty(c["segment_checkpoint"]) ? joinpath(out, "segment_spaces.ckpt") : c["segment_checkpoint"]
            r = run_tune_stage(cfg, spaces!())
            info["g"] = r.g; info["F"] = r.F; info["points"] = r.selection["physical_points"]
        elseif s == "spectrum"
            cfg = copy(c); cfg["g"] = resolved_g(c)
            isempty(cfg["segment_checkpoint"]) && (cfg["segment_checkpoint"] = joinpath(out, "segment_spaces.ckpt"))
            r = run_spectrum(cfg)
            info["states"] = length(r.rows); info["residency"] = r.residency; info["concurrency"] = r.concurrency
            info["max_H_residual"] = maximum((x.rh for x in r.rows); init=0.0)
            info["max_C2_residual"] = maximum((x.rc for x in r.rows); init=0.0)
            info["gate"] = "PASS"                     # v0.1.16: run_spectrum throws on any gate failure
        elseif s == "targets"
            cfg = copy(c); cfg["g"] = resolved_g(c)
            isempty(cfg["segment_checkpoint"]) && (cfg["segment_checkpoint"] = joinpath(out, "segment_spaces.ckpt"))
            st2 = extend_targets(cfg)
            info["targets"] = length(st2); info["solved"] = count(v -> startswith(string(v), "solved"), values(st2))
        elseif s == "kk"
            r = run_kk_stage(merge(c, Dict("g" => resolved_g(c))), spaces!(), sha_ref[])
            info["alpha"] = r.calibration.alpha; info["coefficients"] = r.calibration.coefficients
            info["projected_states"] = r.nstates
        elseif s == "ope"
            info["edges"] = run_ope_stage(merge(c, Dict("g" => resolved_g(c))), spaces!(), sha_ref[])
        elseif s == "summary"
            rows = run_rows(out); F = tuning_F(rows); info["g"] = resolved_g(c)
            F === nothing || (info["F"] = F.F; info["E0"] = F.E0; info["EJ"] = F.EJ; info["ET"] = F.ET)
            open(joinpath(out, "summary.tsv"), "w") do io
                println(io, "sA\tsB\tL\tX\tindex\tenergy\tdelta\tC2\tH_res\tC2_res")
                sc = F === nothing ? NaN : 2 / (F.EJ - F.E0)
                for (a, b, L) in unique([(r.sa, r.sb, r.L) for r in rows])
                    blk = sort([r for r in rows if (r.sa, r.sb, r.L) == (a, b, L)]; by=r -> r.energy)
                    for (i, r) in enumerate(blk)
                        println(io, join((a, b, L, r.x, i, r.energy, F === nothing ? NaN : sc * (r.energy - F.E0), r.c2, r.rh, r.rc), "\t"))
                    end
                end
            end
            info["summary"] = joinpath(out, "summary.tsv")
            info["physics_report"] = physics_report(c, out, F)
        end
        wall = time() - t0
        pr = producer(); invd = _stage_inventory(out, s); invs[s] = invd.digest
        _manifest_update!(man, m -> (m["stages"][s] = merge(Dict{String,Any}(info), Dict{String,Any}(
            "status" => "done", "config_digest" => full, "stage_digest" => sd, "wall_s" => wall, "peak_rss_gib" => Sys.maxrss() / 2^30,
            "finished" => string(round(Int, time())), "producer_version" => pr.tcc, "producer_tree" => pr.code_tree,
            "producer_commit" => pr.git_commit, "outputs_digest" => invd.digest, "outputs_count" => invd.count,
            "upstream_digest" => up))))
        println("STAGE_DONE $s wall=$(round(wall; digits=2))"); flush(stdout)
    end
    TOML.parsefile(man)
end

"""
    physics_report(c, out, F) -> path

v0.1.13: one place for the physical results of a run (whatever stages exist):
coupling and tuning functional, KK calibration, KK-projected primaries at the report threshold
(Delta in units where Delta_J = 2), OPE estimates and the phi x phi singlet tower.
Writes out/physics_report.toml and out/physics_primaries.csv.
"""
function physics_report(c, out, F)
    rep = Dict{String,Any}("N" => c["N"], "g" => resolved_g(c), "tcc_version" => string(TCC_VERSION),
        "status" => "finite-N estimates; KK low-K2 states are primary candidates, not certified CFT data")
    F === nothing || merge!(rep, Dict("F" => F.F, "E0" => F.E0, "E_current" => F.EJ, "E_stress" => F.ET))
    sel = joinpath(out, "tune", "selected_g.toml"); isfile(sel) && (rep["tuning"] = TOML.parsefile(sel))
    cal = joinpath(out, "kk", "calibration.ckpt")
    if isdir(cal)
        k = _checked_load(cal)
        merge!(rep, Dict("alpha" => k.alpha, "E0_kk" => k.E0, "generator_coefficients" => k.coefficients,
                         "descendant_gap_error" => k.descendant_gap_error, "vacuum_leakage" => k.leakage))
    end
    thr = string(get(get(c, "kk", Dict()), "report_threshold", 0.25))
    ps = joinpath(out, "kk", "projected_states.csv"); prim = Any[]
    if isfile(ps)
        L = readlines(ps); h = [strip(x, '"') for x in split(L[1], ',')]
        col(n) = findfirst(==(n), h)
        for l in L[2:end]
            f = [strip(x, '"') for x in split(l, ',')]
            f[col("threshold")] == thr || continue
            push!(prim, (f[col("pair")], parse(Int, f[col("L")]), parse(Int, f[col("C2")]), parse(Float64, f[col("energy")]),
                         parse(Float64, f[col("delta")]), parse(Float64, f[col("K2")]), f[col("SO5_irrep_types")]))
        end
        sort!(prim; by=x -> (x[1], x[2], x[3], x[4]))
        open(joinpath(out, "physics_primaries.csv"), "w") do io
            println(io, "pair,L,C2,energy,Delta,K2,SO5_irrep_types")
            for p in prim; println(io, join(p, ",")); end
        end
        rep["primary_candidates"] = length(prim); rep["report_threshold"] = parse(Float64, thr)
        for (pair, Lv, c2) in unique([(p[1], p[2], p[3]) for p in prim])
            rep["Delta_$(pair)_L$(Lv)_C2_$(c2)"] = [p[5] for p in prim if (p[1], p[2], p[3]) == (pair, Lv, c2)][1:min(6, end)]
        end
    end
    ot = joinpath(out, "ope", "ope.toml"); isfile(ot) && (rep["ope"] = TOML.parsefile(ot))
    tw = joinpath(out, "ope", "phi_phi_towers.csv")
    if isfile(tw)
        rows = [split(l, ',') for l in readlines(tw)[2:end]]
        sg = [(parse(Float64, strip(r[6], '"')), parse(Float64, strip(r[9], '"'))) for r in rows
              if strip(r[1], '"') == "projected_t$(thr)" && strip(r[4], '"') == "0"]
        sort!(sg)
        rep["singlet_tower_Delta"] = first.(sg); rep["singlet_tower_phiphi"] = last.(sg)
    end
    open(io -> TOML.print(io, rep), joinpath(out, "physics_report.toml"), "w")
    joinpath(out, "physics_report.toml")
end
