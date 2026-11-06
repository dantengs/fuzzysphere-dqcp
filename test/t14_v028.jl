# v0.2.8: external audit A3, A4 (A2: t15_stageb_restart.jl).
#   A4  a stage whose upstream outputs changed after it finished is re-run, not skipped; a record written
#       before v0.2.8 (no upstream digest) is accepted only when its upstream stages finished earlier and
#       still match their inventories, and the digest is then backfilled
#   A3  half-integer ope3 edges follow their windows: windows recomputed at another g with the same number
#       of states get new edges (v0.2.7 reused the old ones); a pre-v0.2.8 edge identity is accepted only
#       while it provably belongs to the current windows
# usage: julia --threads=T t14_v028.jl N out
include(joinpath(@__DIR__, "..", "src", "Tcc.jl"))
using .Tcc, FuzzifiED, TOML, Printf, LinearAlgebra
FuzzifiED.SilentStd = true
N = parse(Int, ARGS[1]); root = ARGS[2]; g = 0.95
mkpath(root)
ok = true
check(c, msg) = (println(c ? "PASS " : "FAIL ", msg); global ok &= c)
function captured(f)                       # stdout of f() as a String (also echoed)
    path, io = mktemp()
    redirect_stdout(f, io); close(io)
    s = read(path, String); print(s); s
end
th = Dict("native" => min(4, Threads.nthreads()), "tile" => 16)

# ---- A4. pipeline upstream digest ------------------------------------------------------------------
dP = joinpath(root, "a4_N$N"); rm(dP; recursive=true, force=true)
conf = Dict("run" => Dict("N" => N, "g" => g, "out" => dP),
            "spectrum" => Dict{String,Any}("nev" => 8, "exchange" => true, "tol" => 1e-11),
            "threads" => th, "targets" => Dict("initial_nev" => 12, "max_nev" => 120),
            "kk" => Dict("cache" => joinpath(root, "kk_cache")), "ope" => Dict("half" => false))
man = run_pipeline(conf); mp = joinpath(dP, "manifest.toml"); done = sort(collect(keys(man["stages"])))
skips(log) = sort([String(m[1]) for m in eachmatch(r"STAGE_SKIP (\w+)", log)])
upreruns(log) = sort([String(m[1]) for m in eachmatch(r"STAGE_RERUN (\w+) \(an upstream", log)])
check(all(haskey(man["stages"][s], "upstream_digest") for s in done), "every finished stage records an upstream digest ($(join(done, ",")))")
log = captured(() -> run_pipeline(conf))
check(skips(log) == done, "unchanged rerun skips every stage")

# the targets stage changes and is re-run on its own; the next full run must re-run what reads it
write(joinpath(dP, "targets", "v028_marker.txt"), "upstream change\n")
log = captured(() -> run_pipeline(conf; stages=["targets"]))
check(occursin("STAGE_RERUN targets (outputs differ", log), "targets re-run after its outputs changed")
below = [s for s in ("kk", "ope", "summary") if s in done]
log = captured(() -> run_pipeline(conf))
check(upreruns(log) == sort(below) && issubset(["basis", "spectrum", "targets"], skips(log)),
      "stages below a re-run targets stage are re-run (upstream changed: $(join(upreruns(log), ","))); basis/spectrum/targets skipped")
log = captured(() -> run_pipeline(conf))
check(skips(log) == done, "after that, every stage skips again")

# legacy records (no upstream digest)
strip_up!(ss) = Tcc._manifest_update!(mp, m -> foreach(s -> delete!(m["stages"][s], "upstream_digest"), ss))
strip_up!(done)
log = captured(() -> run_pipeline(conf)); m4 = TOML.parsefile(mp)
check(skips(log) == done && count("upstream digest backfilled", log) == length(done) &&
      all(m4["stages"][s]["upstream_digest"] == man["stages"][s]["upstream_digest"] || s in ("kk", "ope", "summary") for s in done),
      "consistent legacy records are skipped and their upstream digest backfilled")
strip_up!(["kk"])
Tcc._manifest_update!(mp, m -> (m["stages"]["targets"]["finished"] = string(parse(Int, m["stages"]["kk"]["finished"]) + 10)))
log = captured(() -> run_pipeline(conf; stages=["kk"]))
check(upreruns(log) == ["kk"], "legacy kk record older than its targets stage is re-run")
strip_up!(["ope"])
write(joinpath(dP, "kk", "v028_marker.txt"), "upstream change\n")
log = captured(() -> run_pipeline(conf; stages=["ope"]))
check(upreruns(log) == ["ope"], "legacy ope record whose kk outputs no longer match the kk inventory is re-run")

# ---- A3. half-integer ope3 edges --------------------------------------------------------------------
let lib = SegLibrary(builder=:tiled, tile_width=16, workers=Threads.nthreads(), opmat_threads=th["native"]), g1 = 0.95, g2 = 1.05, hl = 2
    half(od, gg) = Tcc._ope3_half!(od, N, gg, lib, hl, 12; act=Threads.nthreads(), image_bytes=1e8, cache="")
    dirs = Dict(k => joinpath(root, "a3_$(k)_N$N") for k in ("reuse", "fresh", "legacy"))
    foreach(d -> rm(d; recursive=true, force=true), values(dirs))
    nrm(h) = [norm(h.amp[l]) for l in 0:hl]
    en(h) = reduce(vcat, [h.win[l].energy for l in 0:hl])
    h1 = half(dirs["reuse"], g1); h2 = half(dirs["reuse"], g2); f2 = half(dirs["fresh"], g2)
    same_count = all(length(h1.win[l].energy) == length(h2.win[l].energy) for l in 0:hl)
    agree = max(maximum(abs.(nrm(h2) - nrm(f2))), maximum(abs.(en(h2) - en(f2))))
    stale = count(f -> occursin(".stale.", f), readdir(joinpath(dirs["reuse"], "half", "edges")))
    check(same_count && agree < 1e-9 && all(h2.amp[l] != h1.amp[l] for l in 0:hl) && stale > 0,
          @sprintf("g changed in the same directory (same state counts): edges recomputed (%d moved aside), |amp| and E equal a fresh run to %.1e", stale, agree))

    c1 = half(dirs["legacy"], g1)
    hd = joinpath(dirs["legacy"], "half"); ed = joinpath(hd, "edges")
    segsha = String(Tcc._ckpt_sha(joinpath(hd, "segment.ckpt")))
    oldsha(l) = segsha * "|L$l|" * string(length(c1.win[l].energy))
    nlegacy = 0
    for f in readdir(ed)                       # rewrite every edge under the identity v0.2.7 gave it
        p = joinpath(ed, f); e = Tcc._checked_load(p); id = e.identity
        m = match(r"^half_gamma2_(rev_)?(\d+)$", id.tag); l = parse(Int, m[2])
        old = m[1] === nothing ? (tag=id.tag, src=oldsha(0), tgt=oldsha(l), version=id.version) :
              (tag=id.tag, src=oldsha(l) * id.src[findlast('|', id.src):end], tgt=oldsha(0), version=id.version)
        rm(p; recursive=true); Tcc._atomic_save(p, merge(e, (identity=old,))); nlegacy += 1
    end
    r = Ref{Any}()
    log1 = captured(() -> (r[] = half(dirs["legacy"], g1))); c1b = r[]
    check(count("OPE3_EDGE_LEGACY", log1) == nlegacy && !occursin("OPE3_EDGE ", log1) && all(c1b.amp[l] == c1.amp[l] for l in 0:hl),
          "pre-v0.2.8 edges of unchanged windows are reused ($nlegacy edges)")
    log2 = captured(() -> (r[] = half(dirs["legacy"], g2))); c2 = r[]
    agree2 = max(maximum(abs.(nrm(c2) - nrm(f2))), maximum(abs.(en(c2) - en(f2))))
    check(!occursin("OPE3_EDGE_LEGACY", log2) && agree2 < 1e-9,
          @sprintf("pre-v0.2.8 edges are not reused once the windows were recomputed at another g (fresh-run agreement %.1e)", agree2))
end

println(ok ? "V028_PASS N=$N" : "V028_FAIL N=$N")
