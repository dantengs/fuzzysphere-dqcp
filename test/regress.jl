# Local regression for every Tcc v0.1.x (N = 4, 5, 6):
#   1. library/tiled segment operators == BuildSegOperators (all 6 pairs, H and C2), elementwise
#   2. full 18-block spectrum (k=12) with and without exchange sectors; residual gates
#   3. KK (6 edges) and Gamma^2 (2 edges) arrays on (0,0): Tcc edge_arrays == frozen reference
#   4. optional comparison with a golden directory from an earlier version
# usage: julia --threads=T regress.jl N out [golden_dir]
include(joinpath(@__DIR__, "..", "src", "Tcc.jl"))
using .Tcc, FuzzifiED, FuzzifiED.SO3lver, LinearAlgebra, Serialization, TOML, Printf
FuzzifiED.SilentStd = true
N = parse(Int, ARGS[1]); out = ARGS[2]; golden = length(ARGS) >= 3 ? ARGS[3] : ""
g = 0.95
mkpath(out); fails = String[]
check(ok, msg) = (ok || push!(fails, msg); println(ok ? "ok   " : "FAIL ", msg))
FuzzifiED.NumThreads = min(8, Threads.nthreads())
spaces = build_segment_spaces(N); tpl = hamiltonian_template(N)
# 1
lib = SegLibrary(builder=:tiled, tile_width=16, workers=Threads.nthreads(), opmat_threads=FuzzifiED.NumThreads)
worst = 0.0
for a in 0:2, b in a:2
    ss = [spaces[a+1], spaces[b+1]]
    for cpd in (tpl.template, tpl.c2)
        R = BuildSegOperators(ss, cpd); S = segops!(lib, ss, cpd)
        global worst = max(worst, maximum(segop_difference(R[k], S[k]).worst for k in eachindex(R)))
    end
end
check(worst < 1e-12, @sprintf("segment operators vs BuildSegOperators: %.2e", worst))
# 2
blocks = [[a, b, L] for a in 0:2 for b in a:2 for L in 0:2]
spec = Dict{Bool,Any}()
for X in (false, true)
    d = joinpath(out, X ? "spec_X" : "spec_full"); rm(d; recursive=true, force=true)
    r = run_spectrum(Dict{String,Any}("N"=>N, "g"=>g, "nev"=>12, "out"=>d, "blocks"=>blocks, "exchange"=>X,
                                      "native"=>FuzzifiED.NumThreads, "action"=>Threads.nthreads(), "tile"=>16, "tol"=>1e-11))
    spec[X] = r.rows
    check(maximum(x.rh for x in r.rows) < 1e-8 && maximum(x.rc for x in r.rows) < 1e-7,
          @sprintf("spectrum X=%s residuals rH=%.1e rC=%.1e", X, maximum(x.rh for x in r.rows), maximum(x.rc for x in r.rows)))
end
# residency modes and concurrency (v0.1.5): every mode reproduces the X=true rows
if isdefined(Tcc, :resolve_residency)
    function rows_equal(A, B)
        k(x) = (x.sa, x.sb, x.L, x.x); d = 0.0; dc = 0.0; ok = Set(k.(A)) == Set(k.(B))
        for kb in unique(k.(A))
            a1 = sort([x for x in A if k(x) == kb]; by=x -> x.energy); b1 = sort([x for x in B if k(x) == kb]; by=x -> x.energy)
            ok &= length(a1) == length(b1)
            for (x, y) in zip(a1, b1); d = max(d, abs(x.energy - y.energy)); dc = max(dc, abs(x.c2 - y.c2)); end
        end
        ok, d, dc
    end
    seg = joinpath(out, "spec_X", "segment_spaces.ckpt")
    for (mode, conc) in (("all", 3), ("pair", 1), ("minimal", 1))
        dR = joinpath(out, "spec_$(mode)_$conc"); rm(dR; recursive=true, force=true)
        cfgR = Dict{String,Any}("N"=>N, "g"=>g, "nev"=>12, "out"=>dR, "blocks"=>blocks, "exchange"=>true,
                                "segment_checkpoint"=>seg, "residency"=>mode, "concurrency"=>conc,
                                "native"=>FuzzifiED.NumThreads, "action"=>Threads.nthreads(), "tile"=>16, "tol"=>1e-11)
        r = run_spectrum(cfgR)
        ok, d, dc = rows_equal(spec[true], r.rows)
        check(ok && d < 1e-9 && dc < 1e-8, @sprintf("residency %s concurrency %d == reference rows: dE %.1e dC2 %.1e (space loads %d)", mode, conc, d, dc, r.space_loads))
        if mode == "minimal"     # restart: drop two results; eig.ckpt is reused, rows identical
            neig() = count(l -> startswith(l, "eig_"), eachline(joinpath(dR, "stages.tsv")))
            n0 = neig()
            for key in ("s1_s1_L0", "s0_s2_L1"); rm(joinpath(dR, "blocks", key, "result.ckpt"); recursive=true); end
            r2 = run_spectrum(cfgR)
            ok2, d2, _ = rows_equal(r.rows, r2.rows)
            check(ok2 && d2 == 0 && neig() == n0, "minimal-residency restart reuses eig.ckpt (new eig stages $(neig() - n0), dE $(d2))")
        end
    end
end
# compare lowest common window per block (full k=12 vs sectors 12+12)
dmax = 0.0
for a in 0:2, b in a:2, L in 0:2
    E0 = sort([x.energy for x in spec[false] if (x.sa, x.sb, x.L) == (a, b, L)])
    E1 = sort([x.energy for x in spec[true] if (x.sa, x.sb, x.L) == (a, b, L)])
    n = min(length(E0), length(E1), a == b ? 12 : length(E0))
    n == 0 && continue
    global dmax = max(dmax, maximum(abs.(E0[1:n] - E1[1:n])))
end
check(dmax < 1e-9, @sprintf("exchange-sector spectrum == full-block spectrum: %.2e", dmax))
# parity selection (v0.1.4): plus-only run == the X=+1 rows of the both-parity run
if isdefined(Tcc, :sector_want)
    dP = joinpath(out, "spec_plus"); rm(dP; recursive=true, force=true)
    rp = run_spectrum(Dict{String,Any}("N"=>N, "g"=>g, "nev"=>12, "out"=>dP, "blocks"=>[[s, s, L] for s in 0:2 for L in 0:2],
                                       "exchange"=>true, "parity"=>"plus", "parity_nev"=>Dict("s0_s0_L0"=>[5, 3]),
                                       "native"=>FuzzifiED.NumThreads, "action"=>Threads.nthreads(), "tile"=>16, "tol"=>1e-11,
                                       "post_bytes"=>1)).rows
    dp = 0.0; okn = true
    for s in 0:2, L in 0:2, sg in (1, -1)
        A = sort([x.energy for x in spec[true] if (x.sa, x.sb, x.L, x.x) == (s, s, L, sg)])
        B = sort([x.energy for x in rp if (x.sa, x.sb, x.L, x.x) == (s, s, L, sg)])
        want = (s, L) == (0, 0) ? (sg == 1 ? 5 : 3) : (sg == 1 ? 12 : 0)
        global okn &= length(B) == min(want, length(A))     # length(A) = min(12, sector dim)
        m = min(length(A), length(B))
        global dp = max(dp, maximum(abs.(A[1:m] - B[1:m]); init=0.0))
    end
    check(dp < 1e-9 && okn, @sprintf("parity selection (plus / parity_nev, chunked post-processing) == both-parity rows: %.2e, counts %s", dp, okn))
end
# 3  Eigenvectors are only meaningful in the segment basis they were computed in: the sign/phase
#    gauge of segment states is not reproducible between two BuildSegSpaces calls, so the
#    edges must use the segment_spaces checkpoint written by the spectrum run itself.
spaces = Tcc._checked_load(joinpath(out, "spec_full", "segment_spaces.ckpt")).spaces
vec(L) = (p = open(deserialize, joinpath(out, "spec_full", "blocks", "s0_s0_L$L", "eig.ckpt", "payload.jls")); p.vectors)
ss = [spaces[1], spaces[1]]; cps = Dict(L => BuildCompSpace(ss, [2N, 0, 0], 2L) for L in 0:2)
kk = kk_terms(N)
edges = [("kk_01", kk, 1, 0, 1), ("kk_10", kk, 1, 1, 0), ("kk_11", kk, 1, 1, 1), ("kk_12", kk, 1, 1, 2), ("kk_21", kk, 1, 2, 1),
         ("kk_22", kk, 1, 2, 2), ("g2_00", gamma2_terms(N, 0), 0, 0, 0), ("g2_02", gamma2_terms(N, 2), 2, 0, 2)]
ew = 0.0; earrays = Dict{String,Any}()
for (nm, terms, r, li, lf) in edges
    S = vec(li); T = vec(lf)
    A = edge_arrays(lib, ss, cps[li], cps[lf], terms, 2r, S, T; workers=Threads.nthreads(), chunk=5)
    B = edge_arrays_reference(ss, cps[li], cps[lf], terms, 2r, S, T; native=FuzzifiED.NumThreads, action=Threads.nthreads())
    e = max(maximum(maximum(abs.(x - y)) for (x, y) in zip(A.overlaps, B.overlaps)), maximum(abs.(A.image_gram - B.image_gram)))
    global ew = max(ew, e); earrays[nm] = A
end
check(ew < 1e-12, @sprintf("KK/Gamma2 edges vs frozen reference: %.2e", ew))
isdefined(lib, :scaled_hits) && println("library builds=$(lib.builds) hits=$(lib.hits) scaled_hits=$(lib.scaled_hits)")
# normalisation-aware sharing: a scaled operator must equal the directly built one
if isdefined(Tcc, :scaled_segop)
    d0 = findfirst(c -> c.amd[1] !== :Identity && c.amd[2] !== :Identity, tpl.template)   # a cross-segment channel
    c0 = tpl.template[d0]; amd = c0.amd[1]; ll = c0.ch[1, 1]; sec = c0.sec[:, 1]
    samd = FuzzifiED.AngModes(amd.l2m, (l2, m2) -> -2.75 * amd.get_comp(l2, m2))
    lib2 = SegLibrary(builder=:tiled, tile_width=16, workers=Threads.nthreads(), opmat_threads=FuzzifiED.NumThreads)
    segop!(lib2, spaces[1], spaces[1], amd, ll, sec)
    S2 = segop!(lib2, spaces[1], spaces[1], samd, ll, sec)
    R2 = BuildSegOperator(spaces[1], spaces[1], samd, ll, sec)
    check(lib2.scaled_hits == 1 && segop_difference(R2, S2).worst < 1e-12,
          @sprintf("scaled library hit == direct build: %.2e (scaled hits %d)", segop_difference(R2, S2).worst, lib2.scaled_hits))
end
# KK terms disk cache: second call is a load and identical
if hasmethod(kk_terms, Tuple{Int}, (:cache,))
    cdir = joinpath(out, "kk_cache"); rm(cdir; recursive=true, force=true)
    k1 = kk_terms(N; cache=cdir); k2 = kk_terms(N; cache=cdir)
    same = all(length(a) == length(b) && all(x.coeff == y.coeff && x.ch == y.ch && x.sec == y.sec for (x, y) in zip(a, b)) for (a, b) in zip(k1, k2))
    check(same && length(readdir(cdir)) == 1, "KK terms disk cache round trip")
end
# 5. KK target windows on the library (v0.1.6): every state of the block below the window
#    frontier is present, nothing else below it, C2 integral
if isdefined(Tcc, :extend_targets)
    dT = joinpath(out, "spec_X")
    rm(joinpath(dT, "targets"); recursive=true, force=true)
    stt = extend_targets(Dict{String,Any}("N"=>N, "g"=>g, "out"=>dT, "initial_nev"=>4, "max_nev"=>80, "native"=>FuzzifiED.NumThreads,
                                          "action"=>Threads.nthreads(), "tile"=>16, "tol"=>1e-11))
    dR = joinpath(out, "spec_ref"); rm(dR; recursive=true, force=true)
    ref = run_spectrum(Dict{String,Any}("N"=>N, "g"=>g, "nev"=>80, "out"=>dR, "blocks"=>[[a, b, L] for a in 0:2 for b in a:2 for L in 0:3],
                                        "exchange"=>false, "segment_checkpoint"=>joinpath(dT, "segment_spaces.ckpt"),
                                        "native"=>FuzzifiED.NumThreads, "action"=>Threads.nthreads(), "tile"=>16, "tol"=>1e-11)).rows
    dt = 0.0; okt = true; nsolved = 0
    for f in filter(x -> endswith(x, ".ckpt"), readdir(joinpath(dT, "targets")))
        t = Tcc._checked_load(joinpath(dT, "targets", f)); a, b, L = t.key[2] - '0', t.key[5] - '0', t.key[8] - '0'
        t.status == "no_source" && continue               # no source block needs this target
        startswith(string(t.status), "guarded") && (global nsolved += 1)
        R = sort([x.energy for x in ref if (x.sa, x.sb, x.L) == (a, b, L)])
        lim = min(t.frontier, length(R) == 80 ? R[end] : Inf) - 1e-8     # strictly inside both windows
        A = sort(t.energy[t.energy .<= lim]); B = R[R .<= lim]
        ok1 = length(A) == length(B) && t.required_energy < t.frontier && all(abs(c - round(c)) < 1e-6 for c in t.c2)
        ok1 || println("  target $(t.key) status=$(t.status) nA=$(length(A)) nB=$(length(B)) need=$(t.required_energy) frontier=$(t.frontier) c2=$(t.c2)")
        global okt &= ok1
        length(A) == length(B) && (global dt = max(dt, maximum(abs.(A - B); init=0.0)))
    end
    check(okt && dt < 1e-9, @sprintf("target windows complete below frontier (%d solved, %d reused): dE %.1e",
                                      nsolved, count(==("reused_run_window"), values(stt)), dt))
end
# 6. unified entry (v0.1.7): basis -> spectrum -> targets -> kk -> ope -> summary, resume, identity guard
if isdefined(Tcc, :run_pipeline)
    dP = joinpath(out, "pipeline"); rm(dP; recursive=true, force=true)
    conf = Dict("run" => Dict("N" => N, "g" => g, "out" => dP),
                "spectrum" => Dict("nev" => 12, "exchange" => true, "tol" => 1e-11),
                "threads" => Dict("native" => FuzzifiED.NumThreads, "tile" => 16),
                "targets" => Dict("initial_nev" => 12, "max_nev" => 80),
                "kk" => Dict("cache" => joinpath(out, "kk_cache")))
    manP = try
        run_pipeline(conf)
    catch err
        println("pipeline error: ", sprint(showerror, err)); nothing
    end
    okp = manP !== nothing && all(get(get(manP["stages"], s, Dict()), "status", "") == "done" for s in Tcc.STAGES if s != "tune")
    rows_p = okp ? run_rows(dP) : NamedTuple[]
    okr = okp && rows_equal(spec[true], rows_p)[1] && rows_equal(spec[true], rows_p)[2] < 1e-9
    nkk = okp ? countlines(joinpath(dP, "kk", "projected_states.csv")) - 1 : -1
    check(okp && okr && nkk > 0, "unified pipeline: all stages done, spectrum == reference, KK projected states $(nkk), F = $(okp ? get(manP["stages"]["summary"], "F", NaN) : NaN)")
    if okp
        m2 = run_pipeline(conf)
        check(m2["stages"]["kk"]["finished"] == manP["stages"]["kk"]["finished"], "unified pipeline: rerun skips finished stages")
        # v0.1.10: KK diagnostics and residency-pair KK identical to the all-resident run
        dg = TOML.parsefile(joinpath(dP, "kk", "diagnostics.toml"))
        npass = count(v -> v isa AbstractDict && get(v, "status", "") == "passed_common_original_indices", values(dg))
        dP2 = joinpath(out, "pipeline_pair"); rm(dP2; recursive=true, force=true)
        conf2 = deepcopy(conf); conf2["run"] = merge(Dict{String,Any}(conf2["run"]), Dict{String,Any}("out" => dP2)); conf2["spectrum"] = merge(Dict{String,Any}(conf2["spectrum"]), Dict{String,Any}("residency" => "pair"))
        conf2["run"]["stages"] = ["basis", "spectrum", "targets", "kk"]
        okpair = try
            run_pipeline(conf2)
            csvnum(a, b) = (A = readlines(a); B = readlines(b); length(A) == length(B) && all(zip(A, B)) do (x, y)
                fx = split(x, ','); fy = split(y, ','); length(fx) == length(fy) && all(zip(fx, fy)) do (u, v)
                    u == v && return true
                    pu = tryparse(Float64, strip(u, '"')); pv = tryparse(Float64, strip(v, '"'))
                    pu !== nothing && pv !== nothing && abs(pu - pv) <= 1e-8 * max(1, abs(pu))
                end end)
            csvnum(joinpath(dP, "kk", "projected_states.csv"), joinpath(dP2, "kk", "projected_states.csv"))
        catch err
            println("pair KK error: ", sprint(showerror, err)); false
        end
        opeT = TOML.parsefile(joinpath(dP, "ope", "ope.toml"))
        check(isfinite(get(opeT, "f_phi_phi_J", NaN)) && get(opeT, "half_signed_conjugacy_error", 1.0) < 1e-7 && haskey(opeT, "raw:phi_phi_S"),
              @sprintf("OPE stage: f_phiphiJ = %.6f, f_phiphiT(half) = %.6f, raw phiphiS = %.6f", get(opeT, "f_phi_phi_J", NaN),
                       get(opeT, "f_phi_phi_T_half", NaN), get(opeT, "raw:phi_phi_S", NaN)))
        check(okpair && haskey(dg, "s0_s0_L1_mu"), "KK: residency-pair run gives the same projected states (1e-8); conjugacy checks passed $(npass); mu diagnostic present")
        # v0.1.13: physics report, and a thread-count change does not force any finished stage to rerun
        rp = TOML.parsefile(joinpath(dP, "physics_report.toml"))
        conf3 = deepcopy(conf); conf3["threads"] = Dict{String,Any}("native" => 1, "tile" => 8, "action" => 2)
        m3 = run_pipeline(conf3)
        noredo = all(m3["stages"][s]["finished"] == manP["stages"][s]["finished"] for s in keys(manP["stages"]))
        check(haskey(rp, "alpha") && get(rp, "primary_candidates", 0) > 0 && haskey(rp, "singlet_tower_Delta") && noredo,
              "physics report written ($(get(rp, "primary_candidates", 0)) primary candidates); thread change reruns nothing")
        # v0.1.13: an interrupted target block resumes from its attempt checkpoints
        tdir = joinpath(dP, "targets"); vic = first(filter(f -> startswith(f, "s0_s0_L1") && endswith(f, ".ckpt"), readdir(tdir)))
        tt = Tcc._checked_load(joinpath(tdir, vic))
        if tt.status == "guarded_lowest_window"
            rm(joinpath(tdir, vic); recursive=true)
            ad = joinpath(tdir, "attempts", replace(vic, ".ckpt" => ""))
            nfail = isdir(ad) ? count(f -> startswith(f, "k"), readdir(ad)) : 0
            cfgT = Tcc.pipeline_config(conf); cfgT["segment_checkpoint"] = joinpath(dP, "segment_spaces.ckpt")
            ns0 = count(l -> startswith(l, "target_s0_s0_L1_k"), eachline(joinpath(dP, "stages_targets.tsv")))
            extend_targets(cfgT)
            ns1 = count(l -> startswith(l, "target_s0_s0_L1_k"), eachline(joinpath(dP, "stages_targets.tsv")))
            t2 = Tcc._checked_load(joinpath(tdir, vic))
            check(ns1 - ns0 == 1 && maximum(abs.(t2.energy - tt.energy)) < 1e-10, "target resume: $(nfail) failed attempts reused, only the accepted k re-solved")
        end
        bad = deepcopy(conf); bad["run"] = merge(Dict{String,Any}(bad["run"]), Dict{String,Any}("g" => g + 0.01))
        refused = try; run_pipeline(bad); false; catch; true; end
        check(refused, "unified pipeline: a different g in the same output directory is refused")
    end
end
# 7. coupling tuning (v0.1.9): tuned root, then spectrum at the tuned g reproduces F = 0
if isdefined(Tcc, :run_tune_stage)
    dT = joinpath(out, "pipeline_tuned"); rm(dT; recursive=true, force=true)
    confT = Dict("run" => Dict("N" => N, "g" => "tuned", "out" => dT, "stages" => ["basis", "tune", "spectrum", "summary"]),
                 "tune" => Dict("start" => 0.95, "bounds" => [0.5, 1.5], "maxpoints" => 24),
                 "spectrum" => Dict("nev" => 12, "exchange" => true, "tol" => 1e-11),
                 "threads" => Dict("native" => FuzzifiED.NumThreads, "tile" => 16))
    mT = try; run_pipeline(confT); catch err; println("tune error: ", sprint(showerror, err)); nothing; end
    if mT === nothing
        check(false, "coupling tuning pipeline")
    else
        sel = TOML.parsefile(joinpath(dT, "tune", "selected_g.toml"))
        Fs = mT["stages"]["summary"]["F"]
        check(abs(sel["F"]) <= 1e-5 && sel["bracket_width"] <= 1e-4 && abs(Fs - sel["F"]) < 1e-7,
              @sprintf("coupling tuning: g* = %.10f, F(tune) = %.2e, F(full spectrum at g*) = %.2e, %d points", sel["g"], sel["F"], Fs, sel["physical_points"]))
    end
end
open(io -> serialize(io, (spectrum=spec[false], spectrum_X=spec[true], edges=earrays)), joinpath(out, "golden.jls"), "w")
# 4
if !isempty(golden)
    G = open(deserialize, joinpath(golden, "golden.jls"))
    key(x) = (x.sa, x.sb, x.L, x.x)
    de = 0.0
    for (new, old) in ((spec[false], G.spectrum), (spec[true], G.spectrum_X))
        for kblk in unique(key.(old))
            a1 = sort([x.energy for x in old if key(x) == kblk]); b1 = sort([x.energy for x in new if key(x) == kblk])
            m = min(length(a1), length(b1))   # older versions could keep >k converged Ritz pairs
            global de = max(de, maximum(abs.(a1[1:m] - b1[1:m]); init=0.0))
        end
    end
    check(de < 1e-10, @sprintf("spectrum vs golden: %.2e", de))
    dc = 0.0
    for (new, old) in ((spec[false], G.spectrum), (spec[true], G.spectrum_X))
        for kblk in unique(key.(old))
            a1 = [x.c2 for x in sort([x for x in old if key(x) == kblk]; by=x -> x.energy)]
            b1 = [x.c2 for x in sort([x for x in new if key(x) == kblk]; by=x -> x.energy)]
            m = min(length(a1), length(b1))
            global dc = max(dc, maximum(abs.(a1[1:m] - b1[1:m]); init=0.0))
        end
    end
    check(dc < 1e-8, @sprintf("C2 labels vs golden: %.2e", dc))
    gm = 0.0
    for (nm, A) in earrays
        B = G.edges[nm]
        # eigenvector signs are gauge: compare |overlaps| and Gram
        cm(x, y) = (r = 1:min(size(x, 1), size(y, 1)); c = 1:min(size(x, 2), size(y, 2)); maximum(abs.(abs.(x[r, c]) - abs.(y[r, c]))))
        nsj = min(size(A.image_gram, 3), size(B.image_gram, 3))
        global gm = max(gm, maximum(cm(x, y) for (x, y) in zip(A.overlaps, B.overlaps)),
                        maximum(abs.(abs.(A.image_gram[:, :, 1:nsj]) - abs.(B.image_gram[:, :, 1:nsj]))))
    end
    check(gm < 1e-9, @sprintf("edges vs golden (|M|, Gram): %.2e", gm))
end
open(io -> TOML.print(io, Dict("N"=>N, "version"=>string(TCC_VERSION), "fails"=>fails)), joinpath(out, "regress.toml"), "w")
println(isempty(fails) ? "REGRESS_PASS N=$N v$(TCC_VERSION)" : "REGRESS_FAIL N=$N: $(join(fails, "; "))")
exit(isempty(fails) ? 0 : 1)
