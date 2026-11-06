# v0.1.16 built-in acceptance gates (review findings V15-1 ... V15-9).
#   1. window_gate rejects non-orthonormal / unconverged / C2-mixed / unsorted / non-finite windows
#   2. column_shas = SHA-256 of the column bytes; any bit change changes it; edge matching uses it
#   3. _replace! / _recover_replace!: commit with marker, roll back without, finish a partial commit
#   4. polish_c2! never rotates frozen columns (and repairs the same mixing when not frozen)
#   5. pipeline: eig/target checkpoints carry column SHAs, gate records and producer; stages record
#      producer and output inventory; a changed inventory re-runs the stage; stale attempt caches
#      (no identity) are moved aside; a cached "passed" attempt below the current requirement is not
#      accepted; a stored target with a bad residual is rejected and recomputed.
# usage: julia --threads=T t12_gates.jl N out
include(joinpath(@__DIR__, "..", "src", "Tcc.jl"))
using .Tcc, FuzzifiED, FuzzifiED.SO3lver, TOML, Printf, LinearAlgebra, Random, SHA
FuzzifiED.SilentStd = true
N = parse(Int, ARGS[1]); root = ARGS[2]; g = 0.95
mkpath(root)
ok = true
check(c, msg) = (println(c ? "PASS " : "FAIL ", msg); global ok &= c)
gatefails(f) = try; f(); false; catch err; err isa Tcc.GateError || rethrow(); true; end
function captured(f)                       # stdout of f() as a String (also echoed)
    path, io = mktemp()
    redirect_stdout(f, io); close(io)
    s = read(path, String); print(s); s
end

# ---- 1. window_gate --------------------------------------------------------------------------------
let rng = MersenneTwister(3), n = 500, m = 6
    Q = Matrix(qr(randn(rng, n, m)).Q)[:, 1:m]; E = collect(1.0:m); rh = fill(1e-9, m); rc = fill(1e-9, m); c2 = [0.0, 2, 5, 5, 9, 14]
    good = !gatefails(() -> window_gate("ok", E, Q; rh, rc, c2))
    Q2 = copy(Q); Q2[:, 2] .+= 1e-4 .* Q[:, 1]
    bad = [gatefails(() -> window_gate("orth", E, Q2; rh, rc, c2)),
           gatefails(() -> window_gate("rh", E, Q; rh=[rh[1:end-1]; 2e-6], rc, c2)),
           gatefails(() -> window_gate("rc", E, Q; rh, rc=[2e-6; rc[2:end]], c2)),
           gatefails(() -> window_gate("c2", E, Q; rh, rc, c2=[c2[1:end-1]; 14.0001])),
           gatefails(() -> window_gate("sort", reverse(E), Q; rh, rc, c2)),
           gatefails(() -> window_gate("nan", [NaN; E[2:end]], Q; rh, rc, c2)),
           gatefails(() -> window_gate("norm", E, 1.00001 .* Q))]
    check(good && all(bad), "window_gate: accepts a clean window, rejects orthogonality/H residual/C2 residual/C2 label/order/NaN/norm ($(count(bad))/7)")
end

# ---- 2. column SHAs ---------------------------------------------------------------------------------
let rng = MersenneTwister(5), V = randn(MersenneTwister(5), 1000, 4)
    s = column_shas(V)
    ref = [bytes2hex(sha256(reinterpret(UInt8, V[:, j]))) for j in 1:4]
    W = copy(V); W[17, 3] = nextfloat(W[17, 3])
    t = column_shas(W)
    m1 = Tcc._match_columns([1.0, 2.0], s[1:2], [1.0, 2.0, 3.0], [s[1], t[3], s[4]])
    check(s == ref && t[3] != s[3] && t[[1, 2, 4]] == s[[1, 2, 4]] && column_shas(view(V, :, 2:3)) == s[2:3] &&
          m1[1] == 1 && m1[2] === nothing,
          "column_shas: byte-exact, one-ulp change detected, views ok; edge matching needs identical SHA")
end

# ---- 3. atomic replacement ------------------------------------------------------------------------
let d = joinpath(root, "replace_test")
    rm(d; recursive=true, force=true); mkpath(d)
    a = joinpath(d, "a.ckpt"); b = joinpath(d, "b.ckpt")
    Tcc._atomic_save(a, (v=1,)); Tcc._atomic_save(b, (v=1,))
    Tcc._replace!([a => (v=2,), b => (v=2,)]; tag="k1")
    ok1 = Tcc._checked_load(a).v == 2 && Tcc._checked_load(b).v == 2 && Tcc._checked_load(joinpath(d, "a.k1.ckpt")).v == 1 &&
          !any(startswith(".replace-"), readdir(d))
    # crash before the marker: .next written, nothing committed -> rolled back
    Tcc._atomic_save(a * ".next", (v=3,))
    Tcc._recover_replace!(d)
    ok2 = Tcc._checked_load(a).v == 2 && !isdir(a * ".next") && isdir(a * ".next.abandoned")
    # crash in the middle of a commit: a committed, b not -> completed
    Tcc._atomic_save(a * ".next", (v=4,)); Tcc._atomic_save(b * ".next", (v=4,))
    write(joinpath(d, ".replace-k2"), "k2\na.ckpt\nb.ckpt\n")
    mv(a, Tcc._superseded_name(a, "k2")); mv(a * ".next", a)
    Tcc._recover_replace!(d)
    ok3 = Tcc._checked_load(a).v == 4 && Tcc._checked_load(b).v == 4 && Tcc._checked_load(joinpath(d, "b.k2.ckpt")).v == 2 &&
          !isfile(joinpath(d, ".replace-k2"))
    check(ok1 && ok2 && ok3, "replacement: commit keeps superseded copy ($ok1), no-marker crash rolled back ($ok2), partial commit completed ($ok3)")
end

# ---- 5 (run first; 4 uses its operators). pipeline ------------------------------------------------
th = Dict("native" => min(4, Threads.nthreads()), "tile" => 16)
dP = joinpath(root, "gates_N$N"); rm(dP; recursive=true, force=true)
conf = Dict("run" => Dict("N" => N, "g" => g, "out" => dP),
            "spectrum" => Dict{String,Any}("nev" => 8, "exchange" => true, "tol" => 1e-11),
            "threads" => th, "targets" => Dict("initial_nev" => 12, "max_nev" => 120),
            "kk" => Dict("cache" => joinpath(root, "kk_cache")), "ope" => Dict("half" => false))
man = run_pipeline(conf)
let okc = true
    for d in readdir(joinpath(dP, "blocks"))
        p = joinpath(dP, "blocks", d, "eig.ckpt"); isdir(p) || continue
        e = Tcc._checked_load(p); r = Tcc._checked_load(joinpath(dP, "blocks", d, "result.ckpt"))
        okc &= hasproperty(e, :colsha) && e.colsha == column_shas(e.vectors) && hasproperty(e, :gate) && e.gate.max_H_residual <= 1e-6 &&
               hasproperty(e, :producer) && e.producer.tcc == string(TCC_VERSION) && r.gate == e.gate
    end
    for f in filter(f -> occursin(r"^s\d_s\d_L\d\.ckpt$", f), readdir(joinpath(dP, "targets")))
        t = Tcc._checked_load(joinpath(dP, "targets", f))
        isempty(t.energy) && continue
        okc &= hasproperty(t, :identity) && t.colsha == column_shas(t.vectors) && hasproperty(t, :gate) && issorted(t.energy) &&
               (t.status != "guarded_lowest_window" || hasproperty(t, :independent_max_energy_difference) || hasproperty(t, :independent_min_margin))
    end
    okm = all(haskey(man["stages"][s], "producer_version") && haskey(man["stages"][s], "outputs_digest") for s in keys(man["stages"])) &&
          length(get(man, "invocations", [])) == 1
    check(okc && okm, "checkpoints carry column SHAs, gate records, identity and producer; stages record producer and output inventory")
end
m2 = run_pipeline(conf)
check(all(m2["stages"][s]["finished"] == man["stages"][s]["finished"] for s in keys(man["stages"])) && length(m2["invocations"]) == 2,
      "unchanged rerun skips every stage (inventories match), invocation history kept")

tdir = joinpath(dP, "targets")
cfgT = Tcc.pipeline_config(conf); cfgT["segment_checkpoint"] = joinpath(dP, "segment_spaces.ckpt")
tkeys = filter(f -> occursin(r"^s\d_s\d_L\d\.ckpt$", f), readdir(tdir))
guarded = [f for f in tkeys if Tcc._checked_load(joinpath(tdir, f)).status == "guarded_lowest_window"]
check(!isempty(guarded), "test setup: $(length(guarded)) guarded target windows")

# 5a. inventory: a target moved aside -> the targets stage is re-run (only that block recomputed)
let f = first(guarded), t0 = Tcc._checked_load(joinpath(tdir, f))
    mv(joinpath(tdir, f), Tcc._superseded_name(joinpath(tdir, f), "rejected"))
    m3 = run_pipeline(conf; stages=["targets"])
    t1 = Tcc._checked_load(joinpath(tdir, f))
    check(m3["stages"]["targets"]["finished"] != man["stages"]["targets"]["finished"] && maximum(abs.(t1.energy - t0.energy)) < 1e-9,
          "changed output inventory re-runs the targets stage; recomputed $f equals the original")
end

# 5b. stored target with a bad residual: rejected and recomputed (V15-4)
let f = guarded[end], p = joinpath(tdir, f), t0 = Tcc._checked_load(p)
    bad = merge(t0, (rh=[1e-3; t0.rh[2:end]],))
    mv(p, Tcc._superseded_name(p, "orig")); Tcc._atomic_save(p, bad)
    log = captured(() -> extend_targets(cfgT)); t1 = Tcc._checked_load(p)
    check(occursin("TARGET_REJECT", log) && maximum(t1.rh) <= 1e-6 && maximum(abs.(t1.energy - t0.energy)) < 1e-9,
          "stored target with H residual 1e-3 is rejected and recomputed")
end

# 5c. legacy attempt cache without identity is moved aside (V15-2)
let f = first(guarded), key = replace(f, ".ckpt" => ""), p = joinpath(tdir, f)
    ad = joinpath(tdir, "attempts", key); rm(ad; recursive=true, force=true); mkpath(ad)
    t0 = Tcc._checked_load(p)
    Tcc._atomic_save(joinpath(ad, "k12.ckpt"), (k=12, energy=t0.energy, parity=t0.parity, frontier=Inf, passed=true, vectors=t0.vectors, frozen=falses(length(t0.energy))))
    mv(p, Tcc._superseded_name(p, "orig"))
    log = captured(() -> extend_targets(cfgT)); t1 = Tcc._checked_load(p)
    check(occursin("TARGET_ATTEMPTS_STALE", log) && isdir(ad * ".stale") && maximum(abs.(t1.energy - t0.energy)) < 1e-9,
          "attempt cache without identity moved aside; target recomputed")
end

# 5d. cached attempt marked passed but below the current requirement is not accepted (V15-3)
let lmax = cfgT["lmax"], cand = [f for f in guarded if endswith(f, "_L$(lmax + 1).ckpt")]
    if isempty(cand)
        check(true, "stale passed attempt: no guarded L=lmax+1 target in this run (skipped)")
    else
        f = first(cand); key = replace(f, ".ckpt" => ""); p = joinpath(tdir, f); t0 = Tcc._checked_load(p)
        aid = merge(t0.identity, (tol=get(cfgT, "tol", 1e-10), guard=get(cfgT, "guard", 1e-5)))
        ad = joinpath(tdir, "attempts", key); rm(ad; recursive=true, force=true); mkpath(ad)
        write(joinpath(ad, "identity.txt"), Tcc._identity_digest(aid) * "\n")
        k0 = first(target_requests(cfgT["initial_nev"], cfgT["max_nev"], maximum(values(t0.sector_dims))))
        keep = 1:2                                   # far too few states for the requirement
        Tcc._atomic_save(joinpath(ad, "k$k0.ckpt"), (k=k0, energy=t0.energy[keep], parity=t0.parity[keep], frontier=Inf, passed=true,
                         vectors=t0.vectors[:, keep], frozen=falses(2), identity=aid))
        mv(p, Tcc._superseded_name(p, "orig"))
        extend_targets(cfgT)
        t1 = Tcc._checked_load(p)
        lim = min(t0.frontier, t1.frontier) - 1e-8
        A = t0.energy[t0.energy .<= lim]; B = t1.energy[t1.energy .<= lim]
        check(t1.frontier > t1.required_energy + 1e-5 && length(A) == length(B) && maximum(abs.(A - B); init=0.0) < 1e-9,
              "cached 'passed' attempt with 2 states not accepted for $key; final window complete and == original")
    end
end

# ---- 4. polish_c2! with a frozen column ------------------------------------------------------------
let segs = Tcc._checked_load(joinpath(dP, "segment_spaces.ckpt")).spaces, L = 2
    tpl = hamiltonian_template(N); lib = SegLibrary(builder=:tiled, tile_width=16, workers=Threads.nthreads(), opmat_threads=th["native"])
    ss = [segs[1], segs[1]]; sH, fH = segops_shared!(lib, ss, tpl.template); sC, fC = segops_shared!(lib, ss, tpl.c2)
    cp = BuildCompSpace(ss, [2N, 0, 0], 2L)
    H = absorb!(set_g!(BuildCompOperator(cp, tpl.template, sH; num_th=th["native"]), tpl, g), fH)
    C = absorb!(BuildCompOperator(cp, tpl.c2, sC; num_th=th["native"]), fC)
    e = Tcc._checked_load(joinpath(dP, "blocks", "s0_s0_L$L", "eig.ckpt")); r = Tcc._checked_load(joinpath(dP, "blocks", "s0_s0_L$L", "result.ckpt"))
    c2 = [x.c2 for x in sort(r.rows; by=x -> x.energy)]
    pairs = [(i, j) for i in eachindex(e.energy) for j in i+1:length(e.energy) if e.parity[i] == e.parity[j] && abs(c2[i] - c2[j]) > 0.5]
    if isempty(pairs)
        check(true, "polish frozen: no same-parity pair with different C2 in (0,0) L$L (skipped)")
    else
        i, j = first(pairs); θ = 1e-3
        V = copy(e.vectors); vi = V[:, i]; vj = V[:, j]; V[:, i] = cos(θ) .* vi .+ sin(θ) .* vj; V[:, j] = -sin(θ) .* vi .+ cos(θ) .* vj
        E = copy(e.energy); E[j] = E[i] + 1e-6                        # pretend an accidental near-degeneracy
        sh = column_shas(V)
        fz = falses(length(E)); fz[i] = true
        V1 = copy(V); E1 = copy(E); polish_c2!(E1, V1, e.parity, H, C; frozen=fz)
        _, rc1 = post_c(C, V1)
        V2 = copy(V); E2 = copy(E); n2 = polish_c2!(E2, V2, e.parity, H, C)
        _, rc2 = post_c(C, V2)
        check(column_shas(V1)[i] == sh[i] && rc1[j] > 1e-6 && n2 >= 1 && max(rc2[i], rc2[j]) < 1e-8,
              @sprintf("polish_c2!: frozen column bit-identical (partner left at rC2 %.1e for the gate); unfrozen pair repaired (rC2 %.1e)", rc1[j], max(rc2[i], rc2[j])))
    end
end

# ---- 6. v0.1.17 complement test: complete window passes, a window with an interior state removed fails ---
let segs = Tcc._checked_load(joinpath(dP, "segment_spaces.ckpt")).spaces, L = 2
    tpl = hamiltonian_template(N); lib = SegLibrary(builder=:tiled, tile_width=16, workers=Threads.nthreads(), opmat_threads=th["native"])
    ss = [segs[1], segs[1]]; sH, fH = segops_shared!(lib, ss, tpl.template)
    cp = BuildCompSpace(ss, [2N, 0, 0], 2L)
    H = absorb!(set_g!(BuildCompOperator(cp, tpl.template, sH; num_th=th["native"]), tpl, g), fH)
    p = exchange_map(cp); sd = Dict(s => length(ExchangeSector(p[1], p[2], s)) for s in (1, -1))
    e = Tcc._checked_load(joinpath(dP, "blocks", "s0_s0_L$L", "eig.ckpt"))
    w = (energy=e.energy, parity=e.parity, vectors=e.vectors)
    r1 = Tcc._complement_check(H, cp, 0, 0, w, sd, true, Threads.nthreads(), 1e-11, 4242)
    s = e.parity[2]; inner = [i for i in eachindex(e.energy) if e.parity[i] == s]
    drop = inner[min(2, length(inner) - 1)]; keep = setdiff(eachindex(e.energy), [drop])
    w2 = (energy=e.energy[keep], parity=e.parity[keep], vectors=e.vectors[:, keep])
    r2 = Tcc._complement_check(H, cp, 0, 0, w2, sd, true, Threads.nthreads(), 1e-11, 4242)
    check(r1.min_margin > -1e-7 && r2.min_margin < -1e-3 && abs(r2.worst_lambda - e.energy[drop]) < 1e-8,
          @sprintf("complement test: complete window margin %.2e; window missing E=%.6f flagged (margin %.2e, found %.6f)",
                   r1.min_margin, e.energy[drop], r2.min_margin, r2.worst_lambda))
end

println(ok ? "GATES_PASS N=$N" : "GATES_FAIL N=$N")
