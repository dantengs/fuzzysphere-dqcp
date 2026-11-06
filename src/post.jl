# Drivers that were scripts under bench/ up to v0.2.8, moved into the library (v0.3.0) so that bin/tcc.jl
# runs every step from one TOML file.  Logic, thresholds and output formats are unchanged:
#   run_coverage      bench/coverage_run.jl   (spectrum windows grown until Delta >= delta_cut)
#   run_targets_pairs bench/targets_run.jl    (KK target windows of selected pairs, outside the manifest)
#   run_ope3_step     bench/ope3_run.jl       (three-point insertions, src/ope3.jl)
#   run_parity        bench/parity_run.jl     (physical parity of every stored state and KK mode)

"""
    run_coverage(conf; delta_cut, kmax=3000) -> manifest

basis -> tune -> spectrum with the windows of `conf`; every block/exchange sector whose window ends below
`delta_cut` (and is not the whole sector) is enlarged (x1.6, at least +8, at most kmax) with spectrum.extend
= true until all reach the cut or are complete; then targets -> kk -> ope -> summary.  Writes
out/coverage.toml and out/coverage_run.toml (the final configuration).  Resumes from out/coverage.toml.
"""
function run_coverage(conf; delta_cut, kmax=3000, stages=nothing)
    conf = deepcopy(conf); cut = Float64(delta_cut)
    sp = get!(conf, "spectrum", Dict{String,Any}())
    c0 = pipeline_config(conf); out = c0["out"]; useX = get(sp, "exchange", true)
    todo = stages === nothing ? c0["stages"] : stages
    pn = Dict{String,Any}(get(sp, "parity_nev", Dict{String,Any}())); bn = Dict{String,Any}(get(sp, "block_nev", Dict{String,Any}()))
    history = Any[]; resumed = false
    if isfile(joinpath(out, "coverage.toml"))
        for h in get(TOML.parsefile(joinpath(out, "coverage.toml")), "history", Any[])
            for (k, v) in h["grow"]; v isa AbstractVector ? (pn[k] = Int.(v)) : (bn[k] = Int(v)); end
            push!(history, h)
        end
        resumed = true; println("COVERAGE_RESUME ", length(history), " recorded extensions")
    end
    for it in 0:50
        conf["spectrum"]["parity_nev"] = pn; conf["spectrum"]["block_nev"] = bn
        (it > 0 || resumed) && (conf["spectrum"]["extend"] = true)
        run_pipeline(conf; stages=[s for s in ("basis", "tune", "spectrum") if s in todo])
        rows = run_rows(out); F = tuning_F(rows); F === nothing && error("no ground/current/stress rows for the energy scale")
        a = 2 / (F.EJ - F.E0)
        grow = Dict{String,Any}(); rep = Dict{String,Any}()
        for (sa, sb, L) in unique([(r.sa, r.sb, r.L) for r in rows])
            key = "s$(sa)_s$(sb)_L$(L)"; diag = useX && sa == sb
            cnt = Dict{Int,Int}(); want = Dict{Int,Int}()
            for x in (diag ? (1, -1) : (0,))
                rs = sort([r for r in rows if (r.sa, r.sb, r.L, r.x) == (sa, sb, L, x)]; by=r -> r.energy)
                n = length(rs); dim = n == 0 ? 0 : rs[1].sector_dim
                top = n == 0 ? NaN : a * (rs[end].energy - F.E0)
                complete = n > 0 && n >= dim
                rep["$(key)_x$x"] = Dict("states" => n, "sector_dim" => dim, "delta_top" => top, "complete" => complete,
                                        "reached" => complete || top >= cut)
                cnt[x] = n; want[x] = n
                if n > 0 && !complete && top < cut
                    k = min(dim, kmax, max(n + 8, ceil(Int, 1.6n)))
                    k > n && (want[x] = k)
                end
            end
            any(want[x] > cnt[x] for x in keys(cnt)) || continue
            grow[key] = diag ? [want[1], want[-1]] : want[0]
        end
        println("COVERAGE it=$it alpha=$a blocks=$(length(rep)) grow=$(length(grow)) min_top=",
                minimum((v["complete"] ? Inf : v["delta_top"]) for v in values(rep)))
        open(joinpath(out, "coverage.toml"), "w") do io
            TOML.print(io, Dict("delta_cut" => cut, "kmax" => kmax, "alpha" => a, "E0" => F.E0, "iteration" => it,
                                "sectors" => rep, "history" => history))
        end
        isempty(grow) && break
        push!(history, Dict("iteration" => length(history), "grow" => grow))
        for (k, v) in grow; v isa AbstractVector ? (pn[k] = v) : (bn[k] = v); end
        flush(stdout)
    end
    open(io -> TOML.print(io, conf), joinpath(out, "coverage_run.toml"), "w")
    m = run_pipeline(conf; stages=[s for s in ("targets", "kk", "ope", "summary") if s in todo])
    r = TOML.parsefile(joinpath(out, "coverage.toml"))
    bad = [k for (k, v) in r["sectors"] if !v["reached"]]
    println(isempty(bad) ? "COVERAGE_DONE all sectors reach Delta >= $cut or are complete" : "COVERAGE_LIMITED (kmax) $(join(sort(bad), " "))")
    m
end

"""KK target windows of `pairs` only (several nodes may fill one run's targets/ in parallel; the manifest is not touched)."""
function run_targets_pairs(c, pairs; source_delta_max=get(c, "source_delta_max", nothing))
    cfg = copy(c); cfg["g"] = resolved_g(c)
    isempty(cfg["segment_checkpoint"]) && (cfg["segment_checkpoint"] = joinpath(c["out"], "segment_spaces.ckpt"))
    cfg["pairs"] = [Tuple(Int.(p)) for p in pairs]
    source_delta_max === nothing || (cfg["source_delta_max"] = Float64(source_delta_max))
    println("TARGETS_RUN pairs=$(cfg["pairs"]) source_delta_max=$(get(cfg, "source_delta_max", "none")) threads=$(Threads.nthreads())"); flush(stdout)
    t0 = time(); st = extend_targets(cfg)
    for k in sort(collect(keys(st))); println("TARGET_STATUS $k $(st[k])"); end
    println("TARGETS_RUN_DONE seconds=$(round(time() - t0; digits=1))")
    st
end

"""Three-point insertions on a finished run; `o` is the [ope3] table (out plus the run_ope3 options)."""
function run_ope3_step(c, o; run_toml="")
    od = abspath(o["out"]); opts = Dict{String,Any}(k => v for (k, v) in o if k != "out")
    abspath(od) == abspath(c["out"]) && error("ope3.out must differ from the run directory")
    mkpath(od)
    open(io -> TOML.print(io, Dict("run_toml" => run_toml, "run" => c["out"], "N" => c["N"], "g" => c["g"], "opts" => opts)),
         joinpath(od, "ope3_input.toml"), "w")
    t0 = time()
    spaces, sha = get(opts, "half_only", false) ? (nothing, "") : ope3_spaces(c)   # v0.2.6: half-only needs no integer basis
    println("OPE3_START N=$(c["N"]) g=$(c["g"]) run=$(c["out"]) basis=$(isempty(sha) ? "-" : sha[1:16]) threads=$(Threads.nthreads())"); flush(stdout)
    r = run_ope3(c, spaces, sha, od; opts=opts)
    println("OPE3_FINISHED seconds=$(round(time() - t0; digits=1)) maxrss_gib=$(round(Sys.maxrss() / 2^30; digits=2))")
    r
end

"""
    run_parity(c, od; targets=false, vacuum_raw=nothing)

Physical parity P_phys = P_seg * P_seg(vacuum) * (-1)^L of every stored eigenstate (blocks, and target
windows if `targets`), and <P> of every KK-projected mode.  The vacuum value comes from the run's (0,0) L0
block, or from `vacuum_raw` (±1) for runs without that block.  Writes od/parity.csv, parity_kk.csv,
parity_summary.toml.
"""
function run_parity(c, od; targets=false, vacuum_raw=nothing)
    od = abspath(od); mkpath(od); out = c["out"]; N = c["N"]
    spaces, bsha = ope3_spaces(c)
    getsp(s) = spaces isa SpaceStore ? getspace!(spaces, s) : spaces[s+1]
    F = tuning_F(run_rows(out)); alpha = F === nothing ? NaN : 2 / (F.EJ - F.E0); E0 = F === nothing ? NaN : F.E0
    keys_ = sort([d for d in readdir(joinpath(out, "blocks")) if occursin(r"^s\d_s\d_L\d+$", d)])
    pairs = sort(unique([(parse(Int, m[1]), parse(Int, m[2])) for m in (match(r"^s(\d)_s(\d)_L", k) for k in keys_)]))
    maps = Dict{Int,Any}(); summ = Dict{String,Any}("N" => N, "g" => c["g"], "run" => out, "tcc_version" => string(TCC_VERSION))
    for s in sort(unique(vcat([[a, b] for (a, b) in pairs]...)))
        t0 = time(); m, w = segment_ph_maps(getsp(s), N); maps[s] = m; summ["segment_s$(s)_orthogonality"] = w
        println("PARITY_SEGMENT s=$s sectors=$(length(m)) orth_err=$w seconds=$(round(time() - t0; digits=1))"); flush(stdout)
    end
    raw = Any[]; maxres = 0.0
    for key in keys_
        m = match(r"^s(\d)_s(\d)_L(\d+)$", key); a, b, l = parse.(Int, m.captures)
        wins = Any[("block", source_window(out, a, b, l))]
        targets && push!(wins, ("target", target_window(out, a, b, l)))
        for (kind, W) in wins
            W === nothing && continue
            W.basis_sha == bsha || error("$key: window in another basis")
            cs = BuildCompSpace([getsp(a), getsp(b)], [2N, 0, 0], 2l; disp_std=false)
            cs.dim == W.dim || error("$key: dimension mismatch")
            p, res = parity_values(comp_parity(cs, maps[a], maps[b]), W.V)
            maxres = max(maxres, maximum(res; init=0.0))
            maximum(res; init=0.0) < 1e-6 && all(x -> abs(abs(x) - 1) < 1e-6, p) || error("$key $kind: not a pure parity eigenbasis (max residual $(maximum(res)))")
            for i in eachindex(p); push!(raw, (a, b, l, kind, i, W.parity[i], W.E[i], W.C2[i], p[i], res[i])); end
            println("PARITY_BLOCK $key $kind states=$(length(p)) max_residual=$(maximum(res; init=0.0))"); flush(stdout)
        end
    end
    v0 = [r for r in raw if r[1:4] == (0, 0, 0, "block") && abs(r[8]) < 1e-4]
    vac = vacuum_raw !== nothing ? Int(vacuum_raw) :
          isempty(v0) ? error("no (0,0) L0 vacuum in this run: set parity.vacuum_raw = ±1") : round(Int, v0[argmin([r[7] for r in v0])][9])
    summ["vacuum_raw_sign"] = vac; summ["max_residual"] = maxres
    open(joinpath(od, "parity.csv"), "w") do io
        println(io, "sA,sB,L,window,index,X,energy,Delta,C2,P_raw,P,residual")
        for r in raw
            println(io, join((r[1], r[2], r[3], r[4], r[5], r[6], r[7], alpha * (r[7] - E0), r[8], r[9], physical_parity(r[9], vac, r[3]), r[10]), ","))
        end
    end
    # v0.2.4: a mode q (columns of ch.Q over ch.source_ids) has <P> = sum_i q_i^2 P_i because the raw states are
    # P eigenstates; |<P>| = 1 means a parity-pure mode, anything else a mode mixing opposite parities.
    praw = Dict{Tuple{Int,Int,Int},Vector{Int}}()
    for r in raw
        r[4] == "block" || continue
        push!(get!(praw, (r[1], r[2], r[3]), Int[]), physical_parity(r[9], vac, r[3]))
    end
    kkrows = Any[]; nmixed = 0
    pdir = joinpath(out, "kk", "projections")
    for f in (isdir(pdir) ? sort(readdir(pdir)) : String[])
        m = match(r"^s(\d)_s(\d)_L(\d+)\.ckpt$", f); m === nothing && continue
        a, b, l = parse.(Int, m.captures); haskey(praw, (a, b, l)) || continue
        x = _checked_load(joinpath(pdir, f)); W = source_window(out, a, b, l; vectors=false)
        (W !== nothing && length(praw[(a, b, l)]) == length(W.E)) || (println("PARITY_KK_SKIP $f (no window)"); continue)
        if x.source_sha == W.sha
            P = praw[(a, b, l)]
        else
            # v0.2.5: the window was extended after the projection (spectrum.extend keeps every old vector and
            # energy bit for bit), so map the projection's source states to the current window by energy.
            idx = [findfirst(e -> abs(e - y) <= 1e-10 * max(1, abs(y)), W.E) for y in x.source_E]
            any(isnothing, idx) && (println("PARITY_KK_SKIP $f (projection states not found in the extended window)"); continue)
            all(abs(W.C2[idx[k]] - x.source_C2[k]) < 1e-4 for k in eachindex(idx)) || (println("PARITY_KK_SKIP $f (C2 mismatch)"); continue)
            P = praw[(a, b, l)][idx]
            println("PARITY_KK_MAPPED $f by energy ($(length(idx)) of $(length(W.E)) window states)")
        end
        for (t, chans) in x.threshold_channels, ch in chans
            ch.Q === nothing && continue
            k2 = diag(ch.Q' * ch.K2 * ch.Q)
            for j in axes(ch.Q, 2)
                q = ch.Q[:, j]; pm = sum(q .^ 2 .* P[ch.source_ids])
                pure = abs(abs(pm) - 1) < 1e-6
                t == 0.25 && !pure && (nmixed += 1)
                push!(kkrows, (a, b, l, t, ch.C2, j, ch.energies[j], alpha * (ch.energies[j] - E0), ch.energy_mixing_residual[j], k2[j],
                               pm, pure ? string(round(Int, pm)) : "mixed"))
            end
        end
    end
    open(joinpath(od, "parity_kk.csv"), "w") do io
        println(io, "sA,sB,L,threshold,C2,mode,energy,Delta,sigmaE,K2,P_expectation,P")
        for r in kkrows; println(io, join(r, ",")); end
    end
    summ["kk_modes"] = length(kkrows); summ["kk_modes_mixed_t0.25"] = nmixed
    println("PARITY_KK modes=$(length(kkrows)) mixed_at_0.25=$nmixed")
    summ["convention"] = "P = P_seg * P_seg(vacuum) * (-1)^L (src/parity.jl); Table VI of arXiv:2306.16435 reproduced 129/129 at N10"
    open(io -> TOML.print(io, summ), joinpath(od, "parity_summary.toml"), "w")
    println("PARITY_DONE rows=$(length(raw)) vacuum_raw=$vac max_residual=$maxres")
    od
end
