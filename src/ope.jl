# OPE stage (v0.1.11): port of Zhou v0.0.5 tuned_postprocess.jl (integer anchors) and
# half_tuned.jl (half-integer (1/2,1/2) weight-(1,1) sector) onto Tcc runs.
#
# Integer sector, (sA,sB) = (0,0):
#   normalization  n = |<phi| Gamma^2_0 |0>|          phi = lowest C2=2 of L0, 0 = lowest C2=0 of L0
#   f_phiphiS      = 1         * |<phi| Gamma^2_0 |S>|      / n     (S = second C2=0 state of L0)
#   f_phiphiT14    = sqrt(5)/2 * |<phi| Gamma^2_0 |T14>|    / n     (T14 = lowest C2=5 of L0)
#   f_phiphistress = sqrt(15/8)* |<stress| Gamma^2_2 |phi>| / n     (stress = lowest C2=0 of L2)
# in the raw window basis and in the KK-projected bases (scalar L0 channels, optionally L2).
# Half sector (1/2,1/2), total flavour weight (1,1), L = 0, 1:
#   f_phiphiJ = sqrt(3) |<J| Gamma^2_1 |phi>| / n ,   f_phiphiT = sqrt(2) |<phi| Gamma^2_0 |T>| / n
# with signed reduced conjugacy gates.  Finite-N estimates; not certified CFT data.

"""Half-integer segment space (one SegSpace, segment spin 1/2) — frozen half_model.jl."""
function build_half_segment(nm::Int)
    sec_pt = stack([[ne, ((nm + 1) * ne) % 2, 1] for ne in 1:2:(2nm-1)])
    qnd_pt = [GetNeQNDiag(2nm), GetLz2QNDiag(nm, 2), GetFlavQNDiag(nm, 2, [1, -1])]
    BuildSegSpaces(2nm, sec_pt, qnd_pt, GetLzLpTerms(nm, 2), GetC2Terms(nm, 2, :SU), [Float64[0.75]])[1]
end
const HALF_DIMS = Dict(4 => (10, 22), 6 => (180, 508), 10 => (394570, 1172134), 11 => (3274744, 9750284), 12 => (28586486, 85263014))

_pick(E, C2, c) = (ids = [i for i in eachindex(E) if abs(C2[i] - c) < 1e-4]; sort(ids; by=i -> E[i]))

"""Integer-sector anchors for one choice of source (L0) and target (L2) bases."""
function _ope_anchors(label, s0, s2, M, A)
    pick(states, c) = sort(filter(x -> x.c2 == c, states); by=x -> x.energy)
    sing = pick(s0, 0); phi = pick(s0, 2); ten = pick(s0, 5); stress = pick(s2, 0)
    (isempty(sing) || isempty(phi)) && return (status="missing_vacuum_or_phi", rows=Any[])
    v = sing[1]; p = phi[1]
    nrm = abs(dot(p.coeff, M * v.coeff)); nrm > 1e-12 || error("Vanishing phi normalization")
    items = Any[]
    length(sing) >= 2 && push!(items, ("phi_phi_S", sing[2], p, M, 1.0))
    isempty(ten) || push!(items, ("phi_phi_T14", ten[1], p, M, sqrt(5) / 2))
    isempty(stress) || push!(items, ("phi_phi_stress", p, stress[1], A, sqrt(15 / 8)))
    rows = Any[]
    for (ch, src, tgt, op, fac) in items
        amp = dot(tgt.coeff, op * src.coeff)
        push!(rows, (label, ch, src.label, tgt.label, src.energy, tgt.energy, src.c2, tgt.c2, amp, nrm, fac,
                     fac * abs(amp) / nrm, src.residual, tgt.residual))
    end
    (status="finite_N_candidate_not_certified_CFT_OPE", rows=rows, normalization=nrm)
end

_raw_states(E, C2) = [(c2=round(Int, C2[i]), energy=E[i], residual=0.0, coeff=Float64[j == i for j in eachindex(E)], label="raw_$(i)")
                      for i in eachindex(E)]
function _projected_states(n, chans)
    st = Any[]
    for ch in chans
        ch.Q === nothing && continue
        for j in axes(ch.Q, 2)
            v = zeros(n); v[ch.source_ids] = ch.Q[:, j]
            push!(st, (c2=ch.C2, energy=ch.energies[j], residual=ch.energy_mixing_residual[j], coeff=v, label="projected_C2_$(ch.C2)_$(j)"))
        end
    end
    sort(st; by=x -> x.energy)
end

"""Solve the half-sector L block (lowest windows until the target C2 multiplets are present)."""
function _half_block!(dir, H, C, D, wanted; act, tol=1e-10)
    isdir(joinpath(dir, "eig.ckpt")) && return _checked_load(joinpath(dir, "eig.ckpt"))
    E = Float64[]; V = zeros(D, 0); gap = 0.0; have = false
    for k in unique(min.((6, 10, 16, 24), max(D - 2, 1)))
        E, Vs, _ = lowest(v -> *(H, v; num_th=act), D, k, 424242 + k; tol)
        V = reduce(hcat, Vs); gap = length(E) > 1 ? E[end] - E[end-1] : Inf
        c2, _ = post_c(C, V; workers=act)
        have = all(any(j -> abs(c2[j] - w) < 1e-4 && (j < length(c2) || length(c2) == D), eachindex(c2)) for w in wanted)
        (gap > 1e-6 || length(E) == D) && have && break
    end
    have || error("half-sector window: target C2 $(wanted) missing")
    rh = post_h(H, V, E; workers=act); c2, rc = post_c(C, V; workers=act)
    maximum(rh) <= 1e-6 && maximum(rc) <= 1e-6 && norm(V' * V - I) <= 1e-6 || error("half-sector H/C2/orthogonality gate")
    e = (energy=E, vectors=V, c2=c2, rh=rh, rc=rc, dim=D, boundary_gap=gap)
    _atomic_save(joinpath(dir, "eig.ckpt"), e); e
end

"""
    run_ope_stage(cfg, spaces, basis_sha)

cfg["ope"]: edges (Gamma^2 [rank, li, lf] on (0,0); default [[0,0,0],[2,0,2]]), half (true),
threshold (0.25, KK projection used for the projected bases), reference ({"f_J"=>..,"f_T"=>..} optional gate).
"""
function run_ope_stage(cfg, spaces, basis_sha)
    out = cfg["out"]; N = cfg["N"]; g = cfg["g"]; oc = get(cfg, "ope", Dict()); act = get(cfg, "action", Threads.nthreads())
    od = joinpath(out, "ope"); mkpath(od)
    lib = SegLibrary(builder=:tiled, tile_width=get(cfg, "tile", 64), workers=get(cfg, "workers", Threads.nthreads()),
                     opmat_threads=get(cfg, "native", 8))
    sp0 = spaces isa SpaceStore ? getspace!(spaces, 0) : spaces[1]
    ss = [sp0, sp0]; report = Dict{String,Any}("N" => N, "g" => g, "radius_squared" => N)
    edges = Dict{Tuple{Int,Int,Int},Any}()
    for (rank, li, lf) in get(oc, "edges", [[0, 0, 0], [2, 0, 2]])
        path = joinpath(od, "edges", "g2_r$(rank)_L$(li)_to_L$(lf).ckpt")
        S0 = source_window(out, 0, 0, li; vectors=false); T0 = source_window(out, 0, 0, lf; vectors=false)
        (S0 === nothing || T0 === nothing) && continue
        old = nothing
        if isdir(path)                     # v0.1.14: reuse only if built from exactly these windows
            old = _checked_load(path)
            (old.source_sha == S0.sha && old.target_sha == T0.sha) || (old = (old, _supersede!(path)))
        end
        if !isdir(path)
            S = source_window(out, 0, 0, li); T = source_window(out, 0, 0, lf)
            S.basis_sha == T.basis_sha == basis_sha || error("OPE windows computed in another segment basis")
            cs = BuildCompSpace(ss, [2N, 0, 0], 2li); ct = BuildCompSpace(ss, [2N, 0, 0], 2lf)
            terms = gamma2_terms(N, rank); SS = _colsha(S); TS = _colsha(T)      # v0.1.16: exact column identities
            A = old === nothing || get(old[1], :basis_sha, basis_sha) != basis_sha ? nothing :
                edge_arrays_incremental(lib, ss, cs, ct, terms, 2rank, S.V, S.E, T.V, T.E, old[1]; workers=act, SS, TS)
            A === nothing && (A = edge_arrays(lib, ss, cs, ct, terms, 2rank, S.V, T.V; workers=act))
            _atomic_save(path, (rank=rank, li=li, lf=lf, source_sha=S.sha, target_sha=T.sha, basis_sha=basis_sha,
                                source_E=S.E, source_C2=S.C2, target_E=T.E, target_C2=T.C2, overlaps=A.overlaps,
                                image_gram=A.image_gram, source_colsha=SS, target_colsha=TS, reused=get(A, :reused, nothing)))
        end
        edges[(rank, li, lf)] = _checked_load(path)
    end
    # ---- integer-sector anchors (raw and KK-projected bases) ------------------------------
    nrm = NaN
    if haskey(edges, (0, 0, 0)) && haskey(edges, (2, 0, 2))
        e0 = edges[(0, 0, 0)]; e2 = edges[(2, 0, 2)]; M = e0.overlaps[1]; A = e2.overlaps[1]
        raw0 = _raw_states(e0.source_E, e0.source_C2); raw2 = _raw_states(e2.target_E, e2.target_C2)
        sets = Any[("raw", raw0, raw2)]
        thr = get(oc, "threshold", 0.25)
        proj(L) = (p = joinpath(out, "kk", "projections", "s0_s0_L$(L).ckpt"); isdir(p) ?
                   last(first(filter(x -> x[1] == thr, _checked_load(p).threshold_channels))) : nothing)
        p0 = proj(0); p2 = proj(2)
        p0 === nothing || push!(sets, ("projected_scalar_raw_spin2", _projected_states(length(e0.source_E), p0), raw2))
        (p0 === nothing || p2 === nothing) || push!(sets, ("projected_all", _projected_states(length(e0.source_E), p0),
                                                           _projected_states(length(e2.target_E), p2)))
        rows = Any[]
        for (label, s0, s2) in sets
            r = _ope_anchors(label, s0, s2, M, A)
            append!(rows, r.rows); report["status_$label"] = r.status
            label == "raw" && hasproperty(r, :normalization) && (nrm = r.normalization)
        end
        _csv(joinpath(od, "ope_integer.csv"), ["basis", "channel", "source_state", "target_state", "source_energy", "target_energy",
             "source_C2", "target_C2", "signed_amplitude_arbitrary_state_phase", "phi_normalization", "factor", "finite_N_estimate",
             "source_energy_mixing_residual", "target_energy_mixing_residual"], rows)
        report["density_normalization"] = nrm
        for r in rows; report["$(r[1]):$(r[2])"] = r[12]; end
        v = _pick(e0.source_E, e0.source_C2, 0); p = _pick(e0.source_E, e0.source_C2, 2); t = _pick(e0.source_E, e0.source_C2, 5)
        isempty(p) || (report["phi_energy"] = e0.source_E[p[1]])
        isempty(t) || (report["tensor_energy"] = e0.source_E[t[1]])
    end
    # ---- scalar / tensor towers phi x phi -> every L0 state (raw and KK-projected), v0.1.12 -------
    if haskey(edges, (0, 0, 0))
        e0 = edges[(0, 0, 0)]; M = e0.overlaps[1]
        v = _pick(e0.source_E, e0.source_C2, 0); p = _pick(e0.source_E, e0.source_C2, 2)
        if !isempty(v) && !isempty(p)
            n0 = abs(M[p[1], v[1]]); isfinite(nrm) || (nrm = n0); report["density_normalization"] = n0
            cal = joinpath(out, "kk", "calibration.ckpt")
            alpha, E0 = isdir(cal) ? (c = _checked_load(cal); (c.alpha, c.E0)) : (NaN, e0.source_E[v[1]])
            cg(c) = abs(c - 5) < 1e-4 ? sqrt(5) / 2 : 1.0
            rows = Any[]
            for i in eachindex(e0.source_E)
                c = round(Int, e0.source_C2[i]); (c == 0 || c == 5) || continue
                push!(rows, ("raw", "raw_$i", 0, c, e0.source_E[i], alpha * (e0.source_E[i] - E0), M[p[1], i], cg(c),
                             cg(c) * abs(M[p[1], i]) / n0, 0.0))
            end
            thr = get(oc, "threshold", 0.25)
            pp = joinpath(out, "kk", "projections", "s0_s0_L0.ckpt")
            if isdir(pp)
                for (t, chans) in _checked_load(pp).threshold_channels, ch in chans
                    (ch.Q === nothing || !(ch.C2 in (0, 5))) && continue
                    for j in axes(ch.Q, 2)
                        q = zeros(length(e0.source_E)); q[ch.source_ids] = ch.Q[:, j]
                        amp = dot(M[p[1], :], q)
                        push!(rows, ("projected_t$(t)", "C2_$(ch.C2)_m$(j)", t, ch.C2, ch.energies[j], alpha * (ch.energies[j] - E0),
                                     amp, cg(ch.C2), cg(ch.C2) * abs(amp) / n0, ch.energy_mixing_residual[j]))
                    end
                end
            end
            _csv(joinpath(od, "phi_phi_towers.csv"), ["basis", "state", "threshold", "C2", "energy", "delta_current_scaled",
                 "signed_amplitude_arbitrary_phase", "CG_factor", "finite_N_magnitude", "energy_mixing_residual"], rows)
            report["tower_rows"] = length(rows)
        end
    end
    # ---- half-integer (1/2,1/2) sector: phi phi J and phi phi T -------------------------------
    if get(oc, "half", true) && isfinite(nrm)
        hd = joinpath(od, "half")
        segp = joinpath(hd, "segment.ckpt")
        isdir(segp) || _atomic_save(segp, (N=N, space=build_half_segment(N)))
        seg = _checked_load(segp).space
        hs = [seg, seg]; tpl = hamiltonian_template(N)
        sp = [BuildCompSpace(hs, [2N, 0, 2], 2l) for l in 0:1]
        haskey(HALF_DIMS, N) && (sp[1].dim, sp[2].dim) != HALF_DIMS[N] && error("half-sector dimension mismatch")
        sH, fH = segops_shared!(lib, hs, tpl.template); sC, fC = segops_shared!(lib, hs, tpl.c2)
        win = Dict{Int,Any}()
        for (l, wanted) in ((0, (2.0, 5.0)), (1, (3.0,)))
            H = absorb!(set_g!(BuildCompOperator(sp[l+1], tpl.template, sH; num_th=act), tpl, g), fH)
            C = absorb!(BuildCompOperator(sp[l+1], tpl.c2, sC; num_th=act), fC)
            win[l] = _half_block!(joinpath(hd, "blocks", "L$l"), H, C, sp[l+1].dim, wanted; act)
        end
        pick1(w, c) = (i = _pick(w.energy, w.c2, c); isempty(i) && error("half sector: missing C2=$c"); i[1])
        ip = pick1(win[0], 2); it = pick1(win[0], 5); ij = pick1(win[1], 3)
        phi = win[0].vectors[:, ip]; ten = win[0].vectors[:, it]; cur = win[1].vectors[:, ij]
        t0 = only(gamma2_terms(N, 0)); t1 = only(gamma2_terms(N, 1))     # one CoupleDecomps each
        s0, f0 = segops_shared!(lib, hs, t0); s1, f1 = segops_shared!(lib, hs, t1)
        op0 = absorb!(BuildCompOperator(sp[1], sp[1], t0, s0, 0; disp_std=false, num_th=act), f0)
        op1 = absorb!(BuildCompOperator(sp[1], sp[2], t1, s1, 2; disp_std=false, num_th=act), f1)
        op1b = absorb!(BuildCompOperator(sp[2], sp[1], t1, s1, 2; disp_std=false, num_th=act), f1)
        amp_t = dot(phi, *(op0, ten; num_th=act)); amp_tb = dot(ten, *(op0, phi; num_th=act))
        amp_j = dot(cur, *(op1, phi; num_th=act)); amp_jb = dot(phi, *(op1b, cur; num_th=act)) / sqrt(3)
        signed = max(abs(amp_t - amp_tb), abs(amp_j + amp_jb))
        signed <= 1e-7 || error("half sector: signed rank0/rank1 conjugacy gate failed ($signed)")
        f_j = sqrt(3) * abs(amp_j) / nrm; f_t = sqrt(2) * abs(amp_t) / nrm
        merge!(report, Dict("f_phi_phi_J" => f_j, "f_phi_phi_T_half" => f_t, "half_dims" => [sp[1].dim, sp[2].dim],
            "half_phi_energy" => win[0].energy[ip], "half_tensor_energy" => win[0].energy[it], "half_current_energy" => win[1].energy[ij],
            "half_signed_conjugacy_error" => signed,
            "J_factor" => "sqrt(3)*abs(Gamma2_rank1)/abs(vacuum_phi_Gamma0)", "T_factor" => "sqrt(2)*abs(Gamma2_rank0)/abs(vacuum_phi_Gamma0)"))
        haskey(report, "phi_energy") && (report["half_vs_integer_phi_energy_difference"] = win[0].energy[ip] - report["phi_energy"])
        ref = get(oc, "reference", nothing)
        if ref !== nothing
            abs(f_j - ref["f_J"]) <= 1e-7 && abs(f_t - ref["f_T"]) <= 1e-7 ||
                error("OPE reference mismatch: f_J=$f_j (ref $(ref["f_J"])), f_T=$f_t (ref $(ref["f_T"]))")
            report["reference_check"] = "passed"
        end
    end
    report["interpretation"] = "raw finite-N estimates; no continuum extrapolation; not certified CFT OPE data"
    open(io -> TOML.print(io, report), joinpath(od, "ope.toml"), "w")
    report
end
