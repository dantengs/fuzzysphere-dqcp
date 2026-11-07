# KK target windows on the Tcc library.
#
# For every (sA,sB) pair the KK projection of a source block (L = s) needs complete target
# windows in L' in {s-1, s, s+1} up to the highest source energy.  For each target block
# L' = 0..Lmax+1 the run's own window is reused when it is provably complete up to the
# required energy; otherwise the lowest window is solved with growing k (exchange-parity
# sectors on diagonal blocks), re-checked with an independent seed (energies and subspace),
# C2 resolved inside numerically degenerate clusters, and checkpointed atomically.
# Acceptance rules: frontier_status / target_requests.

"""Largest source L of a run (v0.2.2): spectrum.lmax, or the largest L of an explicit block list added later
with `extend` (lmax itself is a frozen physics key, so a window extension to L = lmax+1 lists the blocks)."""
source_lmax(cfg) = max(Int(get(cfg, "lmax", 2)), maximum((Int(b[3]) for b in get(cfg, "blocks", Any[]) if b isa AbstractVector); init=0))

"""States requested per attempt: initial, then x1.5 (at least +8) up to cap (frozen rule)."""
function target_requests(initial::Int, cap::Int, dim::Int)
    out = Int[]; n = min(initial, dim)
    while true
        push!(out, n); n >= min(cap, dim) && break
        n = min(cap, dim, max(n + 8, ceil(Int, 1.5n)))
    end
    unique(out)
end

"""Energy up to which a sector window is complete (Inf: whole sector; -Inf: cut inside a cluster)."""
function sector_frontier(E::AbstractVector, dim::Int; guard=1e-5)
    isempty(E) && return dim == 0 ? Inf : -Inf
    length(E) == dim && return Inf
    length(E) > 1 && E[end] - E[end-1] > guard ? E[end] : -Inf
end

"""Frontier of a (possibly parity-resolved) window: minimum over its sectors."""
function window_frontier(E, parity, sector_dims::Dict{Int,Int}; guard=1e-5)
    minimum(sector_frontier(sort(E[parity .== s]), n; guard) for (s, n) in sector_dims; init=Inf)
end

"""Rotate each numerically degenerate cluster so that C2 is diagonal inside it."""
function resolve_c2!(E, V, parity, C; tol=1e-7, workers=Threads.nthreads(), frozen=nothing, H=nothing)
    # frozen (v0.1.14): columns kept from a prior window; clusters made only of frozen columns are
    # left bit-identical (they were resolved when that window was written)
    i = 1
    while i <= length(E)
        j = i
        while j < length(E) && E[j+1] - E[i] <= tol; j += 1; end
        if j > i
            for s in unique(parity[i:j])                       # never mix exchange parities
                idx = [k for k in i:j if parity[k] == s]; length(idx) < 2 && continue
                frozen !== nothing && all(frozen[idx]) && continue
                W = V[:, idx]; CW = zeros(size(W))
                act_block!(CW, C, W; workers)
                U = eigen(Hermitian(W' * CW)).vectors; V[:, idx] = W * U
            end
        end
        i = j + 1
    end
    # v0.1.16 (V15-5): frozen columns are never rotated by the polish either
    H === nothing || polish_c2!(E, V, parity, H, C; workers, frozen)
    V
end

"""
    polish_c2!(E, V, parity, H, C; window=1e-3, trigger=1e-7) -> number of polished groups

v0.1.15: accidental near-degeneracies between different SO(5) irreps (N9 (0,0) L4: C2 = 5 and 14
states 1.8e-5 apart) leave converged eigenvectors mixed at the Lanczos tolerance (mixing ~ r/gap,
C2 residual 1.3e-6).  For every column whose C2 residual exceeds `trigger`, the same-parity states
within `window` in energy are rotated jointly: C2 is diagonalised in their span, then H inside each
C2 eigenspace (Rayleigh-Ritz).  Energies are replaced by the Ritz values (changes ~1e-12); columns
outside such groups are untouched.

v0.1.16 (V15-5): `frozen` (Bool mask) marks columns that must stay bit-identical (kept from an
earlier, accepted window); they neither trigger a group nor join one.  A residual that can only be
removed by rotating a frozen column is left in place and caught by the window gate.
"""
function polish_c2!(E, V, parity, H, C; window=1e-3, trigger=1e-7, workers=Threads.nthreads(), frozen=nothing)
    isempty(E) && return 0
    isfrozen(j) = frozen !== nothing && frozen[j]
    _, rc = post_c(C, V; workers)
    done = falses(length(E)); ngroups = 0
    for i in sortperm(rc; rev=true)
        (rc[i] <= trigger || done[i] || isfrozen(i)) && continue
        idx = [j for j in eachindex(E) if parity[j] == parity[i] && abs(E[j] - E[i]) < window && !isfrozen(j)]
        length(idx) < 2 && continue
        W = V[:, idx]; CW = zeros(size(W)); act_block!(CW, C, W; workers)
        F = eigen(Hermitian((W' * CW + CW' * W) / 2)); U = W * F.vectors
        groups = Vector{Int}[]                              # clusters of C2 eigenvalues
        for (k, c) in enumerate(F.values)
            isempty(groups) || abs(c - F.values[last(groups)[1]]) > 1e-6 ? push!(groups, [k]) : push!(last(groups), k)
        end
        newV = Vector{Float64}[]; newE = Float64[]
        for gk in groups
            Ug = U[:, gk]; HU = zeros(size(Ug)); act_block!(HU, H, Ug; workers)
            G = eigen(Hermitian((Ug' * HU + HU' * Ug) / 2)); R = Ug * G.vectors
            append!(newE, G.values); append!(newV, [R[:, q] for q in axes(R, 2)])
        end
        o = sortperm(newE); ix = sort(idx; by=j -> E[j])
        for (q, j) in enumerate(ix); E[j] = newE[o[q]]; V[:, j] = newV[o[q]]; end
        done[idx] .= true; ngroups += 1
    end
    ngroups
end

_parity_set(p) = p == "plus" ? (1,) : p == "minus" ? (-1,) : p == "both" ? (1, -1) : error("target parity must be both | plus | minus")

function _target_window(H, cp, a, b, k::Int, seed::Int, useX::Bool, act::Int, tol; parity="both", prior=nothing)
    # prior (v0.1.14): an existing window (energy, parity, full-D vectors, optional `frozen` mask of
    # columns already C2-resolved); its columns are kept bit-identical and each sector only computes
    # the states that follow them (lowest_deflated).  Returned `frozen`: resolved prior columns.
    D = cp.dim
    secs = if useX && a == b
        perm, sgn = exchange_map(cp)
        [(s, ExchangeSector(perm, sgn, s)) for s in _parity_set(parity)]
    else
        [(0, nothing)]
    end
    E = Float64[]; P = Int[]; vecs = Vector{Float64}[]; dims = Dict{Int,Int}(); old = Bool[]
    for (s, x) in secs
        n = x === nothing ? D : length(x); dims[s] = n; n == 0 && continue
        f = x === nothing ? (v -> *(H, v; num_th=act)) : reduced_action(H, x; num_th=act)
        oc = prior === nothing ? Int[] : findall(==(s), prior.parity)
        for c in oc
            push!(E, prior.energy[c]); push!(P, s); push!(vecs, prior.vectors[:, c])
            push!(old, hasproperty(prior, :frozen) && prior.frozen[c])
        end
        kadd = k - length(oc); kadd <= 0 && continue
        e, V = if isempty(oc)
            lowest(f, n, k, seed + 7(s + 1); tol)[1:2]
        else
            V0 = zeros(n, length(oc))
            for (j, c) in enumerate(oc)
                x === nothing ? (V0[:, j] .= prior.vectors[:, c]) : sector_reduce!(view(V0, :, j), x, prior.vectors[:, c])
            end
            r = lowest_deflated(f, n, kadd, V0, seed + 7(s + 1) + 104729; tol, Eref=prior.energy[oc])
            !isempty(r[1]) && minimum(r[1]) < maximum(prior.energy[oc]) - 1e-8 &&
                error("target window sector $s: a deflated state lies below the prior window")
            r[1:2]
        end
        for (i, v) in enumerate(V)
            push!(E, e[i]); push!(P, s); push!(vecs, x === nothing ? v : sector_expand!(zeros(D), x, v)); push!(old, false)
        end
    end
    o = sortperm(E)
    (energy=E[o], parity=P[o], vectors=isempty(vecs) ? zeros(D, 0) : reduce(hcat, vecs[o]), sector_dims=dims, frozen=old[o])
end

"""
    extend_targets(cfg) -> Dict(key => status)

cfg: N, g, out (run_spectrum directory: blocks/*/eig.ckpt, result.ckpt), pairs (default: every pair
of the run), lmax (largest source L, default 2), initial_nev (default 12), max_nev (default 100),
guard (1e-5), independent_check (true), exchange (true), native, action, tile, tol, source_delta_max (v0.2.7), and the
segment source of the run (segment_checkpoint, or out/segment_spaces.ckpt).
Writes out/targets/s{a}_s{b}_L{l}.ckpt.
"""
function extend_targets(cfg)
    N = cfg["N"]; g = cfg["g"]; out = cfg["out"]; lmax = source_lmax(cfg)
    guard = get(cfg, "guard", 1e-5); tol = get(cfg, "tol", 1e-10)
    init = get(cfg, "initial_nev", 12); cap = get(cfg, "max_nev", 100)
    useX = get(cfg, "exchange", true); native = get(cfg, "native", 8); action = get(cfg, "action", 64)
    st = Stage(joinpath(out, "stages_targets.tsv")); FuzzifiED.NumThreads = native
    seg = get(cfg, "segment_checkpoint", "")
    split = joinpath(out, "segment_split")
    store = (get(cfg, "residency", "all") != "all" && isdir(joinpath(split, "space_2"))) ? SpaceStore(split) : nothing
    spaces, basis_sha = if store !== nothing
        (nothing, _checked_load(joinpath(split, "bases")).source_sha)
    elseif isempty(seg)
        own = joinpath(out, "segment_spaces.ckpt")
        (_checked_load(own).spaces, strip(read(joinpath(own, "sha256.txt"), String)))
    else
        s = load_segment_spaces(seg); (s.spaces, s.sha)
    end
    tpl = hamiltonian_template(N)
    lib = SegLibrary(builder=:tiled, tile_width=get(cfg, "tile", 64), workers=get(cfg, "workers", Threads.nthreads()),
                     opmat_threads=native)
    blockdir(a, b, l) = joinpath(out, "blocks", "s$(a)_s$(b)_L$(l)")
    pairs = get(cfg, "pairs", nothing)
    pairs === nothing && (pairs = unique([(parse(Int, m[1]), parse(Int, m[2])) for m in
                                          (match(r"^s(\d)_s(\d)_L\d$", d) for d in readdir(joinpath(out, "blocks"))) if m !== nothing]))
    status = Dict{String,Any}()
    _recover_replace!(joinpath(out, "targets"))      # v0.1.16: complete or roll back an interrupted replacement
    nbrs(s) = s == 0 ? [1] : [s - 1, s, s + 1]
    for (a, b) in sort(collect(pairs))
        win = Dict{Int,Any}()
        for s in 0:lmax
            isdir(joinpath(blockdir(a, b, s), "eig.ckpt")) || continue
            e = _checked_load(joinpath(blockdir(a, b, s), "eig.ckpt"))
            e.basis_sha == basis_sha || error("block $(a)$(b)L$s was computed in another segment basis")
            win[s] = e
        end
        required = Dict(l => maximum((maximum(win[s].energy; init=-Inf) for s in keys(win) if l in nbrs(s)); init=-Inf) for l in 0:lmax+1)
        # v0.2.7: targets.source_delta_max caps the energy the target windows must reach.  Without it a target
        # chases the highest source energy of its neighbours (source windows can end far above the coverage
        # cut); with it, KK covers every source state up to the cap (sources above it are reported as
        # partial_source_coverage by kk_classify, which only uses sources below the target frontier).
        dmax = get(cfg, "source_delta_max", nothing)
        if dmax !== nothing
            F = tuning_F(run_rows(out)); F === nothing && error("targets.source_delta_max needs ground/current/stress rows")
            ecap = F.E0 + Float64(dmax) * (F.EJ - F.E0) / 2
            for l in collect(keys(required)); required[l] = min(required[l], ecap); end
        end
        if store !== nothing                         # residency pair/minimal: <= 2 spaces resident
            for s in collect(keys(store.live)); s in (a, b) || dropspace!(store, lib, s); end
        end
        ss = store === nothing ? [spaces[a+1], spaces[b+1]] : [getspace!(store, a), getspace!(store, b)]
        sH, fH = segops_shared!(lib, ss, tpl.template); sC, fC = segops_shared!(lib, ss, tpl.c2)
        lastacc = nothing                               # v0.1.17: last complete target of this pair (for the first k)
        for l in 0:lmax+1
            key = "s$(a)_s$(b)_L$(l)"; dest = joinpath(out, "targets", key * ".ckpt")
            need = required[l]
            cp = BuildCompSpace(ss, [2N, 0, 0], 2l); D = cp.dim
            tpar = get(cfg, "target_parity", "both")
            # v0.1.16 (V15-2): a target window is the lowest window of one fixed operator; it is bound to
            # exactly that operator (N, g, segment basis, block, dimension, parity restriction), not to
            # the source windows, which only set the energy it must cover.
            tid = (N=N, g=g, basis_sha=basis_sha, key=key, dim=D, exchange=useX && a == b, target_parity=string(tpar))
            secdims_ref = Ref{Any}(nothing)
            secdims() = secdims_ref[] !== nothing ? secdims_ref[] : (secdims_ref[] = (useX && a == b) ?
                (p = exchange_map(cp); Dict(s => length(ExchangeSector(p[1], p[2], s)) for s in _parity_set(tpar))) : Dict(0 => D))
            prev = nothing                                  # window to deflate from (v0.1.14)
            oldn = -1                                       # states of the active target (superseded name)
            if isdir(dest)
                t = _checked_load(dest); oldn = length(t.energy)
                _target_identity_ok(t, tid) || gatefail("target $dest belongs to another operator (N, g, basis, block, dim or parity restriction differ)")
                if t.status == "empty_block" || need == -Inf
                    status[key] = "exists"; continue
                end
                hasproperty(t, :sector_dims) && t.sector_dims !== nothing && (secdims_ref[] = Dict{Int,Int}(t.sector_dims))
                why = _target_invalid(t, secdims(), guard; independent=_check_kind(get(cfg, "independent_check", true)) != "none")
                if why !== nothing
                    # never reused, never deflated from: moved aside and recomputed from scratch
                    println("TARGET_REJECT $key: $why")
                    mv(dest, _superseded_name(dest, "rejected")); oldn = -1
                    status[key] = "rejected"
                elseif window_frontier(t.energy, t.parity, secdims(); guard) > need + guard
                    lastacc = (energy=t.energy, parity=t.parity, D=D)
                    status[key] = "exists"; continue
                else
                    # a guarded window was C2-resolved and gated when written: its columns stay frozen
                    isempty(t.energy) || (prev = (energy=t.energy, parity=t.parity, vectors=t.vectors,
                                                 frozen=fill(t.status == "guarded_lowest_window", length(t.energy)), src=string(t.status)))
                    status[key] = "extending"
                end
            end
            save_target!(payload) = oldn < 0 ? _atomic_save(dest, payload) : _replace!([dest => payload]; tag="k$(oldn)")
            if D == 0 || need == -Inf
                save_target!((key=key, N=N, g=g, dim=D, energy=Float64[], parity=Int[], vectors=zeros(D, 0),
                              c2=Float64[], rh=Float64[], rc=Float64[], required_energy=need, frontier=Inf,
                              status=D == 0 ? "empty_block" : "no_source", basis_sha=basis_sha, identity=tid,
                              producer=producer(), colsha=String[]))
                status[key] = D == 0 ? "empty_block" : "no_source"; continue
            end
            H = absorb!(set_g!(BuildCompOperator(cp, tpl.template, sH; num_th=action), tpl, g), fH)
            C = absorb!(BuildCompOperator(cp, tpl.c2, sC; num_th=action), fC)
            # v0.1.12: target_parity restricts diagonal targets to one exchange sector (e.g. the singlet
            # tower lives in X = +1 of (0,0)); the window frontier is then taken over that sector only.
            sd = secdims()
            # 1. reuse the run's window when it is complete up to the requirement and passes the gate
            #    (v0.1.16, V15-4: residuals, orthonormality, integer C2 -- otherwise solved afresh)
            if haskey(win, l)
                e = win[l]
                keep = [i for i in eachindex(e.parity) if haskey(sd, e.parity[i])]
                e = (energy=e.energy[keep], parity=e.parity[keep], vectors=e.vectors[:, keep])
                fr = window_frontier(e.energy, e.parity, sd; guard)
                if fr > need + guard
                    c2, rc = post_c(C, e.vectors; workers=action); rh = post_h(H, e.vectors, e.energy; workers=action)
                    gd = try
                        window_gate("target $key (run window)", e.energy, e.vectors; rh, rc, c2)
                    catch err
                        err isa GateError || rethrow()
                        println("TARGET_RUNWINDOW_REJECT $key: ", sprint(showerror, err)); nothing
                    end
                    if gd !== nothing
                        save_target!((key=key, N=N, g=g, dim=D, energy=e.energy, parity=e.parity, vectors=e.vectors,
                                      c2=c2, rh=rh, rc=rc, required_energy=need, frontier=fr, status="reused_run_window",
                                      basis_sha=basis_sha, identity=tid, producer=producer(), colsha=column_shas(e.vectors),
                                      sector_dims=sd, gate=gd))
                        lastacc = (energy=e.energy, parity=e.parity, D=D)
                        status[key] = "reused_run_window"; continue
                    end
                end
            end
            if haskey(win, l)
                e = win[l]; keep = [i for i in eachindex(e.parity) if haskey(sd, e.parity[i])]
                # the run window replaces the old target only when it is its own lineage (reused run window:
                # the extended run block keeps those vectors bit-identical); a guarded target keeps its own
                if prev === nothing || (prev.src == "reused_run_window" && length(keep) > length(prev.energy))
                    prev = (energy=e.energy[keep], parity=e.parity[keep], vectors=e.vectors[:, keep],
                            frozen=falses(length(keep)), src="run_window")
                end
            end
            # 2. fresh lowest windows with growing k, independent re-check.
            # v0.1.13: every attempt is checkpointed (failed attempts: energies only; the accepted attempt
            # with its vectors until the final target is written), as is the independent check, so an
            # interrupted job resumes at the next untried k instead of restarting the block.
            # v0.1.14: each attempt is deflated from the previous one (its vectors are kept, only the
            # following states are solved); the latest failed attempt keeps its vectors, older ones
            # are reduced to energies.
            # v0.1.16 (V15-2/3/9): attempts and checks carry the operator identity (+ tol, guard); a cached
            # attempt is judged against the *current* requirement (its stored `passed` is ignored), a
            # cached check is used only for the very attempt it checked (payload SHA of that attempt).
            adir = joinpath(out, "targets", "attempts", key)
            aid = merge(tid, (tol=tol, guard=guard))
            _validate_attempts!(adir, aid)
            pcount(p) = p === nothing ? 0 : maximum((count(==(s), p.parity) for s in keys(sd)); init=0)
            maxdim = maximum(values(sd))
            short(x, k) = !all(count(==(s), x.parity) >= min(k, n) for (s, n) in sd)
            found = nothing
            if get(cfg, "target_growth", "fresh") == "fresh"
                # v0.1.17: every attempt is a fresh solve (deflating from a failed attempt costs more than
                # solving the larger window from scratch: N9 (0,1) L4, 405 -> 608 states 285 s deflated vs
                # 296 s for all 608 fresh).  Deflation is kept only to extend a guarded target whose columns
                # must stay bit-identical.  The next k is extrapolated from the failed attempt's spectrum.
                sprior = (prev !== nothing && any(prev.frozen)) ? prev : nothing
                function find_fresh(k)
                    while k <= pcount(sprior) && k < maxdim; k = min(maxdim, max(k + 8, ceil(Int, 1.5k))); end
                    while true
                        ap = joinpath(adir, "k$k.ckpt"); x = nothing
                        if isdir(ap)
                            x = _checked_load(ap)
                            hasproperty(x, :identity) && x.identity == aid || gatefail("attempt $ap: identity differs from its directory sidecar")
                            if short(x, k)
                                println("TARGET_ATTEMPT_SHORT $key k=$k"); mv(ap, _superseded_name(ap, "short")); x = nothing
                            elseif hasproperty(x, :vectors) && window_frontier(x.energy, x.parity, sd; guard) > need + guard
                                return (x, ap, k)
                            end
                        end
                        if x === nothing
                            y = stage!(st, "target_$(key)_k$k"; detail="D=$D need=$need prior=$(pcount(sprior)) fresh") do
                                _target_window(H, cp, a, b, k, 71231 + 100a + 10b + l, useX, action, tol; parity=tpar, prior=sprior)
                            end
                            fry = window_frontier(y.energy, y.parity, sd; guard)
                            if fry > need + guard
                                _atomic_save(ap, (k=k, energy=y.energy, parity=y.parity, frontier=fry, passed=true, vectors=y.vectors,
                                                  frozen=y.frozen, identity=aid, required_energy=need, producer=producer()))
                                return (y, ap, k)
                            end
                            _atomic_save(ap, (k=k, energy=y.energy, parity=y.parity, frontier=fry, passed=false, identity=aid,
                                              required_energy=need, producer=producer()))
                            x = y
                        end
                        k >= min(cap, maxdim) && return nothing
                        k = _next_k(k, x.energy, x.parity, sd, need, guard, cap, maxdim)
                    end
                end
                # v0.1.17b: first k estimated from the run window of this block (extrapolated like a failed
                # attempt) or, for L = lmax+1, from the previous target of the pair scaled by the dimension
                # ratio; never below initial_nev.  targets.first_k = "initial" restores the plain start.
                k0 = min(init, maxdim)
                if get(cfg, "first_k", "estimate") == "estimate"
                    est = if haskey(win, l)
                        e = win[l]; kp = [i for i in eachindex(e.parity) if haskey(sd, e.parity[i])]
                        _initial_k(e.energy[kp], e.parity[kp], sd, need, guard, cap, maxdim, 1.0)
                    elseif lastacc !== nothing && lastacc.D > 0
                        _initial_k(lastacc.energy, lastacc.parity, sd, need, guard, cap, maxdim, D / lastacc.D)
                    else
                        nothing
                    end
                    est === nothing || (k0 = max(k0, est))
                end
                found = find_fresh(k0)
            else
                # v0.1.14 deflation chain (target_growth = "deflate")
                lastap = ""
                for k in target_requests(init, cap, maxdim)
                    k <= pcount(prev) && continue                # the prior window already holds k per sector
                    ap = joinpath(adir, "k$k.ckpt")
                    if isdir(ap)
                        x = _checked_load(ap)
                        hasproperty(x, :identity) && x.identity == aid || gatefail("attempt $ap: identity differs from its directory sidecar")
                        if hasproperty(x, :vectors) && window_frontier(x.energy, x.parity, sd; guard) > need + guard
                            found = (x, ap, k); break
                        end
                        hasproperty(x, :vectors) && (prev = (energy=x.energy, parity=x.parity, vectors=x.vectors,
                                                               frozen=x.frozen, src="attempt"); lastap = ap)
                        short(x, k) || continue
                        println("TARGET_ATTEMPT_SHORT $key k=$k"); mv(ap, _superseded_name(ap, "short")); lastap = ""
                    end
                    y = stage!(st, "target_$(key)_k$k"; detail="D=$D need=$need prior=$(pcount(prev))") do
                        _target_window(H, cp, a, b, k, 71231 + 100a + 10b + l, useX, action, tol; parity=tpar, prior=prev)
                    end
                    fry = window_frontier(y.energy, y.parity, sd; guard)
                    _atomic_save(ap, (k=k, energy=y.energy, parity=y.parity, frontier=fry, passed=fry > need + guard, vectors=y.vectors,
                                      frozen=y.frozen, identity=aid, required_energy=need, producer=producer()))
                    if !isempty(lastap)          # its vectors now live in this attempt: keep only its energies
                        z = _checked_load(lastap); rm(lastap; recursive=true, force=true)
                        _atomic_save(lastap, (k=z.k, energy=z.energy, parity=z.parity, frontier=z.frontier, passed=false, identity=aid))
                    end
                    lastap = ap
                    if fry > need + guard
                        found = (y, ap, k); break
                    end
                    prev = (energy=y.energy, parity=y.parity, vectors=y.vectors, frozen=y.frozen, src="attempt")
                end
            end
            found === nothing && error("INCOMPLETE_TARGET_WINDOW $key: cap $cap reached (raise max_nev)")
            w, ap, k = found
            begin
                wsha = _ckpt_sha(ap)
                ind = Dict{Symbol,Any}()
                ckind = _check_kind(get(cfg, "independent_check", true))
                fresh_growth = get(cfg, "target_growth", "fresh") == "fresh"
                ckind == "complement" && !fresh_growth && (ckind = "fresh")      # deflate mode keeps the v0.1.13 check
                while ckind != "none"
                    cpth = joinpath(adir, "check_k$k.ckpt")
                    chk = nothing
                    if isdir(cpth)
                        c0 = _checked_load(cpth)
                        if hasproperty(c0, :identity) && c0.identity == aid && get(c0, :attempt_sha, "") == wsha &&
                           get(c0, :kind, "fresh") == ckind
                            chk = c0
                        else
                            println("TARGET_CHECK_STALE $key k=$k"); mv(cpth, _superseded_name(cpth, "stale"))
                        end
                    end
                    if ckind == "fresh"
                        if chk === nothing
                            kc = pcount(w)                    # fresh, undeflated solve with as many states per sector
                            v = stage!(st, "target_$(key)_check"; detail="k=$kc") do
                                _target_window(H, cp, a, b, kc, 99871 + 100a + 10b + l, useX, action, tol; parity=tpar)
                            end
                            de = 0.0; sv = 1.0
                            for sct in keys(sd)                  # compare sector by sector
                                iw = findall(==(sct), w.parity); iv = findall(==(sct), v.parity); m = length(iw)
                                m == 0 && continue
                                if length(iv) < m; de = Inf; sv = 0.0; break; end
                                de = max(de, maximum(abs.(v.energy[iv[1:m]] - w.energy[iw])))
                                sv = min(sv, minimum(svdvals(v.vectors[:, iv[1:m]]' * w.vectors[:, iw])))
                            end
                            chk = (k=k, kind="fresh", energy=v.energy, max_energy_difference=de, min_singular_value=sv, identity=aid,
                                   attempt_sha=wsha, producer=producer())
                            _atomic_save(cpth, chk)
                        end
                        chk.max_energy_difference <= GATE.independent_de || gatefail("independent lowest-window mismatch in $key: dE $(chk.max_energy_difference)")
                        chk.min_singular_value >= GATE.independent_sv || gatefail("independent lowest-window subspace mismatch in $key: smin $(chk.min_singular_value)")
                        ind[:independent_kind] = "fresh"
                        ind[:independent_min_singular_value] = chk.min_singular_value
                        ind[:independent_max_energy_difference] = chk.max_energy_difference
                        ind[:independent_check_sha] = _ckpt_sha(cpth)
                        break
                    end
                    # v0.1.17 complement test: per sector, the lowest states of H on the orthogonal complement of the
                    # accepted window (fresh seed).  The window is the complete lowest window of its sector iff
                    # the complement starts above the window top; the gap must exceed the guard.
                    if chk === nothing
                        r = stage!(st, "target_$(key)_complement"; detail="k=$(pcount(w))") do
                            _complement_check(H, cp, a, b, w, sd, useX, action, tol, 99871 + 100a + 10b + l)
                        end
                        chk = merge(r, (k=k, kind="complement", identity=aid, attempt_sha=wsha, producer=producer()))
                        _atomic_save(cpth, chk)
                    end
                    chk.min_margin < -1e-7 &&
                        gatefail("complement test in $key: a state at $(chk.worst_lambda) lies inside the accepted window (top $(chk.worst_top)); the window missed it")
                    if chk.min_margin > guard
                        ind[:independent_kind] = "complement"
                        ind[:independent_min_margin] = chk.min_margin
                        ind[:independent_check_sha] = _ckpt_sha(cpth)
                        break
                    end
                    # the window top cuts a degenerate cluster: solve a larger window and test again
                    println("TARGET_COMPLEMENT_GROW $key k=$k margin=$(chk.min_margin)")
                    k >= min(cap, maxdim) && error("INCOMPLETE_TARGET_WINDOW $key: cap $cap reached (raise max_nev)")
                    found = find_fresh(min(cap, maxdim, max(k + 8, ceil(Int, 1.5k))))
                    found === nothing && error("INCOMPLETE_TARGET_WINDOW $key: cap $cap reached (raise max_nev)")
                    w, ap, k = found; wsha = _ckpt_sha(ap)
                end
                # C2 resolution + polish; frozen (prior, accepted) columns must come out bit-identical
                E = copy(w.energy); P = copy(w.parity)
                fz = hasproperty(w, :frozen) ? Vector{Bool}(w.frozen) : fill(false, length(E))
                sha0 = sort(column_shas(w.vectors[:, fz]))
                V = resolve_c2!(E, copy(w.vectors), P, C; workers=action, frozen=fz, H=H)
                o = _sortperm_stable(E)                     # the polish may move Ritz values by ~1e-12
                if o != eachindex(E)
                    E = E[o]; P = P[o]; V = V[:, o]; fz = fz[o]
                end
                colsha = column_shas(V)
                sort(colsha[fz]) == sha0 || gatefail("target $key: a frozen prior column changed during C2 resolution")
                c2, rc = post_c(C, V; workers=action); rh = post_h(H, V, E; workers=action)
                fr = window_frontier(E, P, sd; guard)
                fr > need + guard || gatefail("target $key: window no longer complete after C2 polish (frontier $fr, need $need)")
                gd = window_gate("target $key", E, V; rh, rc, c2)
                save_target!(merge((key=key, N=N, g=g, dim=D, energy=E, parity=P, vectors=V, c2=c2, rh=rh,
                                    rc=rc, required_energy=need, frontier=fr, status="guarded_lowest_window", requested=k,
                                    basis_sha=basis_sha, identity=tid, producer=producer(), colsha=colsha, sector_dims=sd,
                                    gate=gd, frozen_columns=count(fz), attempt_sha=wsha), NamedTuple(ind)))
                rm(ap; recursive=true, force=true)          # vectors now live in the final target checkpoint
                lastacc = (energy=E, parity=P, D=D)
                status[key] = "solved_k$k"
            end
            H = nothing; C = nothing; GC.gc(false)
        end
        store === nothing && for s in unique((a, b)); release!(lib, spaces[s+1]); end
    end
    store === nothing || for s in collect(keys(store.live)); dropspace!(store, lib, s); end
    status
end

# ---- v0.1.16 target identity / validity (V15-2, V15-4, V15-9) ---------------------------------

"""Identity of a stored target against the current operator; legacy targets (no `identity`) are
matched on N, g, basis, block dimension and the parities they contain."""
function _target_identity_ok(t, tid)
    (t.N == tid.N && t.g == tid.g && t.basis_sha == tid.basis_sha && t.dim == tid.dim) || return false
    hasproperty(t, :identity) && return t.identity == tid
    tid.exchange || return all(==(0), t.parity)
    tid.target_parity == "both" || return all(==(tid.target_parity == "plus" ? 1 : -1), t.parity)
    true
end

"""`nothing` when the stored target passes every acceptance check, else the reason."""
function _target_invalid(t, secdims, guard; independent::Bool=true)
    t.status in ("guarded_lowest_window", "reused_run_window", "no_source") || return "status $(t.status)"
    m = length(t.energy)
    size(t.vectors) == (t.dim, m) || return "vector shape $(size(t.vectors)) != ($(t.dim), $m)"
    length(t.parity) == length(t.c2) == length(t.rh) == length(t.rc) == m || return "row lengths differ"
    all(p -> haskey(secdims, p), t.parity) || return "parity outside the requested sectors"
    fr = window_frontier(t.energy, t.parity, secdims; guard)
    (fr == t.frontier || abs(fr - t.frontier) <= 1e-8) || return "stored frontier $(t.frontier) != recomputed $fr"
    if t.status == "guarded_lowest_window" && independent
        if hasproperty(t, :independent_min_margin)                        # v0.1.17 complement test
            t.independent_min_margin > guard || return "complement margin $(t.independent_min_margin)"
        else
            hasproperty(t, :independent_min_singular_value) || return "no independent-check receipt"
            t.independent_min_singular_value >= GATE.independent_sv || return "independent subspace $(t.independent_min_singular_value)"
            hasproperty(t, :independent_max_energy_difference) && t.independent_max_energy_difference > GATE.independent_de &&
                return "independent energy difference $(t.independent_max_energy_difference)"
        end
    end
    try
        window_gate("stored target $(t.key)", t.energy, t.vectors; rh=t.rh, rc=t.rc, c2=t.c2)
    catch err
        err isa GateError || rethrow(); return sprint(showerror, err)
    end
    hasproperty(t, :colsha) && t.colsha !== nothing && length(t.colsha) == m &&
        column_shas(t.vectors) != t.colsha && return "column SHA mismatch"
    nothing
end

_identity_digest(x) = bytes2hex(sha256(sprint(show, x)))

"""
Attempt directories carry an identity sidecar (identity.txt: digest of operator identity + tol +
guard).  A directory without it (written before v0.1.16) or with another identity is moved aside
(attempts/<key>.stale[.n], never deleted) and the block starts afresh.
"""
function _validate_attempts!(adir, aid)
    want = _identity_digest(aid); f = joinpath(adir, "identity.txt")
    if isdir(adir)
        isfile(f) && strip(read(f, String)) == want && return true
        if any(x -> endswith(x, ".ckpt"), readdir(adir))
            d = adir * ".stale"; n = 1; while ispath(d); d = adir * ".stale.$n"; n += 1; end
            println("TARGET_ATTEMPTS_STALE $adir -> $d"); mv(adir, d)
        end
    end
    mkpath(adir); write(f, want * "\n")
    true
end

"""v0.1.17: next attempt size from a failed attempt: at least x1.5 (+8), and per incomplete sector
the count extrapolated to the required energy with density growing like (E - E1)^1.5 (+10 %, +8)."""
function _next_k(k, E, P, sd, need, guard, cap, maxdim)
    kn = max(k + 8, ceil(Int, 1.5k))
    for (s, n) in sd
        Es = sort(E[P .== s]); m = length(Es)
        (m >= n || m < 2 || sector_frontier(Es, n; guard) > need + guard) && continue
        span = Es[end] - Es[1]; span > 0 || continue
        kn = max(kn, ceil(Int, 1.1 * m * ((need + 2guard - Es[1]) / span)^1.5) + 8)
    end
    min(kn, cap, maxdim)
end

"""
    _initial_k(E, P, sd, need, guard, cap, maxdim, ratio) -> Int or nothing

v0.1.17b: per-sector states needed to cover `need` (+2 guard), estimated from a known window (E, P):
the count below the requirement if the window reaches it, otherwise the extrapolation of `_next_k`
(level count ~ (E - E1)^1.5); `ratio` scales for a block of another dimension.  10-15 % margin + 8.
"""
function _initial_k(E, P, sd, need, guard, cap, maxdim, ratio)
    kn = 0
    for (s, n) in sd
        Es = sort(E[P .== s]); m = length(Es); m >= 2 || continue
        if Es[end] >= need + 2guard
            kn = max(kn, ceil(Int, 1.15 * ratio * count(<=(need + 2guard), Es)) + 8)
        else
            span = Es[end] - Es[1]; span > 0 || continue
            kn = max(kn, ceil(Int, 1.1 * ratio * m * ((need + 2guard - Es[1]) / span)^1.5) + 8)
        end
    end
    kn == 0 ? nothing : min(kn, cap, maxdim)
end

_check_kind(v) = v === true ? "complement" : v === false ? "none" : (string(v) in ("complement", "fresh", "none") ? string(v) :
                 error("targets.independent_check must be true | false | \"complement\" | \"fresh\" | \"none\""))

"""
    _complement_check(H, cp, a, b, w, sd, useX, act, tol, seed; nextra=2)

v0.1.17: for every sector of the accepted window `w`, the `nextra` lowest eigenvalues of H restricted to
the orthogonal complement of the window's columns in that sector (lowest_deflated, independent seed).
Returns margins = (lowest complement eigenvalue) - (window top) per sector (Inf for a complete sector).
"""
function _complement_check(H, cp, a, b, w, sd, useX, act, tol, seed; nextra::Int=2)
    D = cp.dim
    secs = (useX && a == b) ? (p = exchange_map(cp); Dict(s => ExchangeSector(p[1], p[2], s) for s in keys(sd))) : Dict(0 => nothing)
    margins = Dict{Int,Float64}(); lambdas = Dict{Int,Vector{Float64}}(); tops = Dict{Int,Float64}(); nops = 0
    for (s, n) in sd
        idx = findall(==(s), w.parity); m = length(idx)
        if m >= n
            margins[s] = Inf; lambdas[s] = Float64[]; tops[s] = m == 0 ? -Inf : maximum(w.energy[idx]); continue
        end
        x = secs[s]
        f = x === nothing ? (v -> *(H, v; num_th=act)) : reduced_action(H, x; num_th=act)
        V0 = zeros(n, m)
        for (j, c) in enumerate(idx)
            x === nothing ? (V0[:, j] .= view(w.vectors, :, c)) : sector_reduce!(view(V0, :, j), x, w.vectors[:, c])
        end
        top = m == 0 ? -Inf : maximum(w.energy[idx])
        E, _, info = lowest_deflated(f, n, nextra, V0, seed + 13s; tol, Eref=w.energy[idx])
        nops += info.numops
        lambdas[s] = E; tops[s] = top
        margins[s] = isempty(E) ? Inf : minimum(E) - top
    end
    worst = argmin(margins)
    (margins=margins, lambdas=lambdas, tops=tops, min_margin=margins[worst], worst_lambda=isempty(lambdas[worst]) ? Inf : minimum(lambdas[worst]),
     worst_top=tops[worst], numops=nops)
end
