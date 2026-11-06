# KK primary projection and OPE insertion edges on Tcc runs (v0.1.7).
#
# Port of the n11 production KK flow (kk/run_kk.jl: make_edge, calibration, classify_edges,
# export_pair) onto Tcc run directories:
#   source windows  = out/blocks/<key>/eig.ckpt (+ result.ckpt C2 labels)
#   target windows  = out/targets/<key>.ckpt (extend_targets)
#   generator edges = edge_arrays on the shared segment-operator library
# The mathematics (C2-preservation check, vacuum calibration with fit_vacuum_generator,
# K^2 = sum_lf (2lf+1)/(2li+1) K'K with strictly lowering mask, thresholds .05/.1/.25/.5,
# CSV exports) is unchanged.  The frozen N=11 legacy-coefficient regression is applied
# when the config provides it (kk.legacy_calibration).

include(joinpath(@__DIR__, "..", "kk", "KKProjection.jl"))   # verbatim n11 module
using .KKProjection: fit_vacuum_generator

_ckpt_sha(path) = strip(read(joinpath(path, "sha256.txt"), String))
_ids(C, c) = findall(x -> abs(x - c) < 1e-5, C)
"""Column SHA-256s of a window: stored ones when the payload carries them for every column, else computed."""
_colsha(W) = (W.colsha !== nothing && length(W.colsha) == size(W.V, 2)) ? Vector{String}(W.colsha) : column_shas(W.V)

"""Source window of a run block: energies, vectors, C2 labels and H residual bound."""
function source_window(out, a, b, l; vectors=true)
    d = joinpath(out, "blocks", "s$(a)_s$(b)_L$(l)")
    isdir(joinpath(d, "eig.ckpt")) || return nothing
    e = _checked_load(joinpath(d, "eig.ckpt")); r = _checked_load(joinpath(d, "result.ckpt"))
    rows = sort(r.rows; by=x -> x.energy)
    [x.energy for x in rows] == e.energy || error("eig/result energy order differs in $d")
    (E=e.energy, V=vectors ? e.vectors : nothing, C2=[x.c2 for x in rows], dim=e.dim, parity=e.parity,
     sha=_ckpt_sha(joinpath(d, "eig.ckpt")), max_h=maximum((x.rh for x in rows); init=0.0), basis_sha=e.basis_sha,
     colsha=hasproperty(e, :colsha) ? e.colsha : nothing)
end

"""Target window written by extend_targets (nothing when the block is empty or unneeded)."""
function target_window(out, a, b, l; vectors=true)
    p = joinpath(out, "targets", "s$(a)_s$(b)_L$(l).ckpt")
    isdir(p) || return nothing
    t = _checked_load(p)
    isempty(t.energy) && return nothing
    (E=t.energy, V=vectors ? t.vectors : nothing, C2=t.c2, dim=t.dim, frontier=t.frontier, sha=_ckpt_sha(p),
     max_h=maximum(t.rh; init=0.0), basis_sha=t.basis_sha, status=string(t.status), parity=t.parity,
     colsha=hasproperty(t, :colsha) ? t.colsha : nothing)
end

mutable struct KKContext
    cfg::Dict{String,Any}
    out::String
    N::Int
    space::Function          # s -> SegSpace (all resident, or loaded from the split store)
    basis_sha::String
    lib::SegLibrary
    terms::Any
    workers::Int
end

function KKContext(cfg, space::Function, basis_sha)
    N = cfg["N"]; native = get(cfg, "native", 8)
    lib = SegLibrary(builder=:tiled, tile_width=get(cfg, "tile", 64), workers=get(cfg, "workers", Threads.nthreads()),
                     opmat_threads=native)
    KKContext(cfg, cfg["out"], N, space, basis_sha, lib, kk_terms(N; cache=get(cfg, "kk_cache", "")),
              get(cfg, "action", Threads.nthreads()))
end

"""Generator overlaps <T|K_p|S> and image Gram for one (pair, li -> lf) edge (checkpointed)."""
function kk_edge!(ctx::KKContext, a, b, li, lf)
    path = joinpath(ctx.out, "kk", "edges", "s$(a)_s$(b)_L$(li)_to_L$(lf).ckpt")
    S = source_window(ctx.out, a, b, li; vectors=false); T = target_window(ctx.out, a, b, lf; vectors=false)
    (S === nothing || T === nothing) && return nothing
    identity = (source_sha=S.sha, target_sha=T.sha, basis_sha=ctx.basis_sha, tcc=string(TCC_VERSION),
                generator=file_digest(joinpath(@__DIR__, "ZhouGenerator.jl")))
    old = nothing
    if isdir(path)
        old = _checked_load(path)
        (old.identity.source_sha, old.identity.target_sha, old.identity.basis_sha) ==
            (identity.source_sha, identity.target_sha, identity.basis_sha) && return old
        old.identity.basis_sha == identity.basis_sha || error("KK edge $path was computed in another segment basis")
    end
    S = source_window(ctx.out, a, b, li); T = target_window(ctx.out, a, b, lf)
    S.basis_sha == T.basis_sha == ctx.basis_sha || error("source/target computed in another segment basis")
    ss = [ctx.space(a), ctx.space(b)]
    cs = BuildCompSpace(ss, [2ctx.N, 0, 0], 2li); ct = BuildCompSpace(ss, [2ctx.N, 0, 0], 2lf)
    cs.dim == S.dim && ct.dim == T.dim || error("source/target basis dimensions differ")
    ch = get(ctx.cfg, "kk_chunk", size(S.V, 2))
    # v0.1.14: windows that grew by deflated extension keep their old vectors -> only new blocks computed
    # v0.1.16 (V15-6): exact column identities (stored by the producer, else computed here)
    SS = _colsha(S); TS = _colsha(T)
    A = old === nothing ? nothing :
        edge_arrays_incremental(ctx.lib, ss, cs, ct, ctx.terms, 2, S.V, S.E, T.V, T.E, old; workers=ctx.workers, chunk=ch, SS, TS)
    A === nothing && (A = edge_arrays(ctx.lib, ss, cs, ct, ctx.terms, 2, S.V, T.V; workers=ctx.workers, chunk=ch))
    old === nothing || _supersede!(path)
    Ms, G = A.overlaps, A.image_gram
    mask = [abs(t - s) > 1e-5 for t in T.C2, s in S.C2]
    all(norm(M .* mask) <= 1e-6 * max(1, norm(M)) for M in Ms) || error("Generator changes C2 in $path")
    e = (identity=identity, a=a, b=b, li=li, lf=lf, source_E=S.E, source_C2=S.C2, target_E=T.E, target_C2=T.C2,
         source_dim=S.dim, target_dim=T.dim, overlaps=Ms, image_gram=G, energy_atol=max(1e-8, 10(S.max_h + T.max_h)),
         target_full=T.frontier == Inf, target_frontier=T.frontier, target_status=T.status,
         source_parity=S.parity, target_parities=sort(unique(T.parity)),
         basis_overlap=li == lf ? T.V' * S.V : zeros(0, 0), source_colsha=SS, target_colsha=TS,
         reused=hasproperty(A, :reused) ? A.reused : nothing)
    _atomic_save(path, e); e
end

"""Move a superseded checkpoint aside (never deleted): path.superseded.<n>."""
function _supersede!(path)
    n = 1; while ispath("$path.superseded.$n"); n += 1; end
    mv(path, "$path.superseded.$n"); "$path.superseded.$n"
end

"""Vacuum calibration of the two generator candidates (frozen n11 rule)."""
function kk_calibration!(ctx::KKContext)
    path = joinpath(ctx.out, "kk", "calibration.ckpt")
    e = kk_edge!(ctx, 0, 0, 0, 1); e === nothing && error("Missing (0,0) L0 -> L1 calibration edge")
    J = source_window(ctx.out, 0, 1, 1; vectors=false); J === nothing && error("Missing (0,1) L1 block for the current")
    if isdir(path)
        c = _checked_load(path)
        hasproperty(c, :edge_identity) && c.edge_identity == e.identity && c.current_sha == J.sha && return c
        _supersede!(path)                                   # windows changed (v0.1.14 extension)
    end
    sing = _ids(e.source_C2, 0); length(sing) >= 2 || error("Missing S (second singlet) in (0,0) L0")
    v, s = sing[1:2]; js = _ids(J.C2, 3); isempty(js) && error("Missing current (C2=3) in (0,1) L1")
    current = js[argmin(J.E[js])]; alpha = 2 / (J.E[current] - e.source_E[v]); alpha > 0 || error("Invalid energy scale")
    delta = alpha * (e.source_E[s] - e.source_E[v]); ds = _ids(e.target_C2, 0); isempty(ds) && error("Missing descendant")
    descendant = ds[argmin(abs.(e.target_E[ds] .- e.source_E[s] .- 1 / alpha))]
    amplitudes = [M[descendant, s] for M in e.overlaps]; sgn = sign(amplitudes[argmax(abs.(amplitudes))])
    sgn != 0 || error("Zero calibration")
    D = eigen(Hermitian(e.image_gram[:, :, v])); factor = Diagonal(sqrt.(max.(D.values, 0))) * D.vectors'
    fit = fit_vacuum_generator(factor, amplitudes * sgn, sqrt(2delta))
    coefficients = fit.coefficients
    legacy = get(get(ctx.cfg, "kk", Dict()), "legacy_calibration", nothing)
    if legacy !== nothing
        norm(coefficients - legacy["coefficients"]) <= 1e-5 || error("Frozen generator coefficient regression failed")
        abs(alpha - legacy["alpha"]) <= 1e-10 || error("Frozen energy scale regression failed")
        coefficients = Float64.(legacy["coefficients"])
    end
    r = (coefficients=coefficients, alpha=alpha, E0=e.source_E[v], current_sha=J.sha, edge_identity=e.identity,
         vacuum=v, singlet=s, current=current, descendant=descendant,
         descendant_gap_error=alpha * (e.target_E[descendant] - e.source_E[s]) - 1, leakage=fit.leakage,
         calibration_error=fit.calibration_error)
    _atomic_save(path, r); r
end

"""Per-C2 K^2 classification of one source block (frozen n11 classify_edges)."""
function kk_classify(edges, S, li, coeff; threshold=0.25)
    required = li == 0 ? [1] : [li - 1, li, li + 1]
    missing = setdiff(required, [e.lf for e in edges]); result = Any[]
    for c in sort(unique(round.(Int, S.C2)))
        allids = _ids(S.C2, c)
        # v0.1.12: a parity-restricted target window (target_parity) only guards sources of a parity it contains
        parok(e, i) = !hasproperty(e, :target_parities) || e.target_parities == [0] || S.parity[i] == 0 || S.parity[i] in e.target_parities
        guarded = isempty(missing) ? [i for i in allids if all(S.E[i] < e.target_frontier - e.energy_atol && parok(e, i) for e in edges)] : Int[]
        excluded = setdiff(allids, guarded)
        if isempty(guarded)
            push!(result, (C2=c, source_ids=guarded, excluded_ids=excluded, missing_L=missing, status="incomplete",
                           K2=nothing, Q=nothing, energies=nothing)); continue
        end
        K2 = zeros(length(guarded), length(guarded)); emptychannels = Int[]
        for e in edges
            ti = _ids(e.target_C2, c); isempty(ti) && push!(emptychannels, e.lf)
            M = coeff[1] * e.overlaps[1][ti, guarded] + coeff[2] * e.overlaps[2][ti, guarded]
            K = M .* [e.target_E[t] < S.E[s] - e.energy_atol for t in ti, s in guarded]
            K2 .+= (2e.lf + 1) / (2li + 1) * (K' * K)
            for (col, s) in enumerate(guarded)            # calibrated capture never exceeds the full image
                full = dot(coeff, e.image_gram[:, :, s] * coeff); capt = sum(abs2, M[:, col])
                capt <= full + 1e-7 * max(1, full) || error("Calibrated capture exceeds image (L$(li)->L$(e.lf), C2=$c)")
            end
        end
        F = eigen(Hermitian(K2)); minimum(F.values) >= -1e-8 || error("Negative K2")
        keep = findall(<=(threshold), F.values); Q = F.vectors[:, keep]
        D = eigen(Hermitian(Q' * Diagonal(S.E[guarded]) * Q)); Q = Q * D.vectors
        residual = [norm(Diagonal(S.E[guarded]) * Q[:, i] - D.values[i] * Q[:, i]) for i in eachindex(D.values)]
        push!(result, (C2=c, source_ids=guarded, excluded_ids=excluded, missing_L=missing,
                       status=isempty(excluded) ? "finite_saved_window_complete_K_action" : "partial_source_coverage",
                       K2=K2, k2_values=F.values, Q=Q, energies=D.values, energy_mixing_residual=residual,
                       globally_guarded_empty_target_L=emptychannels, CFT_primary_certified=false))
    end
    result
end

"""SO(5) highest-weight types with Casimir c (type only, not copies) — frozen v0.0.5 so5_types."""
function so5_types(c)
    found = String[]
    for x in 0:(ceil(Int, 2 * sqrt(2 * max(c, 0))) + 2)
        l1 = x / 2; l1 * (l1 + 3) / 2 > c + 1e-8 && continue
        for y in (x % 2):2:x
            l2 = y / 2; abs((l1 * (l1 + 3) + l2 * (l2 + 1)) / 2 - c) < 1e-8 || continue
            a = (x - y) ÷ 2; b = y; dim = (a + 1) * (b + 1) * (a + b + 2) * (2a + b + 3) ÷ 6
            push!(found, "B2[$a;$b]_dim$(dim)")
        end
    end
    found
end

"""Reduced Hermitian conjugacy of a forward/reverse edge pair (frozen v0.0.5 reverse_check).
Eligible only when both target windows are the run's own windows (same eigenvector gauge)."""
function kk_conjugacy(fwd, rev, factor)
    (fwd.target_status == "reused_run_window" && rev.target_status == "reused_run_window") ||
        return Dict{String,Any}("status" => "ineligible_different_target_eigenbasis_payloads", "factor" => factor)
    ri = [findfirst(==(E), rev.target_E) for E in fwd.source_E]; fi = [findfirst(==(E), fwd.target_E) for E in rev.source_E]
    all(!isnothing, ri) && all(!isnothing, fi) || error("Common original-index conjugacy window missing")
    errors = [norm(rev.overlaps[p][Int.(ri), :] + factor * fwd.overlaps[p][Int.(fi), :]') for p in eachindex(fwd.overlaps)]
    maximum(errors) < 1e-7 || error("Common-index reduced Hermitian conjugacy failed: $errors")
    Dict{String,Any}("status" => "passed_common_original_indices", "factor" => factor, "errors" => errors)
end

"""Same-L rotation-identity ambiguity diagnostic (frozen v0.0.5 mu_diagnostic)."""
function kk_mu(e, coeff)
    M = coeff[1] * e.overlaps[1] + coeff[2] * e.overlaps[2]; L = e.li; rot = sqrt(L * (L + 1)); B = e.basis_overlap
    mu = dot(B, M) / max(sum(abs2, B), eps()) / rot
    shifted = M - mu * rot * B
    mask = [e.target_E[t] < e.source_E[s] - e.energy_atol for t in eachindex(e.target_E), s in eachindex(e.source_E)]
    err = norm((M - shifted) .* mask)
    err < 1e-5 * max(norm(M), 1) || error("Same-L overlap mixes distinct energies beyond residual tolerance")
    Dict{String,Any}("L" => L, "fitted_mu_identity_component" => mu, "nonlowering_residual_norm" => norm(shifted .* (.!mask)),
                     "lowering_shift_error" => err, "status" => "rotation_identity_ambiguity_diagnostic_only")
end

function _csv(path, header, rows)
    io = IOBuffer(); println(io, join(header, ","))
    for row in rows; println(io, join(("\"" * replace(string(x), '"' => "\"\"") * "\"" for x in row), ",")); end
    mkpath(dirname(path)); write(path, String(take!(io)))
end

"""KK for every pair of the run: edges, calibration, classification, CSV exports."""
function run_kk_stage(cfg, spaces, basis_sha)
    # spaces: a vector of the three SegSpaces, or a SpaceStore (residency pair/minimal: <= 2 resident)
    store = spaces isa SpaceStore ? spaces : nothing
    getsp = store === nothing ? (s -> spaces[s+1]) : (s -> getspace!(store, s))
    ctx = KKContext(cfg, getsp, basis_sha); out = ctx.out; lmax = source_lmax(cfg)
    thresholds = get(get(cfg, "kk", Dict()), "thresholds", [0.05, 0.1, 0.25, 0.5])
    fit = kk_calibration!(ctx)
    pairs = sort(unique([(parse(Int, m[1]), parse(Int, m[2])) for m in
                         (match(r"^s(\d)_s(\d)_L\d$", d) for d in readdir(joinpath(out, "blocks"))) if m !== nothing]))
    kp = get(get(cfg, "kk", Dict()), "pairs", nothing)            # v0.1.12: e.g. [[0,0]] for the singlet tower
    kp === nothing || (pairs = [p for p in pairs if [p[1], p[2]] in [Int.(x) for x in kp]])
    lmax = get(get(cfg, "kk", Dict()), "lmax", lmax)
    states = Any[]; coverage = Any[]
    diag = Dict{String,Any}()
    for (a, b) in pairs
        if store !== nothing
            for s in collect(keys(store.live)); s in (a, b) || dropspace!(store, ctx.lib, s); end
        end
        for li in 0:lmax
            S = source_window(out, a, b, li; vectors=false); S === nothing && continue
            path = joinpath(out, "kk", "projections", "s$(a)_s$(b)_L$(li).ckpt")
            edges = Any[]
            for lf in (li == 0 ? [1] : [li - 1, li, li + 1])
                e = kk_edge!(ctx, a, b, li, lf); e === nothing || push!(edges, e)
            end
            if isdir(path)                                  # reuse only if built from exactly these edges
                r0 = _checked_load(path)
                (r0.source_sha == S.sha && hasproperty(r0, :edge_identities) && r0.edge_identities == [e.identity for e in edges] &&
                 r0.calibration.coefficients == fit.coefficients && r0.calibration.alpha == fit.alpha) || _supersede!(path)
            end
            r = if isdir(path)
                _checked_load(path)
            else
                tc = [(t, kk_classify(edges, S, li, fit.coefficients; threshold=t)) for t in thresholds]
                x = (a=a, b=b, L=li, source_sha=S.sha, source_E=S.E, source_C2=S.C2, calibration=fit, threshold_channels=tc,
                     edge_identities=[e.identity for e in edges])
                _atomic_save(path, x); x
            end
            for (t, chans) in r.threshold_channels, ch in chans
                types = join(so5_types(ch.C2), "|")
                push!(coverage, (a, b, li, ch.C2, t, join(ch.source_ids, ";"), join(ch.excluded_ids, ";"), join(ch.missing_L, ";"), ch.status, types))
                ch.Q === nothing && continue
                for j in axes(ch.Q, 2)
                    push!(states, ("s$(a)_s$(b)", li, ch.C2, j, t, ch.energies[j], fit.alpha * (ch.energies[j] - fit.E0),
                                   dot(ch.Q[:, j], ch.K2 * ch.Q[:, j]), ch.status, types))
                end
            end
        end
        # reduced conjugacy (L0<->L1: sqrt(3), L1<->L2: sqrt(5/3)) and same-L mu diagnostic
        edge_at = (li, lf) -> (p = joinpath(out, "kk", "edges", "s$(a)_s$(b)_L$(li)_to_L$(lf).ckpt"); isdir(p) ? _checked_load(p) : nothing)
        for (li, lf, fac) in ((0, 1, sqrt(3)), (1, 2, sqrt(5 / 3)))
            f = edge_at(li, lf); r = edge_at(lf, li)
            (f === nothing || r === nothing) && continue
            diag["s$(a)_s$(b)_L$(li)_L$(lf)_conjugacy"] = kk_conjugacy(f, r, fac)
        end
        e11 = edge_at(1, 1); e11 === nothing || (diag["s$(a)_s$(b)_L1_mu"] = kk_mu(e11, fit.coefficients))
        store === nothing && for s in unique((a, b)); release!(ctx.lib, getsp(s)); end
    end
    store === nothing || for s in collect(keys(store.live)); dropspace!(store, ctx.lib, s); end
    open(io -> TOML.print(io, diag), joinpath(out, "kk", "diagnostics.toml"), "w")
    _csv(joinpath(out, "kk", "projected_states.csv"), ["pair", "L", "C2", "mode", "threshold", "energy", "delta", "K2", "status", "SO5_irrep_types"], states)
    _csv(joinpath(out, "kk", "channel_coverage.csv"), ["sA", "sB", "L", "C2", "threshold", "source_ids", "excluded_ids", "missing_L", "status", "SO5_irrep_types"], coverage)
    (calibration=fit, nstates=length(states), diagnostics=diag)
end
