# Coupling tuning stage: safeguarded root search for F(g) = 0.
#
#   F(g) = 2 (E_stress - E_0) / (E_current - E_0) - 3
#   ground  = lowest C2 = 0 state of (0,0,L0)     current = lowest C2 = 3 state of (0,1,L1)
#   stress  = lowest C2 = 0 state of (0,0,L2)
# H(g) = H_den - g H_pair is assembled once per target block (g-independent geometry and
# segment operators from the shared library); every g only rewrites channel coefficients.
# Diagonal (0,0) targets may be solved in the exchange-even sector (the singlet C2 = 0
# lies in X = +1); the selected state is always gated on its C2 label.
# Acceptance rules are the frozen ones: safeguarded bracket/secant search inside bounds,
# |F| <= ftol AND bracket width <= xtol, >= 0.95 overlap of every anchor with the closest
# distinct g, and an independently seeded final solve that again satisfies |F| <= ftol.

const TUNE_TARGETS = ((name="ground", sa=0, sb=0, l=0, c2=0.0),
                      (name="current", sa=0, sb=1, l=1, c2=3.0),
                      (name="stress", sa=0, sb=0, l=2, c2=0.0))

mutable struct TuneBlock
    t::Any
    H::Any
    C::Any
    fH::Vector{Float64}
    dim::Int
    sector::Any          # ExchangeSector (X = +1) or nothing
end

"""Pure safeguarded root controller (bracket/secant search inside bounds)."""
function find_root(evaluate, start; bounds=(0.90, 1.12), maxpoints=16, ftol=1e-5, xtol=1e-4)
    lo, hi = bounds; points = Dict{Float64,Float64}()
    function record(g)
        g = Float64(g); lo <= g <= hi || error("g outside authorized bounds")
        if !haskey(points, g)
            length(points) >= maxpoints && error("Distinct-g point budget exhausted")
            f = Float64(evaluate(g)); isfinite(f) || error("Nonfinite root residual")
            points[g] = f
        end
        points[g]
    end
    sgnchange(a, b) = points[a] == 0 || points[b] == 0 || (points[a] < 0) != (points[b] < 0)
    brackets() = (gs = sort(collect(keys(points))); [(a, b) for (a, b) in zip(gs, gs[2:end]) if sgnchange(a, b)])
    record(start)
    trials = Float64[]
    abs(points[start]) <= ftol && append!(trials, (start - 0.49xtol, start + 0.49xtol))
    for step in (0.005, 0.015, 0.035, 0.075); append!(trials, (start - step, start + step)); end
    append!(trials, (lo, hi))
    for g in trials
        isempty(brackets()) || break
        record(min(hi, max(lo, g)))
    end
    opts = brackets(); isempty(opts) && error("No signed bracket within bounds and point budget")
    a, b = opts[argmin([y - x for (x, y) in opts])]
    best = start
    while true
        inside = [g for g in keys(points) if a <= g <= b]
        best = inside[argmin([(abs(points[g]), g) for g in inside])]
        abs(points[best]) <= ftol && b - a <= xtol && break
        length(points) >= maxpoints && error("Point budget exhausted before F AND width tolerances")
        if abs(points[best]) <= ftol && b - a > xtol && length(points) <= maxpoints - 2
            left = max(a, best - 0.49xtol); right = min(b, best + 0.49xtol)
            record(left); record(right)
            if sgnchange(left, right); a, b = left, right; continue; end
        end
        fa, fb = points[a], points[b]
        trial = fb != fa ? (a * fb - b * fa) / (fb - fa) : (a + b) / 2
        margin = 1e-6 * (b - a); trial = min(b - margin, max(a + margin, trial))
        (haskey(points, trial) || !(a < trial < b)) && (trial = (a + b) / 2)
        fp = record(trial)
        (fa == 0 || fp == 0 || (fa < 0) != (fp < 0)) ? (b = trial) : (a = trial)
    end
    others = [g for g in keys(points) if g != best]; isempty(others) && error("No distinct-g continuity point")
    neighbor = others[argmin([(abs(g - best), g) for g in others])]
    (g=best, F=points[best], bracket=[a, b], bracket_width=b - a, neighbor_g=neighbor,
     physical_points=length(points), points=sort([(g=g, F=f) for (g, f) in points]; by=x -> x.g))
end

"""Assemble the three target blocks once (g-independent)."""
function tune_blocks(cfg, spaces, tpl)
    N = cfg["N"]; native = get(cfg, "native", 8); act = get(cfg, "action", Threads.nthreads())
    lib = SegLibrary(builder=:tiled, tile_width=get(cfg, "tile", 64), workers=get(cfg, "workers", Threads.nthreads()),
                     opmat_threads=native)
    useX = get(get(cfg, "tune", Dict()), "exchange", true)
    blocks = Dict{String,TuneBlock}()
    getsp(s) = spaces isa SpaceStore ? getspace!(spaces, s) : spaces[s+1]
    for t in TUNE_TARGETS
        ss = [getsp(t.sa), getsp(t.sb)]
        sH, fH = segops_shared!(lib, ss, tpl.template); sC, fC = segops_shared!(lib, ss, tpl.c2)
        cp = BuildCompSpace(ss, [2N, 0, 0], 2t.l)
        H = BuildCompOperator(cp, tpl.template, sH; num_th=act)
        C = absorb!(BuildCompOperator(cp, tpl.c2, sC; num_th=act), fC)
        sec = (useX && t.sa == t.sb) ? (p = exchange_map(cp); ExchangeSector(p[1], p[2], 1)) : nothing
        blocks[t.name] = TuneBlock(t, H, C, fH, cp.dim, sec)
    end
    blocks
end

set_tune_g!(b::TuneBlock, tpl, g) = absorb!(set_g!(b.H, tpl, g), b.fH)

"""Lowest target state of one block at the current g (frozen window/C2 gates)."""
function tune_state(b::TuneBlock, seed::Int; act=Threads.nthreads(), tol=1e-9)
    n = b.sector === nothing ? b.dim : length(b.sector)
    f = b.sector === nothing ? (v -> *(b.H, v; num_th=act)) : reduced_action(b.H, b.sector; num_th=act)
    for k in unique(min.([8, 12, 24], n))
        E, V, info = lowest(f, n, k, seed; tol)
        W = reduce(hcat, [b.sector === nothing ? v : sector_expand!(zeros(b.dim), b.sector, v) for v in V])
        rh = post_h(b.H, W, E; workers=act); C, rc = post_c(b.C, W; workers=act)
        orth = norm(W' * W - I)
        maximum(rh) <= 1e-7 && orth <= 1e-7 || error("tuning: H residual/orthogonality gate failed ($(b.t.name))")
        ids = findall(i -> abs(C[i] - b.t.c2) <= 1e-5 && rc[i] <= 1e-6, eachindex(E))
        gap = length(E) > 1 ? E[end] - E[end-1] : Inf
        chosen = isempty(ids) ? 0 : first(ids)
        chosen > 0 && !all(i -> rc[i] <= 1e-6 && abs(C[i] - round(C[i])) <= 1e-5, 1:chosen) &&
            error("tuning: unresolved lower C2 may hide target $(b.t.name)")
        if chosen > 0 && (chosen < length(E) || length(E) == n) && (gap > 1e-6 || length(E) == n)
            return (energy=E[chosen], vector=W[:, chosen], c2=C[chosen], rh=rh[chosen], rc=rc[chosen], index=chosen,
                    requested=k, returned=length(E), energies=E, c2_values=C, seed=seed, sector_dim=n)
        end
    end
    error("tuning: target/window coverage failed at k24 ($(b.t.name))")
end

"""
    run_tune_stage(cfg, spaces) -> result NamedTuple, writes out/tune/selected_g.toml

cfg["tune"]: start (default run g or 0.9753477137817606), bounds ([0.90, 1.12]), maxpoints (16),
ftol (1e-5), xtol (1e-4), exchange (true).  Point checkpoints: out/tune/points/<g bits>/<target>.ckpt.
"""
function run_tune_stage(cfg, spaces)
    out = joinpath(cfg["out"], "tune"); mkpath(out)
    tc = get(cfg, "tune", Dict()); tpl = hamiltonian_template(cfg["N"])
    act = get(cfg, "action", Threads.nthreads()); st = Stage(joinpath(out, "stages.tsv"))
    blocks = stage!(st, "tune_assemble") do; tune_blocks(cfg, spaces, tpl); end
    start = Float64(get(tc, "start", cfg["g"] isa Real ? cfg["g"] : 0.9753477137817606))
    bounds = Tuple(Float64.(get(tc, "bounds", [0.90, 1.12])))
    states = Dict{Tuple{Float64,String},Any}()
    function point!(g, revision="main")
        res = Dict{String,Any}()
        for t in TUNE_TARGETS
            key = (g, t.name * ":" * revision)
            haskey(states, key) && (res[t.name] = states[key]; continue)
            p = joinpath(out, "points", bitstring(g), revision, t.name * ".ckpt")
            s = if isdir(p)
                _checked_load(p)
            else
                for b in values(blocks); set_tune_g!(b, tpl, g); end
                off = revision == "main" ? 0 : 7919
                x = stage!(st, "tune_$(t.name)_$(revision)"; detail="g=$g") do
                    tune_state(blocks[t.name], 5052026 + 100t.sa + 10t.sb + t.l + off; act)
                end
                _atomic_save(p, merge(x, (g=g, target=t.name, revision=revision))); x
            end
            states[key] = s; res[t.name] = s
        end
        F = 2 * (res["stress"].energy - res["ground"].energy) / (res["current"].energy - res["ground"].energy) - 3
        println("TUNE g=$g F=$F revision=$revision"); flush(stdout)
        (F=F, states=res)
    end
    r = find_root(g -> point!(g).F, start; bounds, maxpoints=get(tc, "maxpoints", 16),
                  ftol=get(tc, "ftol", 1e-5), xtol=get(tc, "xtol", 1e-4))
    best = point!(r.g); nb = point!(r.neighbor_g)
    overl = Dict(t.name => abs(dot(best.states[t.name].vector, nb.states[t.name].vector)) for t in TUNE_TARGETS)
    all(v -> v >= 0.95, values(overl)) || error("tuning: anchor continuity overlap below 0.95: $overl")
    fin = point!(r.g, "final")
    abs(fin.F) <= get(tc, "ftol", 1e-5) || error("tuning: independent final solve misses the root tolerance")
    de = maximum(abs(fin.states[t.name].energy - best.states[t.name].energy) for t in TUNE_TARGETS)
    de <= 1e-7 || error("tuning: independent final energies differ by $de")
    sel = Dict{String,Any}("g" => r.g, "F" => fin.F, "search_F" => r.F, "bracket" => r.bracket, "bracket_width" => r.bracket_width,
        "neighbor_g" => r.neighbor_g, "continuity_overlaps" => overl, "final_energy_difference" => de,
        "points" => [Dict("g" => p.g, "F" => p.F) for p in r.points], "physical_points" => r.physical_points,
        "energies" => Dict(t.name => fin.states[t.name].energy for t in TUNE_TARGETS),
        "exchange_even_sector" => get(tc, "exchange", true), "N" => cfg["N"], "tcc_version" => string(TCC_VERSION),
        "claim" => "finite-N stress/current-gap matching only; not a CFT critical point",
        "ground_scope" => "lowest C2=0 within s0_s0_L0; not certified global ground over all sectors")
    open(io -> TOML.print(io, sel), joinpath(out, "selected_g.toml"), "w")
    (g=r.g, F=fin.F, selection=sel)
end
