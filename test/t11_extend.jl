# v0.1.14 deflated window extension: an existing run extended to larger windows must equal a fresh
# run with the larger windows, with the stored vectors kept bit-identical and only the new parts
# (eigenstates, edge blocks) computed.
# usage: julia --threads=T t11_extend.jl N out
include(joinpath(@__DIR__, "..", "src", "Tcc.jl"))
using .Tcc, FuzzifiED, FuzzifiED.SO3lver, TOML, Printf, LinearAlgebra, SparseArrays, Random
FuzzifiED.SilentStd = true
N = parse(Int, ARGS[1]); root = ARGS[2]; g = 0.95
mkpath(root)
ok = true
check(c, msg) = (println(c ? "PASS " : "FAIL ", msg); global ok &= c)

# ---- 1. lowest_deflated on a sparse symmetric matrix (Lanczos path, n > 400) ---------------------
let n = 3000, rng = MersenneTwister(7)
    A = sprandn(rng, n, n, 4 / n); A = A + A' + spdiagm(0 => collect(range(0, 30; length=n)) .+ 0.01 .* randn(rng, n))
    f = v -> A * v
    Ef, Vf, _ = Tcc.lowest(f, n, 40, 11; tol=1e-12)
    E0, V0, _ = Tcc.lowest(f, n, 16, 12; tol=1e-12)
    E1, V1, _ = lowest_deflated(f, n, 24, reduce(hcat, V0), 13; tol=1e-12, Eref=E0)
    E = sort(vcat(E0, E1)); W = reduce(hcat, vcat(V0, V1))
    de = maximum(abs.(E - Ef)); orth = norm(W' * W - I); res = maximum(norm(A * W[:, j] - vcat(E0, E1)[j] * W[:, j]) for j in axes(W, 2))
    check(de < 1e-9 && orth < 1e-9 && res < 1e-8, @sprintf("lowest_deflated 16+24 == lowest 40 (n=%d): dE %.1e, orth %.1e, residual %.1e", n, de, orth, res))
end

# ---- 2. pipeline: small windows, then extend; compare with a fresh large-window run ---------------
th = Dict("native" => min(8, Threads.nthreads()), "tile" => 16)
function conf(out, nev, ext)
    Dict("run" => Dict("N" => N, "g" => g, "out" => out),
         "spectrum" => Dict{String,Any}("nev" => nev, "exchange" => true, "tol" => 1e-11, "extend" => ext),
         "threads" => th, "targets" => Dict("initial_nev" => 12, "max_nev" => 120),
         "kk" => Dict("cache" => joinpath(root, "kk_cache")), "ope" => Dict("half" => false))
end
dS = joinpath(root, "ext_N$N"); dF = joinpath(root, "fresh_N$N")
rm(dS; recursive=true, force=true); rm(dF; recursive=true, force=true)
nev0, nev1 = 8, 16
t0 = time(); run_pipeline(conf(dS, nev0, false)); wsmall = time() - t0
small = Dict(d => Tcc._checked_load(joinpath(dS, "blocks", d, "eig.ckpt")) for d in readdir(joinpath(dS, "blocks")) if isdir(joinpath(dS, "blocks", d, "eig.ckpt")))
refused = try; run_pipeline(conf(dS, nev1, false)); false; catch err; occursin("extend", sprint(showerror, err)); end
check(refused, "larger nev without spectrum.extend is refused (with a hint)")
t0 = time(); mE = run_pipeline(conf(dS, nev1, true)); wext = time() - t0
t0 = time(); mF = run_pipeline(conf(dF, nev1, false)); wfresh = time() - t0
@printf("wall: small %.1f s, extension %.1f s, fresh large %.1f s\n", wsmall, wext, wfresh)
check(all(get(mE["stages"][s], "status", "") == "done" for s in Tcc.STAGES if s != "tune") && haskey(mE, "window_history"),
      "extension: every stage done, window history recorded")

# 2a. spectrum: old columns bit-identical, new windows equal to the fresh run, superseded checkpoints kept
let bitok = true, de = 0.0, dc = 0.0, nnew = 0, nsup = 0
    for (d, e0) in small
        e1 = Tcc._checked_load(joinpath(dS, "blocks", d, "eig.ckpt"))
        for j in eachindex(e0.energy)
            i = findfirst(==(e0.energy[j]), e1.energy)
            bitok &= i !== nothing && e1.vectors[:, i] == e0.vectors[:, j] && e1.parity[i] == e0.parity[j]
        end
        nnew += length(e1.energy) - length(e0.energy)
        length(e1.energy) > length(e0.energy) && (nsup += isdir(joinpath(dS, "blocks", d, "eig.k$(length(e0.energy)).ckpt")))
        eF = Tcc._checked_load(joinpath(dF, "blocks", d, "eig.ckpt"))
        for s in unique(eF.parity)
            a = e1.energy[e1.parity .== s]; b = eF.energy[eF.parity .== s]
            length(a) == length(b) || (bitok = false; println("  $d x=$s: extended $(length(a)) vs fresh $(length(b))"); continue)
            de = max(de, maximum(abs.(a - b); init=0.0))
        end
        r1 = Tcc._checked_load(joinpath(dS, "blocks", d, "result.ckpt")).rows; rF = Tcc._checked_load(joinpath(dF, "blocks", d, "result.ckpt")).rows
        dc = max(dc, maximum(abs.(sort([x.c2 for x in r1]) - sort([x.c2 for x in rF])); init=0.0))
        dc = max(dc, maximum((x.rh for x in r1); init=0.0) > 1e-6 ? Inf : 0.0)
    end
    check(bitok && de < 1e-9 && dc < 1e-6 && nnew > 0 && nsup > 0,
          @sprintf("spectrum extension: old vectors bit-identical, %d new states, windows == fresh (dE %.1e, dC2 %.1e), %d superseded eig kept", nnew, de, dc, nsup))
end

# 2b. targets: complete for the larger sources, equal to the fresh run below both frontiers
let okt = true, dt = 0.0
    for f in filter(f -> occursin(r"^s\d_s\d_L\d\.ckpt$", f), readdir(joinpath(dS, "targets")))
        t1 = Tcc._checked_load(joinpath(dS, "targets", f)); tF = Tcc._checked_load(joinpath(dF, "targets", f))
        okt &= t1.required_energy < t1.frontier
        lim = min(t1.frontier, tF.frontier) - 1e-8
        A = sort(t1.energy[t1.energy .<= lim]); B = sort(tF.energy[tF.energy .<= lim])
        length(A) == length(B) || (okt = false; println("  target $f: $(length(A)) vs $(length(B))"); continue)
        dt = max(dt, maximum(abs.(A - B); init=0.0))
    end
    check(okt && dt < 1e-9, @sprintf("target windows after extension complete and == fresh below frontier (dE %.1e)", dt))
end

# 2c. KK / OPE edges: incremental arrays == a full recomputation on the extended windows
segs = Tcc._checked_load(joinpath(dS, "segment_spaces.ckpt")).spaces
lib = SegLibrary(builder=:tiled, tile_width=16, workers=Threads.nthreads(), opmat_threads=th["native"])
terms = kk_terms(N; cache=joinpath(root, "kk_cache"))
let nre = 0, nedge = 0, derr = 0.0
    for f in filter(f -> endswith(f, ".ckpt"), readdir(joinpath(dS, "kk", "edges")))
        e = Tcc._checked_load(joinpath(dS, "kk", "edges", f)); nedge += 1
        (hasproperty(e, :reused) && e.reused !== nothing) || continue
        nre += 1
        S = source_window(dS, e.a, e.b, e.li); T = target_window(dS, e.a, e.b, e.lf); ss = [segs[e.a+1], segs[e.b+1]]
        cs = BuildCompSpace(ss, [2N, 0, 0], 2e.li); ct = BuildCompSpace(ss, [2N, 0, 0], 2e.lf)
        A = edge_arrays(lib, ss, cs, ct, terms, 2, S.V, T.V)
        derr = max(derr, maximum(norm(A.overlaps[p] - e.overlaps[p]) for p in eachindex(A.overlaps)), norm(A.image_gram - e.image_gram))
    end
    check(nre > 0 && derr < 1e-10, @sprintf("KK edges: %d of %d re-used old blocks, == full recomputation (%.1e)", nre, nedge, derr))
end
let nre = 0, derr = 0.0
    for f in filter(f -> endswith(f, ".ckpt"), readdir(joinpath(dS, "ope", "edges")))
        e = Tcc._checked_load(joinpath(dS, "ope", "edges", f))
        (hasproperty(e, :reused) && e.reused !== nothing) || continue
        nre += 1
        S = source_window(dS, 0, 0, e.li); T = source_window(dS, 0, 0, e.lf); ss = [segs[1], segs[1]]
        cs = BuildCompSpace(ss, [2N, 0, 0], 2e.li); ct = BuildCompSpace(ss, [2N, 0, 0], 2e.lf)
        A = edge_arrays(lib, ss, cs, ct, gamma2_terms(N, e.rank), 2e.rank, S.V, T.V)
        derr = max(derr, norm(A.overlaps[1] - e.overlaps[1]), norm(A.image_gram - e.image_gram))
    end
    check(nre > 0 && derr < 1e-10, @sprintf("OPE Gamma^2 edges: %d incremental, == full recomputation (%.1e)", nre, derr))
end

# 2d. physics: KK projected states and phi x phi towers equal to the fresh large-window run
num(x) = (v = tryparse(Float64, x); v === nothing ? x : v)
rkey(r) = Tuple(u isa Float64 ? round(u; digits=6) : u for u in r)
function csvrows(path, cols)
    L = readlines(path)[2:end]
    sort([Tuple(num(String(strip(x, '"'))) for x in split(l, ',')[cols]) for l in L]; by=rkey)
end
function rows_close(A, B; tol=1e-7)
    length(A) == length(B) || return false
    all(zip(A, B)) do (a, b)
        all(zip(a, b)) do (u, v)
            u == v && return true
            pu = u; pv = v; pu isa Float64 && pv isa Float64 && abs(pu - pv) <= tol * max(1, abs(pu))
        end
    end
end
kE = csvrows(joinpath(dS, "kk", "projected_states.csv"), [1, 2, 3, 5, 6, 8]); kF = csvrows(joinpath(dF, "kk", "projected_states.csv"), [1, 2, 3, 5, 6, 8])
check(rows_close(kE, kF), "KK projected states (pair, L, C2, threshold, energy, K2): extended == fresh ($(length(kE)) rows)")
tE = csvrows(joinpath(dS, "ope", "phi_phi_towers.csv"), [1, 4, 5, 9]); tF = csvrows(joinpath(dF, "ope", "phi_phi_towers.csv"), [1, 4, 5, 9])
check(rows_close(tE, tF), "OPE phi x phi towers (basis, C2, energy, |f|): extended == fresh ($(length(tE)) rows)")
oE = TOML.parsefile(joinpath(dS, "ope", "ope.toml")); oF = TOML.parsefile(joinpath(dF, "ope", "ope.toml"))
ks = [k for k in keys(oF) if oF[k] isa Real && startswith(k, "raw:")]
check(!isempty(ks) && all(abs(oE[k] - oF[k]) < 1e-8 for k in ks), "OPE raw anchors equal ($(join(ks, ", ")))")

# 2e. a second pass reruns nothing and computes nothing
m2 = run_pipeline(conf(dS, nev1, true))
check(all(m2["stages"][s]["finished"] == mE["stages"][s]["finished"] for s in keys(mE["stages"])), "rerun of the extended run skips every stage")
println(ok ? "EXTEND_PASS N=$N" : "EXTEND_FAIL N=$N")
