# v0.1.17: particle-hole half construction of the segment spaces (src/ph.jl) versus a direct build.
#   1. same sectors / l ranges / pointers / configurations; gauge-invariant multiplet projectors equal
#   2. full 18-block spectrum (both exchange parities) and C2 labels equal
#   3. KK / Gamma^2 edges on (0,0): |matrix elements| and image Gram equal
#   4. (N <= 6) full pipeline basis -> spectrum -> targets -> kk -> ope -> summary with ph_half = true
#      vs false: KK projected states and OPE numbers equal
# usage: julia --threads=T t13_ph.jl N out [projectors|spectrum|all|pipeline] [nev] [blocks]
#   blocks: "all" (default) or "a,b,L;a,b,L" for the spectrum check (must contain 0,0,0..2 for the edges)
#   pipeline: checks 1 and 4 at any N (4 is part of "all" only for N <= 6)
#   env TCC_T13_DIRECT=<segment checkpoint>: use an existing direct build (e.g. the frozen N12 basis)
#   env TCC_T13_MAXNEV: targets.max_nev of the pipeline check (default 120 for N <= 6, else 1200)
include(get(ENV, "TCC_SRC", joinpath(@__DIR__, "..", "src", "Tcc.jl")))
using .Tcc, FuzzifiED, FuzzifiED.SO3lver, LinearAlgebra, SparseArrays, Serialization, Printf, TOML
FuzzifiED.SilentStd = true
N = parse(Int, ARGS[1]); out = ARGS[2]; mode = length(ARGS) >= 3 ? ARGS[3] : "all"
nev = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 12
bsel = length(ARGS) >= 5 ? ARGS[5] : "all"
direct_ckpt = get(ENV, "TCC_T13_DIRECT", "")
mkpath(out); FuzzifiED.NumThreads = min(8, Threads.nthreads())
fails = String[]; check(ok, msg) = (ok || push!(fails, msg); println(ok ? "PASS " : "FAIL ", msg); flush(stdout))

t_full = @elapsed orig = isempty(direct_ckpt) ? build_segment_spaces(N) : load_segment_spaces(direct_ckpt).spaces
isempty(direct_ckpt) || println("direct = existing checkpoint $direct_ckpt (time below is the load)")
t_ph = @elapsed (h = ph_build_half(N); ph = ph_expand(h))          # = build_segment_spaces(N; ph=true)
b_full = Base.summarysize(orig); b_half = Base.summarysize(h.spaces); h = nothing
@printf("N=%d build direct %.2f s | PH half+expand %.2f s | stored multiplet data %.1f%% of direct\n", N, t_full, t_ph, 100b_half / b_full)

# 1. structure and projectors per (space, sector, l)
# Gauge-invariant without dense projectors (feasible at N = 11): with U = A'B (the multiplet gauge),
# B = A U and B1 = A1 U must hold with U orthogonal, i.e. both builds span the same multiplets and
# the L+ images transform with the same U.
tallnorm(E) = sqrt(max(0.0, opnorm(Symmetric(E' * E))))         # 2-norm of a tall matrix via its m x m Gram
let wp = 0.0, w1 = 0.0, wu = 0.0, wo = 0.0, same = true
    for (a, b) in zip(orig, ph)
        same &= a.sec == b.sec && a.l_rng == b.l_rng && a.ptr_st == b.ptr_st
        for i in axes(a.sec, 2)
            same &= a.bs[i].cfs.conf == b.bs[i].cfs.conf
            for il in eachindex(a.l_rng[i])
                r = (a.ptr_st[i][il]:a.ptr_st[i][il+1]-1) .- (a.ptr_st[i][1] - 1)
                isempty(r) && continue
                A = a.sts[i][:, r]; B = b.sts[i][:, r]; U = A' * B
                wo = max(wo, opnorm(A' * A - I), opnorm(B' * B - I))
                wu = max(wu, opnorm(U' * U - I))
                wp = max(wp, tallnorm(B - A * U))
                if isassigned(a.sts1, i)
                    w1 = max(w1, tallnorm(b.sts1[i][:, r] - a.sts1[i][:, r] * U))
                end
            end
        end
    end
    check(same, "sector list, l ranges, pointers and configurations identical")
    check(wo < 1e-10, @sprintf("multiplet states orthonormal in both builds: %.2e", wo))
    check(wu < 1e-10 && wp < 1e-10, @sprintf("same multiplet subspaces (sts): |U'U-I| %.2e, |B-AU| %.2e", wu, wp))
    check(w1 < 1e-9, @sprintf("L+ images (sts1) with the same gauge U: %.2e", w1))
end

if mode in ("spectrum", "all")
    save(dir, sp) = (isdir(dir) || Tcc._atomic_save(dir, (meta=(N=N, ph=dir), spaces=sp)); dir)
    blocks = bsel == "all" ? [[a, b, L] for a in 0:2 for b in a:2 for L in 0:2] :
             [parse.(Int, split(x, ",")) for x in split(bsel, ";")]
    res = Dict{String,Any}()
    for (nm, sp) in (("direct", orig), ("ph", ph))
        d = joinpath(out, "spec_$nm"); rm(d; recursive=true, force=true)
        t = @elapsed (res[nm] = run_spectrum(Dict{String,Any}("N"=>N, "g"=>0.95, "nev"=>nev, "out"=>d, "blocks"=>blocks, "exchange"=>true,
            "segment_checkpoint"=>(nm == "direct" && !isempty(direct_ckpt) ? direct_ckpt : save(joinpath(out, "seg_$nm"), sp)), "native"=>FuzzifiED.NumThreads,
            "action"=>Threads.nthreads(), "tile"=>16, "tol"=>1e-11)).rows)
        @printf("spectrum %s: %d states, %.1f s\n", nm, length(res[nm]), t)
    end
    key(x) = (x.sa, x.sb, x.L, x.x)
    let de = 0.0, dc = 0.0, ncmp = 0
        for k in unique(key.(res["direct"]))
            A = sort([x for x in res["direct"] if key(x) == k]; by=x -> x.energy)
            B = sort([x for x in res["ph"] if key(x) == k]; by=x -> x.energy)
            length(A) == length(B) || (push!(fails, "state count $k"); continue)
            for i in eachindex(A)
                de = max(de, abs(A[i].energy - B[i].energy)); dc = max(dc, abs(A[i].c2 - B[i].c2)); ncmp += 1
            end
        end
        check(de < 1e-9 && dc < 1e-8, @sprintf("spectrum (%d blocks, both exchange parities, %d states) PH vs direct: dE %.2e, dC2 %.2e", length(blocks), ncmp, de, dc))
    end
    # 3. KK / Gamma^2 edges on (0,0): gauge-invariant |<T|O|S>| and image Gram
    function edges(nm, sp)
        ss = [sp[1], sp[1]]; cps = Dict(L => BuildCompSpace(ss, [2N, 0, 0], 2L) for L in 0:2)
        V(L) = open(deserialize, joinpath(out, "spec_$nm", "blocks", "s0_s0_L$L", "eig.ckpt", "payload.jls")).vectors
        lib = SegLibrary(builder=:tiled, tile_width=16, workers=Threads.nthreads(), opmat_threads=FuzzifiED.NumThreads)
        kk = kk_terms(N)
        Dict(e => edge_arrays(lib, ss, cps[li], cps[lf], t, 2r, V(li), V(lf); workers=Threads.nthreads())
             for (e, t, r, li, lf) in (("kk_12", kk, 1, 1, 2), ("kk_22", kk, 1, 2, 2), ("g2_02", gamma2_terms(N, 2), 2, 0, 2)))
    end
    let E1 = edges("direct", orig), E2 = edges("ph", ph), we = 0.0
        for k in keys(E1)
            A = E1[k]; B = E2[k]
            for (x, y) in zip(A.overlaps, B.overlaps)
                r = 1:min(size(x, 1), size(y, 1)); c = 1:min(size(x, 2), size(y, 2))
                we = max(we, maximum(abs.(abs.(x[r, c]) - abs.(y[r, c]))))
            end
            n = min(size(A.image_gram, 3), size(B.image_gram, 3))
            we = max(we, maximum(abs.(abs.(A.image_gram[:, :, 1:n]) - abs.(B.image_gram[:, :, 1:n]))))
        end
        check(we < 1e-9, @sprintf("KK/Gamma2 edges PH vs direct (|M|, Gram): %.2e", we))
    end
end

if (mode == "all" && N <= 6) || mode == "pipeline"
    # 4. production path: the basis stage builds the segment spaces itself
    rows(path, cols) = sort([Tuple(split(l, ',')[cols]) for l in readlines(path)[2:end]])
    num(x) = parse(Float64, strip(x, '"'))
    outs = Dict{Bool,String}()
    for flag in (false, true)
        d = joinpath(out, "pipe_ph$(flag)"); rm(d; recursive=true, force=true); outs[flag] = d
        run_pipeline(Dict("run" => Dict("N" => N, "g" => 0.95, "out" => d), "basis" => Dict("ph_half" => flag),
                          "spectrum" => Dict{String,Any}("nev" => 8, "exchange" => true, "tol" => 1e-11),
                          "threads" => Dict("native" => FuzzifiED.NumThreads, "tile" => 16),
                          "targets" => Dict("initial_nev" => 12, "max_nev" => parse(Int, get(ENV, "TCC_T13_MAXNEV", N <= 6 ? "120" : "1200"))),
                          "kk" => Dict("cache" => joinpath(out, "kk_cache")), "ope" => Dict("half" => false)))
    end
    m1 = Tcc._checked_load(joinpath(outs[true], "segment_spaces.ckpt")).meta
    A = rows(joinpath(outs[false], "kk", "projected_states.csv"), [1, 2, 3, 5, 6, 8])
    B = rows(joinpath(outs[true], "kk", "projected_states.csv"), [1, 2, 3, 5, 6, 8])
    same = length(A) == length(B) && all(zip(A, B)) do (a, b)
        a[1:4] == b[1:4] && abs(num(a[5]) - num(b[5])) < 1e-8 && abs(num(a[6]) - num(b[6])) < 1e-7 * max(1, abs(num(a[6])))
    end
    oA = TOML.parsefile(joinpath(outs[false], "ope", "ope.toml")); oB = TOML.parsefile(joinpath(outs[true], "ope", "ope.toml"))
    ks = [k for k in keys(oA) if oA[k] isa Real && (startswith(k, "raw:") || startswith(k, "f_phi_phi"))]
    dope = maximum(abs(oA[k] - oB[k]) for k in ks)
    check(m1.ph_half == true && same && dope < 1e-8,
          @sprintf("pipeline with ph_half: meta flag set, %d KK projected states equal, OPE (%d numbers) max diff %.1e", length(A), length(ks), dope))
end
println(isempty(fails) ? "PH_PASS N=$N" : "PH_FAIL N=$N: $(join(fails, "; "))")
