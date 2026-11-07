# blind_pd_pipeline v0.1.0 — cluster-side exporter (read-only on production data)
#
# Reads a Tcc run directory (run.out of the pipeline: blocks/, kk/, targets/) and writes portable files:
#   <out>/meta.json             calibration, blocks (result rows + per-state parity), edge index,
#                               production projections (all thresholds) for regression gates
#   <out>/edges/<key>.f64       raw little-endian Float64: O1 (nt x ns), O2 (nt x ns), G (2 x 2 x ns), column-major
#
# Usage: julia --startup-file=no export_kk.jl <run_dir> <out_dir>
# Only Base + Serialization + SHA are used. Nothing in <run_dir> is modified.

using Serialization, SHA, Printf

const EXPORTER_VERSION = "blind_pd_pipeline/export_kk v0.3.6"
# v0.3.5: a (pair, L) whose KK sources are a saved target window (kkext, a run whose KK sources are saved target windows) has no blocks/<key>;
# its entry is taken from the source window of its outgoing edges ("source_window" = "target", "x" = null, filled
# from parity.csv by the reader).  Per-column eigenvector SHAs of both edge sides are exported for the gauge gate.

load(p) = deserialize(joinpath(p, "payload.jls"))
ckpt_sha(p) = strip(read(joinpath(p, "sha256.txt"), String))

# ---------------- minimal JSON writer ----------------
jesc(s) = replace(string(s), "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n")
function jw(io, x)
    if x === nothing
        print(io, "null")
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Integer
        print(io, x)
    elseif x isa AbstractFloat
        isfinite(x) ? print(io, repr(Float64(x))) : print(io, "null")
    elseif x isa AbstractString
        print(io, "\"", jesc(x), "\"")
    elseif x isa AbstractDict
        print(io, "{"); first = true
        for (k, v) in x
            first || print(io, ","); first = false
            print(io, "\"", jesc(k), "\":"); jw(io, v)
        end
        print(io, "}")
    elseif x isa NamedTuple
        jw(io, Dict(string(k) => getfield(x, k) for k in keys(x)))
    elseif x isa AbstractVector || x isa Tuple
        print(io, "["); for (i, v) in enumerate(x); i > 1 && print(io, ","); jw(io, v); end; print(io, "]")
    elseif x isa AbstractMatrix
        jw(io, [collect(x[i, :]) for i in axes(x, 1)])          # row-major list of rows
    else
        print(io, "\"", jesc(repr(x)), "\"")
    end
end

function main(run, out)
    mkpath(joinpath(out, "edges"))
    kk = joinpath(run, "kk")
    cal = load(joinpath(kk, "calibration.ckpt"))
    meta = Dict{String,Any}()
    meta["exporter"] = EXPORTER_VERSION
    meta["run_dir"] = run
    man = joinpath(run, "manifest.toml")
    meta["run_manifest_sha256"] = isfile(man) ? bytes2hex(sha256(read(man))) : nothing
    meta["calibration"] = Dict("coefficients" => cal.coefficients, "alpha" => cal.alpha, "E0" => cal.E0,
        "vacuum" => cal.vacuum, "singlet" => cal.singlet, "current" => cal.current, "descendant" => cal.descendant,
        "descendant_gap_error" => cal.descendant_gap_error, "leakage" => cal.leakage,
        "calibration_error" => cal.calibration_error, "ckpt_sha" => ckpt_sha(joinpath(kk, "calibration.ckpt")))

    blocks = Dict{String,Any}()
    for d in sort(readdir(joinpath(run, "blocks")))
        rp = joinpath(run, "blocks", d, "result.ckpt"); isdir(rp) || continue
        r = load(rp); rows = sort(r.rows; by=x -> x.energy)
        blocks[d] = Dict("dim" => r.dim, "energy" => [x.energy for x in rows], "c2" => [x.c2 for x in rows],
                         "x" => [x.x for x in rows], "rh" => [x.rh for x in rows],
                         "eig_ckpt_sha" => isdir(joinpath(run, "blocks", d, "eig.ckpt")) ? ckpt_sha(joinpath(run, "blocks", d, "eig.ckpt")) : nothing,
                         "result_ckpt_sha" => ckpt_sha(rp), "gate" => r.gate, "producer" => r.producer)
    end

    edges = Any[]
    for f in sort(readdir(joinpath(kk, "edges")))
        m = match(r"^s(\d)_s(\d)_L(\d+)_to_L(\d+)\.ckpt$", f); m === nothing && continue
        p = joinpath(kk, "edges", f); e = load(p)
        key = replace(f, ".ckpt" => "")
        ns = length(e.source_E); nt = length(e.target_E)
        size(e.overlaps[1]) == (nt, ns) && size(e.overlaps[2]) == (nt, ns) || error("overlap shape $f")
        size(e.image_gram) == (2, 2, ns) || error("gram shape $f")
        bin = joinpath(out, "edges", key * ".f64")
        open(bin, "w") do io
            write(io, Float64.(e.overlaps[1])); write(io, Float64.(e.overlaps[2])); write(io, Float64.(e.image_gram))
        end
        bk = "s$(e.a)_s$(e.b)_L$(e.li)"
        haskey(blocks, bk) && get(blocks[bk], "source_window", "block") == "block" && (blocks[bk]["parity"] = e.source_parity)
        if !haskey(blocks, bk)
            issorted(e.source_E) || error("unsorted source window $f")
            blocks[bk] = Dict("dim" => e.source_dim, "energy" => e.source_E, "c2" => e.source_C2, "x" => nothing,
                              "rh" => nothing, "parity" => e.source_parity, "source_window" => "target",
                              "source_colsha" => e.source_colsha, "source_status" => e.source_status, "from_edge" => key)
        elseif get(blocks[bk], "source_window", "block") == "target"
            sw = blocks[bk]
            (sw["energy"] == e.source_E && sw["c2"] == e.source_C2 && sw["source_colsha"] == e.source_colsha) ||
                error("inconsistent target source windows for $bk ($f vs $(sw["from_edge"]))")
        end
        push!(edges, Dict("key" => key, "a" => e.a, "b" => e.b, "li" => e.li, "lf" => e.lf,
            "source_colsha" => (hasproperty(e, :source_colsha) ? e.source_colsha : nothing),
            "target_colsha" => (hasproperty(e, :target_colsha) ? e.target_colsha : nothing),
            "ns" => ns, "nt" => nt, "source_E" => e.source_E, "source_C2" => e.source_C2, "source_parity" => e.source_parity,
            "target_E" => e.target_E, "target_C2" => e.target_C2, "target_frontier" => e.target_frontier,
            "target_status" => e.target_status, "target_full" => e.target_full, "target_parities" => e.target_parities,
            "energy_atol" => e.energy_atol, "source_dim" => e.source_dim, "target_dim" => e.target_dim,
            "identity" => e.identity, "ckpt_sha" => ckpt_sha(p), "bin" => "edges/" * key * ".f64",
            "bin_sha256" => bytes2hex(open(sha256, bin)), "layout" => "O1[nt,ns],O2[nt,ns],G[2,2,ns] Float64 LE column-major"))
    end

    projs = Any[]
    for f in sort(readdir(joinpath(kk, "projections")))
        endswith(f, ".ckpt") || continue
        p = joinpath(kk, "projections", f); r = load(p)
        for (t, chans) in r.threshold_channels, ch in chans
            push!(projs, Dict("block" => "s$(r.a)_s$(r.b)_L$(r.L)", "threshold" => t, "C2" => ch.C2,
                "source_ids" => ch.source_ids, "excluded_ids" => ch.excluded_ids, "missing_L" => ch.missing_L,
                "status" => ch.status,
                "k2_values" => (hasproperty(ch, :k2_values) ? ch.k2_values : nothing),
                "energies" => ch.energies, "ckpt_sha" => ckpt_sha(p)))
        end
    end
    # v0.3.6 (template): a neighbouring target window that is an empty sector has no KK edge (src/kkstage.jl: kk_edge!);
    # list such targets so that an analysis can tell an empty neighbour from a missing export.  Only the targets of
    # missing edges are opened, so no large window is loaded.
    have = Set((Int(e["a"]), Int(e["b"]), Int(e["li"]), Int(e["lf"])) for e in edges)
    empty_targets = String[]
    for bk in keys(blocks)
        mb = match(r"^s(\d)_s(\d)_L(\d+)$", bk); mb === nothing && continue
        a, b, li = parse.(Int, mb.captures)
        for lf in (li == 0 ? (1,) : (li - 1, li, li + 1))
            (a, b, li, lf) in have && continue
            tp = joinpath(run, "targets", "s$(a)_s$(b)_L$(lf).ckpt")
            isdir(tp) && isempty(load(tp).energy) && push!(empty_targets, "s$(a)_s$(b)_L$(lf)")
        end
    end
    meta["empty_targets"] = sort(unique(empty_targets))
    meta["blocks"] = blocks; meta["edges"] = edges; meta["production_projections"] = projs
    open(io -> jw(io, meta), joinpath(out, "meta.json"), "w")
    println("exported ", length(edges), " edges, ", length(blocks), " blocks, ", length(projs), " projection channels -> ", out)
end

main(ARGS[1], ARGS[2])
