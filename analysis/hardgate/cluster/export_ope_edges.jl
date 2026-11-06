# blind_pd_pipeline v0.2.0 — cluster-side exporter of fullQ insertion-operator edges (read-only)
#
# Input : <fullQ dir> = the out directory of the Tcc step fullq for one N   (edges/fullQ_<sector>_<insertion>_r<rank>.ckpt)
# Output: <out>/ope_meta.json and <out>/ope_edges/<name>.f64  (parts p=1..P of overlaps[p][nt, ns], Float64 LE col-major)
# overlaps[p][t, j] = reduced <target raw state t of block (sector, L=rank) || O_p^(rank) || source KK mode j>
# Usage: julia --startup-file=no export_ope_edges.jl <fullQ dir> <out>
using Serialization, SHA
load(p) = deserialize(joinpath(p, "payload.jls"))
jesc(s) = replace(string(s), "\\" => "\\\\", "\"" => "\\\"")
function jw(io, x)
    if x === nothing; print(io, "null")
    elseif x isa Bool; print(io, x ? "true" : "false")
    elseif x isa Integer; print(io, x)
    elseif x isa AbstractFloat; isfinite(x) ? print(io, repr(Float64(x))) : print(io, "null")
    elseif x isa AbstractString; print(io, "\"", jesc(x), "\"")
    elseif x isa AbstractDict
        print(io, "{"); f = true
        for (k, v) in x; f || print(io, ","); f = false; print(io, "\"", jesc(k), "\":"); jw(io, v); end; print(io, "}")
    elseif x isa NamedTuple; jw(io, Dict(string(k) => getfield(x, k) for k in keys(x)))
    elseif x isa AbstractVector || x isa Tuple
        print(io, "["); for (i, v) in enumerate(x); i > 1 && print(io, ","); jw(io, v); end; print(io, "]")
    else; print(io, "\"", jesc(repr(x)), "\""); end
end
function main(src, out)
    mkpath(joinpath(out, "ope_edges")); E = Any[]
    for f in sort(readdir(joinpath(src, "edges")))
        endswith(f, ".ckpt") || continue
        x = load(joinpath(src, "edges", f)); name = replace(f, ".ckpt" => "")
        nt, ns = size(x.overlaps[1])
        bin = joinpath(out, "ope_edges", name * ".f64")
        open(bin, "w") do io; for M in x.overlaps; write(io, Float64.(M)); end; end
        push!(E, Dict("name" => name, "parts" => length(x.overlaps), "nt" => nt, "ns" => ns, "scols" => x.scols,
                      "identity" => x.identity, "bin" => "ope_edges/" * name * ".f64",
                      "bin_sha256" => bytes2hex(open(sha256, bin)),
                      "ckpt_sha" => strip(read(joinpath(src, "edges", f, "sha256.txt"), String))))
    end
    meta = Dict("exporter" => "blind_pd_pipeline/export_ope_edges v0.2.0", "source_dir" => src, "edges" => E)
    for g in ("run_manifest.toml", "validation.json", "export_verification.toml")
        p = joinpath(src, g); isfile(p) && (meta[g * "_sha256"] = bytes2hex(open(sha256, p)))
    end
    open(io -> jw(io, meta), joinpath(out, "ope_meta.json"), "w")
    println("exported ", length(E), " ope edges -> ", out)
end
main(ARGS[1], ARGS[2])
