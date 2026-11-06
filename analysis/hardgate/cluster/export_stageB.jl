# blind_pd_pipeline v0.3.0 — read-only exporter for stageB sectors (half-integer windows, (1,1) extended windows)
# Input : <stageB dir> = the out directory of the Tcc step stageb for one N and sector (half or 11)
# Output: <out>/stageB_meta.json  +  <out>/bin/*.f64
#   windows/L*.ckpt : energies and every per-state numeric vector (C2, parity labels, residuals); vectors NOT exported
#   kk/edges/*.ckpt : generator overlaps O1,O2 [nt,ns], image gram [2,2,ns] (same layout as export_kk.jl)
#   kk/L*.ckpt      : production KK modes (label, C2, P, pseg, energy, K2, coeff over window states)
#   edges/*.ckpt    : insertion edges overlaps_p [nt, ns] (sources = production source modes, scols)
using Serialization, SHA
load(p) = deserialize(joinpath(p, "payload.jls"))
function jesc(s)
    io = IOBuffer()
    for c in string(s)
        if c == '"'; print(io, "\\\"")
        elseif c == '\\'; print(io, "\\\\")
        elseif c == '\n'; print(io, "\\n")
        elseif c == '\r'; print(io, "\\r")
        elseif c == '\t'; print(io, "\\t")
        elseif c < ' '; print(io, "\\u", lpad(string(UInt16(c), base=16), 4, '0'))
        else; print(io, c); end
    end
    String(take!(io))
end
function jw(io, x)
    if x === nothing; print(io, "null")
    elseif x isa Bool; print(io, x ? "true" : "false")
    elseif x isa Integer; print(io, x)
    elseif x isa AbstractFloat; isfinite(x) ? print(io, repr(Float64(x))) : print(io, "null")
    elseif x isa AbstractString || x isa Symbol; print(io, "\"", jesc(x), "\"")
    elseif x isa AbstractDict
        print(io, "{"); f = true
        for (k, v) in x; f || print(io, ","); f = false; print(io, "\"", jesc(k), "\":"); jw(io, v); end; print(io, "}")
    elseif x isa NamedTuple; jw(io, Dict(string(k) => getfield(x, k) for k in keys(x)))
    elseif x isa AbstractVector || x isa Tuple
        print(io, "["); for (i, v) in enumerate(x); i > 1 && print(io, ","); jw(io, v); end; print(io, "]")
    else; print(io, "\"", jesc(first(repr(x), 200)), "\""); end
end
wbin(path, arrs) = (open(path, "w") do io; for a in arrs; write(io, Float64.(a)); end; end; bytes2hex(open(sha256, path)))
function main(src, out)
    mkpath(joinpath(out, "bin")); meta = Dict{String,Any}("exporter" => "blind_pd_pipeline/export_stageB v0.3.0", "source_dir" => src)
    for f in ("stageB_manifest.toml", "input_binding.toml", "validation.json")
        p = joinpath(src, f); isfile(p) && (meta[f * "_sha256"] = bytes2hex(open(sha256, p)); meta[f] = read(p, String))
    end
    W = Dict{String,Any}()
    for f in sort(readdir(joinpath(src, "windows")))
        x = load(joinpath(src, "windows", f)); n = length(x.energy); d = Dict{String,Any}("n" => n)
        for k in keys(x)
            v = getfield(x, k)
            if v isa AbstractVector && length(v) == n && eltype(v) <: Real
                d[string(k)] = collect(v)
            elseif v isa Real || v isa AbstractString
                d[string(k)] = v
            elseif !(v isa AbstractArray)
                d[string(k)] = first(repr(v), 300)
            end
        end
        d["ckpt_sha"] = strip(read(joinpath(src, "windows", f, "sha256.txt"), String)); W[replace(f, ".ckpt" => "")] = d
        x = nothing; GC.gc()
    end
    meta["windows"] = W
    KE = Any[]
    for f in sort(readdir(joinpath(src, "kk", "edges")))
        e = load(joinpath(src, "kk", "edges", f)); ns = length(e.source_E); nt = length(e.target_E)
        b = joinpath(out, "bin", "kk_" * replace(f, ".ckpt" => "") * ".f64")
        h = wbin(b, [e.overlaps[1], e.overlaps[2], e.image_gram])
        d = Dict{String,Any}(string(k) => getfield(e, k) for k in keys(e) if !(getfield(e, k) isa AbstractMatrix || getfield(e, k) isa AbstractArray{<:Any,3} || k == :overlaps))
        d["ns"] = ns; d["nt"] = nt; d["bin"] = relpath(b, out); d["bin_sha256"] = h; d["name"] = replace(f, ".ckpt" => "")
        push!(KE, d)
    end
    meta["kk_edges"] = KE
    KM = Dict{String,Any}()
    for f in sort(readdir(joinpath(src, "kk")))
        endswith(f, ".ckpt") || continue
        x = load(joinpath(src, "kk", f))
        KM[replace(f, ".ckpt" => "")] = Dict("L" => x.L, "threshold" => x.threshold, "source_sha" => x.source_sha,
            "calibration" => Dict("coefficients" => x.calibration.coefficients, "alpha" => x.calibration.alpha, "E0" => x.calibration.E0),
            "modes" => [Dict(string(k) => getfield(m, k) for k in keys(m)) for m in x.modes])
    end
    meta["kk_modes"] = KM
    IE = Any[]
    for f in sort(readdir(joinpath(src, "edges")))
        x = load(joinpath(src, "edges", f)); nt, ns = size(x.overlaps[1])
        b = joinpath(out, "bin", "ope_" * x.name * ".f64"); h = wbin(b, x.overlaps)
        push!(IE, Dict("name" => x.name, "parts" => length(x.overlaps), "nt" => nt, "ns" => ns, "scols" => x.scols,
                       "identity" => x.identity, "bin" => relpath(b, out), "bin_sha256" => h))
    end
    meta["ope_edges"] = IE
    open(io -> jw(io, meta), joinpath(out, "stageB_meta.json"), "w")
    println("exported windows=", length(W), " kk_edges=", length(KE), " kk_mode_sets=", length(KM), " ope_edges=", length(IE), " -> ", out)
end
main(ARGS[1], ARGS[2])
