# Full-Q OPE components of the integer sectors (v0.3.0; from xy_full_sources.jl 7d131f7d, N5-N11, and its
# sector-selecting copy xy_full_sources_sectors.jl e561eb36, N12).  For every selected (sA,sB) pair the KK
# scalar sources (Delta <= 5) are contracted with their FULL saved coefficient vectors (no support threshold,
# no tail drop) against the KK targets of spin l (Delta <= 6.6), for the singlet insertion and, on the diagonal
# pairs, also Gamma^2 and the tensor Q_G (two parts each, combined later by the analysis).  No eigen-solve.
#
# [fullq] keys: out, parity_kk (parity_kk.csv of the run; default <[parity] out>/parity_kk.csv),
#   sectors (default all six in the production order (0,1),(0,2),(1,2),(0,0),(1,1),(2,2), restricted to the
#   run's blocks), rank_max (4), source_delta_max (5.0), target_delta_max (6.6), threshold (0.25),
#   image_bytes (4e9), workers (nthreads).

const FULLQ_VERSION = "xy-full-sources-v2"
const FULLQ_PROFILE = "full_saved_KK_Q_sources_no_tail_drop_v1"
const FULLQ_SECTORS = [(0, 1), (0, 2), (1, 2), (0, 0), (1, 1), (2, 2)]
const FULLQ_HEADER = ["N","sector","insertion","rank","X_label","X_C2","X_Delta","X_K2","X_sigmaE","Z_label","Z_C2","Z_Delta","Z_K2","Z_sigmaE","me_1","me_2",
    "X_P","Z_P","X_P_expectation","Z_P_expectation","X_parity_verified","Z_parity_verified","X_parity_csv_row","Z_parity_csv_row",
    "source_profile","source_captured_weight","source_discarded_weight","source_coeff_norm2","source_support_count","source_raw_column_count",
    "source_window_sha","source_projection_sha","source_Q_sha","target_window_sha","target_projection_sha","parity_file_sha","edge_identity_sha","part_count","PH_projection_applied"]

_matrix_sha(A) = bytes2hex(sha256(collect(reinterpret(UInt8, vec(Matrix{Float64}(A))))))
_csvcell(x) = (s = string(x); occursin(r"[,\"\r\n]", s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s)
function _csvfields(line)
    fields = String[]; buf = IOBuffer(); quoted = false; i = firstindex(line)
    while i <= lastindex(line)
        ch = line[i]
        if ch == '"'
            j = nextind(line, i)
            quoted && j <= lastindex(line) && line[j] == '"' ? (write(buf, '"'); i = j) : (quoted = !quoted)
        elseif ch == ',' && !quoted
            push!(fields, String(take!(buf)))
        else
            write(buf, ch)
        end
        i = nextind(line, i)
    end
    !quoted || error("unterminated CSV quote")
    push!(fields, String(take!(buf))); fields
end
function _atomic_toml(path, data)
    tmp = path * ".partial"; open(io -> TOML.print(io, data), tmp, "w"); mv(tmp, path; force=true)
end

"""parity_kk.csv rows of integer sectors at `thr`, keyed by (sA, sB, L, C2, mode)."""
function read_kk_parities(path, thr)
    lines = readlines(path); isempty(lines) && error("empty parity CSV")
    head = _csvfields(first(lines)); required = ["sA", "sB", "L", "threshold", "C2", "mode", "Delta", "P_expectation", "P"]
    all(k -> k in head, required) || error("parity CSV missing required fields")
    idx = Dict(k => findfirst(==(k), head) for k in required); result = Dict{NTuple{5,Int},Any}()
    for (line_number, line) in enumerate(lines[2:end])
        isempty(strip(line)) && continue
        fields = _csvfields(line); length(fields) == length(head) || error("bad parity CSV row $(line_number + 1)")
        value(k) = parse(Float64, fields[idx[k]])
        abs(value("threshold") - thr) <= 1e-12 || continue
        # integer sectors only; never round a half-sector parity row onto an integer (0,0) identity
        abs(value("sA") - round(value("sA"))) <= 1e-12 && abs(value("sB") - round(value("sB"))) <= 1e-12 || continue
        key = (round(Int, value("sA")), round(Int, value("sB")), round(Int, value("L")), round(Int, value("C2")), round(Int, value("mode")))
        haskey(result, key) && error("duplicate parity mode key $key")
        d = value("Delta"); pe = value("P_expectation"); pp = round(Int, value("P"))
        all(isfinite, (d, pe)) || error("nonfinite parity evidence for $key")
        pp in (-1, 0, 1) || error("invalid parity for $key")
        result[key] = (Delta=d, P=pp, expectation=pe, row=line_number + 1, verified=pp != 0 && abs(pe - pp) <= 1e-6)
    end
    result
end

"""
    run_fullq(c, o) -> output directory

`c`: pipeline config of the finished run (g resolved); `o`: the [fullq] table (see the file header).
"""
function run_fullq(c, o)
    N = Int(c["N"]); RUN = _canonical_future(c["out"]); OD = _canonical_future(o["out"])
    _contains_path(RUN, OD) && error("output must be separate from the read-only run directory")
    SOURCE_MAX = Float64(get(o, "source_delta_max", 5.0)); TARGET_MAX = Float64(get(o, "target_delta_max", 6.6))
    THRESHOLD = Float64(get(o, "threshold", 0.25)); NORM_TOL = 1e-8
    LMAX = Int(get(o, "rank_max", 4)); 0 <= LMAX <= 4 || error("fullq.rank_max must lie in 0:4")
    present = unique([(b[1], b[2]) for b in c["blocks"]])
    SECTORS = haskey(o, "sectors") ? [Tuple(Int.(s)) for s in o["sectors"]] : [s for s in FULLQ_SECTORS if s in present]
    PARITY_PATH = abspath(o["parity_kk"]); PARITIES = read_kk_parities(PARITY_PATH, THRESHOLD); PARITY_SHA = file_digest(PARITY_PATH)
    pr = producer_info(); MANIFEST_PATH = joinpath(OD, "run_manifest.toml")
    mkpath(joinpath(OD, "edges")); mkpath(joinpath(OD, "operator_terms"))
    ident0 = Dict("version" => FULLQ_VERSION, "N" => N, "g" => Float64(c["g"]), "run" => RUN, "parity_sha256" => PARITY_SHA,
                  "source_Delta_max" => SOURCE_MAX, "target_Delta_max" => TARGET_MAX, "threshold" => THRESHOLD)
    if isfile(MANIFEST_PATH)
        old = TOML.parsefile(MANIFEST_PATH)
        for (k, v) in ident0
            get(old, k, nothing) == v || error("fullq input changed ($k); select a new output directory")
        end
    end
    manifest = merge(Dict{String,Any}(ident0), Dict{String,Any}("status" => "running", "profile" => FULLQ_PROFILE, "rank_max" => LMAX,
        "source_policy" => "V_projected=V_raw*full_Q_selected; no coefficient omitted", "PH_projection_applied" => false,
        "code" => Dict(string(k) => v for (k, v) in pairs(pr)), "sectors_processed" => [collect(s) for s in SECTORS],
        "sectors" => Any[], "targets" => Any[], "edges" => Any[]))
    _atomic_toml(MANIFEST_PATH, manifest)

    # Prefer the split-space read-only store if it already exists; at most one target-rank raw window lives
    # beside the projected scalar sources; library state is released between sectors.
    cfg = copy(c); isdir(joinpath(RUN, "segment_split", "space_2")) && (cfg["residency"] = "pair")
    spaces, basis_sha = ope3_spaces(cfg)
    manifest["segment_residency"] = spaces isa SpaceStore ? "pair_split_store" : "preexisting_all_segment_checkpoint"
    manifest["basis_sha"] = string(basis_sha)
    WORKERS = min(Threads.nthreads(), Int(get(o, "workers", Threads.nthreads()))); IMAGE_BYTES = Float64(get(o, "image_bytes", 4e9))
    lib = SegLibrary(builder=:tiled, tile_width=get(c, "tile", 64), workers=WORKERS, opmat_threads=get(c, "native", 8))
    spc(s) = spaces isa SpaceStore ? getspace!(spaces, s) : spaces[s+1]
    F = tuning_F(run_rows(RUN)); alpha = 2 / (F.EJ - F.E0); E0 = F.E0
    isfinite(alpha) && alpha > 0 || error("invalid energy calibration")
    Delta(E) = alpha * (E - E0)
    win(a, b, l) = (w = source_window(RUN, a, b, l); w === nothing ? target_window(RUN, a, b, l) : w)
    terms_cache = Dict{Tuple{Symbol,Int},Any}()
    terms(kind, l) = get!(() -> insertion_terms(N, kind, l; cache=joinpath(OD, "operator_terms")), terms_cache, (kind, l))
    function modes(a, b, l, W; dmax=TARGET_MAX)
        projection = joinpath(RUN, "kk", "projections", "s$(a)_s$(b)_L$(l).ckpt")
        isdir(projection) || error("required KK projection missing: $projection")
        x = _checked_load(projection); projection_sha = string(_ckpt_sha(projection))
        x.source_sha == W.sha || error("KK projection/source raw SHA mismatch: $projection")
        collect(x.source_E) == collect(W.E) || error("KK projection/source energy order mismatch")
        hasproperty(W, :basis_sha) && string(W.basis_sha) != string(basis_sha) && error("window/segment basis SHA mismatch")
        hit = filter(t -> abs(t[1] - THRESHOLD) <= 1e-12, x.threshold_channels)
        length(hit) == 1 || error("missing/nonunique threshold $THRESHOLD")
        result = Any[]
        for ch in last(only(hit))
            ch.Q === nothing && continue
            ids = collect(ch.source_ids)
            length(unique(ids)) == length(ids) && all(i -> 1 <= i <= length(W.E), ids) || error("bad KK source IDs")
            size(ch.Q, 1) == length(ids) || error("Q/source IDs dimension mismatch")
            k2 = diag(ch.Q' * ch.K2 * ch.Q)
            for j in axes(ch.Q, 2)
                d = Delta(ch.energies[j]); d <= dmax || continue
                coeff = zeros(Float64, length(W.E)); coeff[ids] = ch.Q[:, j]
                all(isfinite, coeff) && abs(sum(abs2, coeff) - 1) <= NORM_TOL || error("invalid full KK coefficient norm")
                ci = round(Int, ch.C2); abs(ci - ch.C2) <= 1e-4 || error("noninteger Casimir")
                key = (a, b, l, ci, Int(j)); haskey(PARITIES, key) || error("required parity identity missing: $key")
                p = PARITIES[key]
                abs(p.Delta - d) <= 1e-7 || error("parity/projection mode energy mismatch: $key")
                all(isfinite, (d, k2[j], ch.energy_mixing_residual[j])) || error("nonfinite KK mode diagnostics")
                push!(result, (c2=ci, energy=Float64(ch.energies[j]), sigmaE=Float64(ch.energy_mixing_residual[j]), K2=Float64(k2[j]),
                    coeff=coeff, label="kk_C2_$(ci)_m$(j)", mode=Int(j), P=p.P, P_expectation=p.expectation, parity_verified=p.verified,
                    parity_row=p.row, projection_sha=projection_sha, window_sha=string(W.sha)))
            end
        end
        result, projection_sha
    end
    real_finite(v) = (isfinite(real(v)) && isfinite(imag(v)) || error("nonfinite matrix element");
                      abs(imag(v)) <= 1e-12 * max(1, abs(real(v))) || error("complex matrix element requires an explicit export convention"); Float64(real(v)))
    function process_sector(io, a, b, counts)
        sector = "s$(a)_s$(b)"
        catalogue_sources = count(kv -> kv.first[1:3] == (a, b, 0) && kv.second.Delta <= SOURCE_MAX, PARITIES)
        W0 = win(a, b, 0); W0 === nothing && error("required scalar source window missing: $sector")
        if isempty(W0.E)
            catalogue_sources == 0 || error("catalogue sources in empty source window")
            push!(manifest["sectors"], Dict("sector" => sector, "status" => "empty_saved_source_window", "catalogue_sources" => 0)); return
        end
        allmodes, source_projection_sha = modes(a, b, 0, W0; dmax=SOURCE_MAX)
        srcs = filter(m -> Delta(m.energy) <= SOURCE_MAX, allmodes)
        length(srcs) == catalogue_sources || error("catalogue/projected scalar-source coverage differs in $sector")
        if isempty(srcs)
            push!(manifest["sectors"], Dict("sector" => sector, "status" => "no_scalar_KK_sources_below_cut", "catalogue_sources" => 0)); return
        end
        source_raw_count = length(W0.E); source_window_sha = string(W0.sha)
        Q = hcat([m.coeff for m in srcs]...)
        size(W0.V, 2) == size(Q, 1) || error("source Q/raw-vector dimension mismatch")
        all(isfinite, Q) && all(isfinite, W0.V) || error("nonfinite source basis")
        norm(Q' * Q - I, Inf) <= NORM_TOL || error("selected KK source Q is not orthonormal")
        projected = W0.V * Q                       # every coefficient, no support threshold or energy clipping
        norm(projected' * projected - I, Inf) <= NORM_TOL || error("full projected source vectors not orthonormal")
        # independent probe: accumulate the densest coefficient vector by axpy instead of the matrix product
        jp = argmax([count(x -> !iszero(x), view(Q, :, k)) for k in axes(Q, 2)]); ref = zeros(Float64, size(W0.V, 1))
        for k in axes(Q, 1); iszero(Q[k, jp]) || BLAS.axpy!(Q[k, jp], view(W0.V, :, k), ref); end
        probe_gap = norm(ref - view(projected, :, jp)) / max(norm(ref), 1.0)
        probe_gap <= 1e-10 || error("independent full-Q vector contraction probe failed: $probe_gap")
        qsha = _matrix_sha(Q)
        Ssha = bytes2hex(sha256(string(FULLQ_VERSION, "|", source_window_sha, "|", source_projection_sha, "|", qsha)))
        Sproj = (V=projected, E=[m.energy for m in srcs], C2=[m.c2 for m in srcs], sha=Ssha)
        W0 = nothing; allmodes = nothing; Q = nothing; GC.gc()
        ss = [spc(a), spc(b)]; qn = [2N, 0, 0]; csS = BuildCompSpace(ss, qn, 0)
        insertions = a == b ? (:singlet, :gamma2, :tensorG) : (:singlet,)
        push!(manifest["sectors"], Dict("sector" => sector, "status" => "full_sources_prepared", "catalogue_sources" => catalogue_sources,
            "selected_sources" => length(srcs), "source_raw_columns" => source_raw_count, "source_window_sha" => source_window_sha,
            "source_projection_sha" => source_projection_sha, "source_Q_sha" => qsha,
            "full_Q_probe_relative_error" => probe_gap, "full_Q_probe_label" => srcs[jp].label))
        for l in 0:LMAX
            target_expected = count(kv -> kv.first[1:3] == (a, b, l) && kv.second.Delta <= TARGET_MAX, PARITIES)
            Tw = win(a, b, l); Tw === nothing && error("required saved target window missing: $sector rank$l")
            if isempty(Tw.E)
                target_expected == 0 || error("catalogue targets in empty target window")
                push!(manifest["targets"], Dict("sector" => sector, "rank" => l, "status" => "empty_saved_target_window", "target_count" => 0)); continue
            end
            target_modes, target_projection_sha = modes(a, b, l, Tw)
            targets = filter(m -> Delta(m.energy) <= TARGET_MAX, target_modes)
            length(targets) == target_expected || error("target catalogue/KK coverage differs")
            push!(manifest["targets"], Dict("sector" => sector, "rank" => l, "status" => (isempty(targets) ? "no_saved_KK_target_candidates_in_cut" : "available_saved_KK_candidates"),
                "target_count" => length(targets), "target_raw_columns" => length(Tw.E), "target_window_sha" => string(Tw.sha), "target_projection_sha" => target_projection_sha))
            isempty(targets) && continue
            csT = l == 0 ? csS : BuildCompSpace(ss, qn, 2l)
            for kind in insertions
                ident = (version=FULLQ_VERSION, profile=FULLQ_PROFILE, sector=sector, insertion=string(kind), rank=l,
                    source_sha=Ssha, source_raw_window_sha=source_window_sha, source_projection_sha=source_projection_sha,
                    source_Q_sha=qsha, source_labels=Tuple(m.label for m in srcs), target_sha=string(Tw.sha),
                    target_projection_sha=target_projection_sha, basis_sha=string(basis_sha), parity_sha=PARITY_SHA,
                    cols=collect(1:length(srcs)), source_Delta_max=SOURCE_MAX, target_Delta_max=TARGET_MAX)
                edge_sha = bytes2hex(sha256(string(ident)))
                e = _ope3_edge(OD, "fullQ_$(sector)_$(kind)_r$l", lib, ss, csS, csT, terms(kind, l), 2l,
                               Sproj, 1:length(srcs), Tw, ident; workers=WORKERS, image_bytes=IMAGE_BYTES)
                np = kind === :gamma2 ? 1 : 2
                length(e.overlaps) == np || error("operator-part count differs")
                all(M -> size(M) == (length(Tw.E), length(srcs)) && all(isfinite, M), e.overlaps) || error("edge overlap shape/nonfinite failure")
                edge_rows = 0
                for (j, x) in enumerate(srcs), z in targets
                    values = [real_finite(dot(z.coeff, view(M, :, j))) for M in e.overlaps]
                    full_norm = sum(abs2, x.coeff); abs(full_norm - 1) <= NORM_TOL || error("source capture norm failure")
                    row = Any[N, sector, string(kind), l, x.label, x.c2, Delta(x.energy), x.K2, x.sigmaE, z.label, z.c2, Delta(z.energy), z.K2, z.sigmaE,
                        values[1], np == 2 ? values[2] : 0.0, x.P, z.P, x.P_expectation, z.P_expectation, x.parity_verified, z.parity_verified, x.parity_row, z.parity_row,
                        FULLQ_PROFILE, 1.0, 0.0, full_norm, count(v -> !iszero(v), x.coeff), source_raw_count, source_window_sha, source_projection_sha, qsha,
                        string(Tw.sha), target_projection_sha, PARITY_SHA, edge_sha, np, false]
                    length(row) == length(FULLQ_HEADER) || error("export schema mismatch")
                    println(io, join(_csvcell.(row), ',')); edge_rows += 1
                end
                edge_rows == length(srcs) * length(targets) || error("incomplete edge export")
                counts[] += edge_rows
                push!(manifest["edges"], Dict("sector" => sector, "rank" => l, "insertion" => string(kind), "source_count" => length(srcs),
                    "target_count" => length(targets), "expected_rows" => edge_rows, "exported_rows" => edge_rows, "identity_sha" => edge_sha, "part_count" => np))
                flush(io); _atomic_toml(MANIFEST_PATH, manifest)
                println("FULLQ_EDGE N=$N $sector $kind rank$l rows=$edge_rows total=$(counts[])"); flush(stdout)
            end
        end
        nothing
    end
    partial = joinpath(OD, "xy_blocks_raw.csv.partial"); final = joinpath(OD, "xy_blocks_raw.csv")
    counts = Ref(0); t0 = time()
    try
        open(partial, "w") do io
            println(io, join(FULLQ_HEADER, ','))
            for (a, b) in SECTORS
                process_sector(io, a, b, counts)
                spaces isa SpaceStore ? foreach(s -> dropspace!(spaces, lib, s), unique([a, b])) : foreach(s -> release!(lib, spaces[s+1]), unique([a, b]))
                GC.gc()
            end
        end
        counts[] > 0 || error("refusing an empty successful export")
        expected = sum(e["expected_rows"] for e in manifest["edges"])
        expected == counts[] || error("manifest/row totals mismatch")
        mv(partial, final; force=true)
        manifest["status"] = "done"; manifest["rows"] = counts[]; manifest["expected_rows"] = expected
        manifest["csv_sha256"] = bytes2hex(sha256(read(final))); manifest["elapsed_seconds"] = time() - t0
        manifest["source_capture_gate"] = "ALL_PASS_full_Q_no_discard"; manifest["full_Q_probe_gate"] = "ALL_PASS_independent_axpy_accumulation"
        _atomic_toml(MANIFEST_PATH, manifest)
        println("FULLQ_DONE N=$N rows=$(counts[]) seconds=$(round(time() - t0; digits=1)) maxrss_gib=$(round(Sys.maxrss() / 2^30; digits=2))")
    catch err
        manifest["status"] = "failed"; manifest["failure"] = sprint(showerror, err); manifest["rows_written_partial"] = counts[]
        _atomic_toml(MANIFEST_PATH, manifest)
        rethrow()
    end
    OD
end
