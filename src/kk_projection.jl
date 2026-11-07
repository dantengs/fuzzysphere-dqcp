module KKProjection

using LinearAlgebra
using Serialization
using SHA

export checked_checkpoint, save_checkpoint, projected_transition, project_channel,
       fit_vacuum_generator, state_window, closure_report

"""Read a checkpoint only after verifying its payload hash."""
function checked_checkpoint(path::AbstractString; expected_meta=nothing)
    payload = joinpath(path, "payload.jls")
    digest = joinpath(path, "sha256.txt")
    isfile(payload) && isfile(digest) || error("Incomplete checkpoint: $path")
    actual = open(io -> bytes2hex(sha256(io)), payload)
    strip(read(digest, String)) == actual || error("Checkpoint hash mismatch: $path")
    value = open(deserialize, payload)
    value isa NamedTuple || error("Invalid checkpoint: $path")
    if expected_meta !== nothing
        haskey(value, :meta) && value.meta == expected_meta ||
            error("Checkpoint identity mismatch: $path")
    end
    value
end

"""Write one immutable, hash-bound checkpoint. A prior result is never overwritten."""
function save_checkpoint(path::AbstractString, value)
    ispath(path) && error("Checkpoint already exists: $path")
    mkpath(dirname(path))
    pending = mktempdir(dirname(path); prefix=".kk-pending-")
    try
        payload = joinpath(pending, "payload.jls")
        open(io -> serialize(io, value), payload, "w")
        write(joinpath(pending, "sha256.txt"), open(io -> bytes2hex(sha256(io)), payload) * "\n")
        mv(pending, path)
    finally
        ispath(pending) && rm(pending; recursive=true, force=true)
    end
    path
end

"""Validate a saved eigensystem and its Casimir/H diagnostic rows."""
function state_window(eig, diagnostics; tolerance=1e-6)
    E, V, rows = eig.energy, eig.vectors, diagnostics.rows
    E isa Vector{Float64} && V isa Matrix{Float64} || error("Wrong saved eigensystem types")
    eig.dim == diagnostics.dim == size(V, 1) || error("Saved basis dimensions differ")
    length(E) == size(V, 2) == length(rows) || error("Saved eigenstate count differs")
    length(E) >= 2 || error("Saved window lacks a guard state")
    all(isfinite, E) && all(isfinite, V) || error("Nonfinite eigenstates")
    maximum(abs, V' * V - I) <= tolerance || error("Saved eigenstates are not orthonormal")
    for i in eachindex(E)
        r = rows[i]
        E[i] == r.energy || error("Energy/index correspondence broken")
        r.rc <= tolerance && r.rh !== nothing && r.rh <= tolerance ||
            error("Saved Casimir/H residual exceeds tolerance")
        abs(r.norm - 1) <= tolerance || error("Saved state is not normalized")
    end
    (E=E, V=V, rows=rows)
end

"""Minimum vacuum leakage with one real calibration constraint."""
function fit_vacuum_generator(vacuum_images::AbstractMatrix, calibration::AbstractVector,
                              target::Real; rtol=1e-11)
    size(vacuum_images, 2) == length(calibration) || throw(DimensionMismatch())
    all(isfinite, vacuum_images) && all(isfinite, calibration) || error("Nonfinite generator fit")
    target > 0 || error("Calibration target must be positive")
    G = Hermitian(real(vacuum_images' * vacuum_images))
    F = eigen(G)
    scale = max(maximum(abs, F.values), eps())
    null = abs.(F.values) .<= rtol * scale
    c = real.(calibration)
    if any(null) && norm(F.vectors[:, null]' * c) > rtol * max(norm(c), 1)
        direction = F.vectors[:, null] * (F.vectors[:, null]' * c)
        method = "calibrated_gram_nullspace"
    else
        direction = pinv(Matrix(G); rtol=rtol) * c
        method = "minimum_leakage_pseudoinverse"
    end
    denominator = dot(c, direction)
    abs(denominator) > eps() || error("Singular generator calibration")
    coefficients = target * direction / denominator
    (coefficients=coefficients, leakage=real(coefficients' * G * coefficients),
     calibration_error=dot(c, coefficients) - target,
     gram_eigenvalues=F.values, method=method)
end

"""Matrix-free operator action: retain only target×source low-state overlaps.

`operator * vector` is evaluated once per source state. `target` may be a
small selected subset of the saved target eigenstates; source vectors are not
copied into another D×k array.
"""
function projected_transition(operator, source, target; source_ids=eachindex(source.E),
                              target_ids=eachindex(target.E))
    si, ti = collect(source_ids), collect(target_ids)
    isempty(si) && error("Empty source selection")
    isempty(ti) && error("Empty target selection")
    if hasproperty(operator, :cpspd)
        operator.cpspd.dim == size(source.V, 1) || throw(DimensionMismatch("source action"))
        operator.cpspf.dim == size(target.V, 1) || throw(DimensionMismatch("target action"))
    else
        size(operator) == (size(target.V, 1), size(source.V, 1)) ||
            throw(DimensionMismatch("operator action"))
    end
    overlap = Matrix{ComplexF64}(undef, length(ti), length(si))
    norm2 = Vector{Float64}(undef, length(si))
    for (j, id) in enumerate(si)
        image = operator * source.V[:, id]
        length(image) == size(target.V, 1) || throw(DimensionMismatch("target action"))
        norm2[j] = real(dot(image, image))
        for (i, tid) in enumerate(ti)
            overlap[i, j] = dot(view(target.V, :, tid), image)
        end
    end
    captured = vec(sum(abs2, overlap; dims=1))
    leak = max.(norm2 .- captured, 0)
    (matrix=overlap, image_norm2=norm2, captured_norm2=captured,
     missing_window_norm2=leak, source_ids=si, target_ids=ti)
end

"""Classify K² within one fixed Casimir channel and known L=0→1 component.

The factor 3 is the reduced q=1 convention. An unmatched target
or a missing L=1 block is an error, never an implicit zero contribution.
"""
function project_channel(overlap, source_E, target_E; threshold=0.25,
                         required_count=1, energy_atol=1e-10)
    size(overlap) == (length(target_E), length(source_E)) || throw(DimensionMismatch())
    isempty(target_E) && error("Missing L=1 target states")
    all(isfinite, overlap) && all(isfinite, source_E) && all(isfinite, target_E) ||
        error("Nonfinite K input")
    ceiling = maximum(target_E)
    maximum(source_E) < ceiling - energy_atol ||
        error("Source window exceeds target ceiling; K² closure is unverified")
    K = overlap .* [Et < Es - energy_atol for Et in target_E, Es in source_E]
    K2 = Hermitian(3 * K' * K)
    F = eigen(K2)
    values = real.(F.values)
    count(<=(threshold), values) >= required_count ||
        error("Too few low-K² modes for requested primary count")
    count_keep = count(<=(threshold), values)
    Q = F.vectors[:, 1:count_keep]
    H = Diagonal(source_E)
    D = eigen(Hermitian(Q' * H * Q))
    V = Q * D.vectors
    residual = [norm(H * view(V, :, j) - D.values[j] * view(V, :, j)) for j in eachindex(D.values)]
    (k2_values=values, k2_matrix=Matrix(K2), projected_energies=D.values,
     projected_coefficients=V, energy_residuals=residual,
     kept_max=values[count_keep],
     discarded_min=count_keep < length(values) ? values[count_keep+1] : Inf,
     target_ceiling=ceiling, closure_margin=ceiling - maximum(source_E))
end

"""Describe physical coverage; false closure is reported rather than concealed."""
function closure_report(source_l::Int, available_target_l::AbstractVector{Int}; generator_rank=1)
    required = collect(max(0, source_l-generator_rank):source_l+generator_rank)
    # Rank one allows same L for L>0; only 0 -> 0 is forbidden.
    filter!(l -> !(source_l == 0 && l == 0 && generator_rank == 1), required)
    missing = setdiff(required, available_target_l)
    (source_l=source_l, required_target_l=required, missing_target_l=missing,
     full_angular_closure=isempty(missing))
end

end
