module SO5C2DensityPhase

export cross_density_coefficient

"""
    cross_density_coefficient(nm)

Coefficient for the all-rank inter-segment density contribution to the
SO(5) Casimir when using pinned FuzzifiED 2.0.1 `GetDensityMod`. The electron
mode carries an `(im)^(2m)` phase. The two density modes contribute
`(-1)^(nm-1)` relative to the unphased four-flavor density convention.
For odd `nm` this returns the original example's `-0.5` exactly; for even
`nm` it returns `+0.5`.
"""
function cross_density_coefficient(nm::Integer)
    nm >= 1 || throw(ArgumentError("nm must be positive"))
    return isodd(nm) ? -0.5 : 0.5
end

end
