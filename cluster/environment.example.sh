#!/bin/bash
# Self-contained environment of Tcc for a cluster (or any Linux machine).  Copy to cluster/environment.sh (ignored by
# git), fill in the three paths, and `source` it in every job and before setup:
#
#   cp cluster/environment.example.sh cluster/environment.sh      # edit TCC_REPO, TCC_DATA, JULIA
#   source cluster/environment.sh
#   JULIA_PKG_OFFLINE=false "$JULIA" deps/setup_env.jl              # once, on a machine with internet (see below)
#   sbatch cluster/validate_env.sbatch                              # once per CPU type: load check + regression
#
# Everything Julia needs (packages, binary artifacts, the General registry, compile caches) lives in ONE depot,
# $TCC_DEPOT (default $TCC_REPO/depot), and Julia runs offline, so a job never reads another depot or the network.
#
# Cluster without internet: build the depot on a Linux x86_64 machine with internet (artifacts are platform-specific;
# WSL works), with the same Julia version and the same TCC_DEPOT layout, then copy the repository with its depot:
#   source cluster/environment.sh; JULIA_PKG_OFFLINE=false "$JULIA" deps/setup_env.jl
#   tar --exclude=depot/compiled -czf tcc_env.tgz -C "$(dirname "$TCC_REPO")" "$(basename "$TCC_REPO")"
# and unpack it on the cluster.  env/Manifest.toml refers to deps/FuzzifiED.jl by a relative path, so the copy works
# from any location.  The compile caches are rebuilt on the first run on each CPU type (about a minute).

export TCC_REPO=${TCC_REPO:-/path/to/fuzzysphere-dqcp-template}   # this repository
export TCC_DATA=${TCC_DATA:-/path/to/your/data}                   # data root: all outputs go here
export JULIA=${JULIA:-julia}                                      # Julia 1.11 (validated with 1.11.7)

export TCC_DEPOT=${TCC_DEPOT:-$TCC_REPO/depot}
export JULIA_DEPOT_PATH=$TCC_DEPOT                 # one depot only (no trailing ':'): nothing outside it is read
export JULIA_PKG_OFFLINE=${JULIA_PKG_OFFLINE:-true}
export JULIA_PKG_PRECOMPILE_AUTO=0                 # compile on first use, never on Pkg operations
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1    # Tcc sets BLAS and FuzzifiED threads itself ([threads] of a plan)
export TCC_PYTHON=${TCC_PYTHON:-python3}           # Python with numpy and pandas (step parity_join, analyses)
