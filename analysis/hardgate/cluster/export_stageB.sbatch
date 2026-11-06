#!/bin/bash
# Export the Stage B outputs of several N for the analysis (read-only on the run directories).
# Edit the #SBATCH lines for your cluster, then:
#   STAGEB_ROOT=<dir holding half/N<n>, 11/N<n>> EXPORT_ROOT=<out dir> JULIA=<julia> sbatch export_stageB.sbatch
# STAGEB_ROOT/half/N<n> are the out directories of the Tcc step stageb (sector half), 11/N<n> of sector 11.
#SBATCH -J blindpd_exportB
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o exportB-%j.out
set -euo pipefail
: "${STAGEB_ROOT:?set STAGEB_ROOT}" "${EXPORT_ROOT:?set EXPORT_ROOT}"
J=${JULIA:-julia}
HALF_N=${HALF_N:-"5 6 7 8 9 10"}
S11_N=${S11_N:-"5 6 7"}
CODE=$(cd "$(dirname "$0")" && pwd)/export_stageB.jl
[ -f "$CODE" ] || CODE=${SLURM_SUBMIT_DIR:-.}/export_stageB.jl
sha256sum "$CODE"
for n in $HALF_N; do /usr/bin/time -f "half N$n %e s %M KB" $J --startup-file=no "$CODE" "$STAGEB_ROOT/half/N$n" "$EXPORT_ROOT/half/N$n"; done
for n in $S11_N; do /usr/bin/time -f "11 N$n %e s %M KB" $J --startup-file=no "$CODE" "$STAGEB_ROOT/11/N$n" "$EXPORT_ROOT/11/N$n"; done
echo EXPORT_B_DONE
