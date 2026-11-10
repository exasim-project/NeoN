#!/usr/bin/env bash
#----------------------------------------------------------------------------------------
# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense
#----------------------------------------------------------------------------------------
# Produce the data of figures/neon_performance.png with the NeoFOAM momentum benchmark.
#
# Requires a sourced OpenFOAM environment (blockMesh). Unless NEOFOAM_DIR is given, NeoFOAM
# is cloned into ${RESULTS_DIR}/NeoFOAM. bench_momentum is (re)built against the NeoN tree
# containing this script, using the OpenMP host backend:
#
#   cmake --preset ${NEOFOAM_PRESET} -DNEOFOAM_BUILD_BENCHMARKS=ON -DNEOFOAM_NEON_DIR=${NEON_DIR} \
#         -DNeoN_WITH_OMP=ON -DNeoN_WITH_THREADS=OFF ${NEOFOAM_CMAKE_ARGS}
#
# Run once per device, e.g. inside a SLURM allocation on a full node:
#
#   NEON_DEVICE=nvidia_h200 ./runPerformance.sh
#   NEON_DEVICE=nvidia_a100 NEON_THREADS=1 ./runPerformance.sh
#
# Environment variables (all optional, shown with defaults):
#   NEOFOAM_DIR      path to an existing NeoFOAM source tree (clone into RESULTS_DIR)
#   NEOFOAM_URL      repository to clone NeoFOAM from        (exasim-project/NeoFOAM)
#   NEOFOAM_BRANCH   NeoFOAM branch to clone                 (develop)
#   NEOFOAM_PRESET   NeoFOAM build preset directory name     (profiling)
#   NEOFOAM_CMAKE_ARGS  additional NeoFOAM configure arguments (none)
#   NEON_DIR         NeoN source tree NeoFOAM is built with  (this repository)
#   NEON_DEVICE      label used in the output file names     (unknown)
#   NEON_THREADS     space-separated host thread counts      (1 2 4 8 16 32)
#   NEON_RES         space-separated cube resolutions        (75 100 ... 275)
#   RESULTS_DIR      directory for the generated csv files   (results)
#   PYTHON           interpreter with foamlib and pandas     (python3)
#
# Writes ${RESULTS_DIR}/momentum_${NEON_DEVICE}.<threads>.csv, see plotPerformance.py.
#----------------------------------------------------------------------------------------
#SBATCH --job-name=neon-joss-perf
#SBATCH --output=neon_joss_perf_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEON_DIR="$(realpath "${NEON_DIR:-${SCRIPT_DIR}/../../..}")"
RESULTS_DIR="$(realpath -m "${RESULTS_DIR:-results}")"
NEOFOAM_URL="${NEOFOAM_URL:-https://github.com/exasim-project/NeoFOAM.git}"
NEOFOAM_BRANCH="${NEOFOAM_BRANCH:-develop}"
PRESET="${NEOFOAM_PRESET:-profiling}"
NEON_DEVICE="${NEON_DEVICE:-unknown}"
THREADS="${NEON_THREADS:-1 2 4 8 16 32}"
RES="${NEON_RES:-75 100 125 150 175 200 225 250 275}"
PYTHON="${PYTHON:-python3}"

command -v blockMesh >/dev/null || { echo "source OpenFOAM first (blockMesh not found)" >&2; exit 1; }
mkdir -p "${RESULTS_DIR}"

if [[ -z "${NEOFOAM_DIR:-}" ]]; then
    NEOFOAM_DIR="${RESULTS_DIR}/NeoFOAM"
    if [[ ! -d "${NEOFOAM_DIR}" ]]; then
        echo ">>> Cloning NeoFOAM (branch=${NEOFOAM_BRANCH})"
        git clone --depth 1 --branch "${NEOFOAM_BRANCH}" "${NEOFOAM_URL}" "${NEOFOAM_DIR}"
    fi
fi
NEOFOAM_DIR="$(realpath "${NEOFOAM_DIR}")"

BENCH="${NEOFOAM_DIR}/build/${PRESET}/bin/benchmarks/bench_momentum"
SUITE="${NEOFOAM_DIR}/benchmarks/benchmarkSuite"
WORK_DIR="${RESULTS_DIR}/cases_${NEON_DEVICE}"

echo ">>> Building NeoFOAM benchmarks (preset=${PRESET})"
(
    cd "${NEOFOAM_DIR}"
    # the NeoFOAM presets append $NEON_DEVICE to the build directory
    # shellcheck disable=SC2086
    env -u NEON_DEVICE cmake --preset "${PRESET}" -DNEOFOAM_BUILD_BENCHMARKS=ON \
        -DNEOFOAM_NEON_DIR="${NEON_DIR}" -DNeoN_WITH_OMP=ON -DNeoN_WITH_THREADS=OFF \
        ${NEOFOAM_CMAKE_ARGS:-}
    env -u NEON_DEVICE cmake --build --preset "${PRESET}" --target bench_momentum
)

# createStudies.py hard-codes its resolutions, so reuse its helpers with our own range
suite() {
    "${PYTHON}" - "$@" <<EOF
import sys
from pathlib import Path
sys.path.insert(0, "${SUITE}")
import createStudies as cs
from foamlib import FoamFile
from foamlib.preprocessing.parameter_study import record_generator

mode, target = sys.argv[1], Path(sys.argv[2])
if mode == "generate":
    res = [int(r) for r in "${RES}".split()]
    (target / "momentum").mkdir(parents=True, exist_ok=True)
    study = record_generator(
        records=cs.build_records("3DCube", res),
        template_case=Path("${SUITE}") / "templates/3DCube",
        output_folder=target / "momentum" / "Cases",
    )
    study.create_study(study_base_folder=target / "momentum")
    # the template only defines solvers for T, but the momentum benchmark looks up U
    for case in (target / "momentum" / "Cases").iterdir():
        fv_solution = FoamFile(case / "system" / "fvSolution")
        for field, src in (("U", "T"), ("nfU", "nfT")):
            if field not in fv_solution["solvers"]:
                fv_solution["solvers", field] = fv_solution["solvers", src]
    cs.prepare_case(target, "momentum")
else:
    cs.execute_case("${BENCH}", target, "momentum")
    cs.gather_results(target, "momentum")
EOF
}

rm -rf "${WORK_DIR}"
suite generate "${WORK_DIR}"

for t in ${THREADS}; do
    echo ">>> ${NEON_DEVICE}: ${t} threads"
    export OMP_NUM_THREADS="${t}" KOKKOS_NUM_THREADS="${t}" OMP_PROC_BIND=spread OMP_PLACES=cores
    find "${WORK_DIR}" -name stats.xml -delete
    suite execute "${WORK_DIR}"
    cp "${WORK_DIR}/momentum/results/momentum.csv" "${RESULTS_DIR}/momentum_${NEON_DEVICE}.${t}.csv"
done

echo ">>> results in ${RESULTS_DIR}"
