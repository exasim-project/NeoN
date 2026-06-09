#!/usr/bin/env bash
#----------------------------------------------------------------------------------------
# SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense
#----------------------------------------------------------------------------------------
# Weak- and strong-scaling distributed benchmarks across a range of MPI rank counts.
# The script runs all sizes inside a single SLURM allocation so that only one queue
# slot is consumed.  Submit with enough tasks for the largest rank count:
#
#   sbatch --ntasks=24 --ntasks-per-node=1 --gpus-per-node=1 \
#          --time=02:00:00 --partition=<partition> \
#          scripts/distBenchSlurm.sh
#
# Environment variables (all optional, shown with defaults):
#   NEON_PRESET      build preset directory name           (profiling)
#   NEON_MPI_SIZES   space-separated rank counts to sweep  (1 2 4 8 12 16 20 24)
#   NEON_DEVICE      label appended to output filenames    (unknown)
#   RESULTS_DIR      directory for XML / JSON output       (dist_bench_results)
#   NEON_REPO_ROOT   path to the NeoN source tree          (directory of this script)
#   BENCH_SUITE      weak, strong, or both                 (both)
#   NEON_PERF_CORES  CPU range for performance cores only  (all cores)
#                    e.g. "0-15" for M2 Ultra / Intel 12th-gen 16P+8E
#----------------------------------------------------------------------------------------
#SBATCH --job-name=neon-dist-bench
#SBATCH --output=dist_bench_%j.log
#SBATCH --error=dist_bench_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${NEON_REPO_ROOT:-${SCRIPT_DIR}/..}"

PRESET="${NEON_PRESET:-profiling}"
BIN_DIR="${REPO_ROOT}/build/${PRESET}/bin/benchmarks"
RESULTS_DIR="${RESULTS_DIR:-dist_bench_results}"
NEON_DEVICE="${NEON_DEVICE:-unknown}"
EXECUTOR="${EXECUTOR:-Serial}"
BENCH_SUITE="${BENCH_SUITE:-both}"

MPI_SIZES="${NEON_MPI_SIZES:-1 2 4 8 12 16 20 24}"
# Seconds before a single mpirun invocation is killed (0 = no limit).
BENCH_TIMEOUT="${BENCH_TIMEOUT:-300}"

# CPU set for performance cores only.  Leave empty to use all cores.
# Override via NEON_PERF_CORES="0-15" (or any hwloc/taskset-compatible range).
# When unset the script auto-detects: sysctl on macOS, hwloc-calc on Linux.
detect_perf_cores() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        # hw.perflevel0 = P-cores on Apple Silicon; their logical IDs start at 0.
        local n
        n=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null)
        [[ -n "${n}" && "${n}" -gt 0 ]] && echo "0-$((n - 1))"
    elif command -v hwloc-calc &>/dev/null; then
        # hwloc 2.5+ knows CoreKind on Intel hybrid; timeout guards against hangs.
        timeout 5 hwloc-calc -I PU CoreKind=performance:all 2>/dev/null
    fi
}
PERF_CORES="${NEON_PERF_CORES:-$(detect_perf_cores)}"

# ---------------------------------------------------------------------------
# Select binaries based on BENCH_SUITE
# ---------------------------------------------------------------------------
BINARIES=()
case "${BENCH_SUITE}" in
  weak)   BINARIES=("bench_distributedweakScaling") ;;
  strong) BINARIES=("bench_distributedstrongScaling") ;;
  both)   BINARIES=("bench_distributedweakScaling" "bench_distributedstrongScaling") ;;
  *)
    echo "ERROR: BENCH_SUITE must be 'weak', 'strong', or 'both' (got '${BENCH_SUITE}')."
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
for BIN in "${BINARIES[@]}"; do
    if [[ ! -x "${BIN_DIR}/${BIN}" ]]; then
        echo "ERROR: binary not found or not executable: ${BIN_DIR}/${BIN}"
        echo "       Build with: cmake --preset ${PRESET} && cmake --build --preset ${PRESET}"
        exit 1
    fi
done

MAX_NP=$(echo "${MPI_SIZES}" | tr ' ' '\n' | sort -n | tail -1)
if [[ -n "${SLURM_NTASKS:-}" && "${SLURM_NTASKS}" -lt "${MAX_NP}" ]]; then
    echo "ERROR: SLURM_NTASKS=${SLURM_NTASKS} < largest requested rank count ${MAX_NP}."
    echo "       Re-submit with --ntasks=${MAX_NP} or reduce NEON_MPI_SIZES."
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

echo "=== NeoN distributed benchmark ==="
echo "    preset    : ${PRESET}"
echo "    suite     : ${BENCH_SUITE}"
echo "    binaries  : ${BINARIES[*]}"
echo "    MPI sizes : ${MPI_SIZES}"
echo "    device    : ${NEON_DEVICE}"
echo "    results   : ${RESULTS_DIR}"
echo "    executor  : ${EXECUTOR}"
echo "    perf cores: ${PERF_CORES:-all}"
echo ""

# ---------------------------------------------------------------------------
# MPI invocation helpers
# ---------------------------------------------------------------------------
BENCH_FLAGS="--benchmark-samples 10 --benchmark-warmup-time 0"

run_mpi() {
    local np="$1"; shift
    local binary="$1"; shift
    if [[ -n "${SLURM_JOB_ID:-}" && -z "${FORCE_MPIRUN:-}" ]]; then
        local cpu_bind=""
        [[ -n "${PERF_CORES}" ]] && cpu_bind="--cpu-bind=map_cpu:${PERF_CORES}"
        srun --ntasks="${np}" ${cpu_bind} "${binary}" ${BENCH_FLAGS} -r xml "$@"
    else
        local mca_flags=""
        local cpu_set=""
        local map_by="core"
        if [[ "$(uname -s)" == "Darwin" ]]; then
            mca_flags="--mca btl_vader_single_copy_mechanism none"
            # macOS hwloc does not support core binding or topology enumeration reliably.
            map_by="slot"
        else
            # Suppress the munge psec component: outside a SLURM allocation munge
            # is not available, and PMIx will error trying to open it.
            mca_flags="--mca psec ^munge"
            [[ -n "${PERF_CORES}" ]] && cpu_set="--bind-to core --cpu-set ${PERF_CORES}"
        fi
        local timeout_prefix=""
        [[ "${BENCH_TIMEOUT}" -gt 0 ]] && timeout_prefix="timeout ${BENCH_TIMEOUT}"
        ${timeout_prefix} mpirun ${mca_flags} ${cpu_set} --map-by "${map_by}" -np "${np}" "${binary}" ${BENCH_FLAGS} -r xml "$@"
    fi
}

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
for BIN in "${BINARIES[@]}"; do
    SUITE_TAG="${BIN#bench_distributed}"   # WeakScaling or StrongScaling
    SUITE_TAG="${SUITE_TAG,}"              # weakScaling or strongScaling (bash 4+)
    echo "--- ${BIN} ---"
    for NP in ${MPI_SIZES}; do
        OUTPUT="${RESULTS_DIR}/${SUITE_TAG}_${NP}ranks_${NEON_DEVICE}.xml"
        echo -n ">>> ${NP} rank(s) -> ${OUTPUT} ... "
        t0=$(date +%s)
        run_mpi "${NP}" "${BIN_DIR}/${BIN}" > "${OUTPUT}"
        echo "$(($(date +%s) - t0))s"
    done
    echo ""
done

# ---------------------------------------------------------------------------
# Convert XML -> JSON for downstream analysis / plotting
# ---------------------------------------------------------------------------
echo ">>> Converting XML to JSON..."
pushd "${RESULTS_DIR}" > /dev/null
python3 "${SCRIPT_DIR}/catch2json.py"
popd > /dev/null

echo ""
echo "=== Done. Results written to ${RESULTS_DIR}/ ==="
