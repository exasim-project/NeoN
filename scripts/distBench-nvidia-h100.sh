#!/bin/bash
# SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

module purge
module load gcc/13.3.0
module load cuda/12.8.1
module load cmake
module load openmpi

export NEON_DEVICE=nvidia_h100

PRESET="${NEON_PRESET:-profiling}"
BINARY="build/${PRESET}/bin/benchmarks/bench_distributedLaplacianOperator"
RESULTS_DIR="${RESULTS_DIR:-.}"

# Square rank counts for the 2D benchmark (ranksXPart^2)
MPI_SIZES="${NEON_MPI_SIZES:-1 4 9}"

mkdir -p "${RESULTS_DIR}"

for NP in ${MPI_SIZES}; do
    echo ">>> Running distributedLaplacianOperator with ${NP} rank(s)..."
    mpirun -np "${NP}" "${BINARY}" -r xml > "${RESULTS_DIR}/dist_bench_${NP}ranks.xml"
done

pushd "${RESULTS_DIR}" > /dev/null
python3 "$(dirname "$0")/catch2json.py"
popd > /dev/null
