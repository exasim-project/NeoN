// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#if NF_WITH_GINKGO

#include "NeoN/linearAlgebra/preconditioner/adicGinkgo.hpp"

namespace NeoN::la::ginkgo
{

#ifdef __CUDACC__

namespace
{

__global__ void adicGkoExtractDiagKernel(
    std::size_t n,
    const scalar* vals,
    const localIdx* col,
    const localIdx* row,
    scalar* diag,
    localIdx* diagPos
)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    auto dp = row[i];
    scalar d = scalar {0};
    for (auto k = row[i]; k < row[i + 1]; ++k)
    {
        if (static_cast<std::size_t>(col[k]) == i)
        {
            dp = k;
            d = vals[k];
            break;
        }
    }
    diagPos[i] = dp;
    diag[i] = d;
}

__global__ void adicGkoComputeRdKernel(
    std::size_t n,
    const scalar* vals,
    const localIdx* col,
    const localIdx* row,
    const localIdx* diagPos,
    const scalar* diag,
    scalar* rd
)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    scalar s = diag[i];
    for (auto k = row[i]; k < diagPos[i]; ++k)
    {
        s -= vals[k] * vals[k] / diag[static_cast<std::size_t>(col[k])];
    }
    rd[i] = scalar {1} / s;
}

__global__ void adicGkoDiagScaleKernel(std::size_t n, const scalar* rd, const scalar* b, scalar* x)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = rd[i] * b[i];
}

__global__ void adicGkoForwardKernel(
    std::size_t n,
    const scalar* vals,
    const localIdx* col,
    const localIdx* row,
    const localIdx* diagPos,
    const scalar* rd,
    const scalar* x,
    scalar* work
)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    scalar s = x[i];
    for (auto k = row[i]; k < diagPos[i]; ++k)
    {
        s -= rd[i] * vals[k] * x[static_cast<std::size_t>(col[k])];
    }
    work[i] = s;
}

__global__ void adicGkoBackwardKernel(
    std::size_t n,
    const scalar* vals,
    const localIdx* col,
    const localIdx* row,
    const localIdx* diagPos,
    const scalar* rd,
    scalar* x,
    const scalar* work
)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    scalar s = work[i];
    for (auto k = diagPos[i] + 1; k < row[i + 1]; ++k)
    {
        s -= rd[i] * vals[k] * work[static_cast<std::size_t>(col[k])];
    }
    x[i] = s;
}

constexpr int blockSize = 256;

unsigned gridSize(std::size_t n) { return static_cast<unsigned>((n + blockSize - 1) / blockSize); }

} // namespace

void adicGkoGenerateCuda(
    std::size_t n,
    const scalar* vals,
    const localIdx* col,
    const localIdx* row,
    scalar* diag,
    scalar* rd,
    localIdx* diagPos,
    CUstream_st* stream
)
{
    if (n == 0) return;
    const auto gs = gridSize(n);
    adicGkoExtractDiagKernel<<<gs, blockSize, 0, stream>>>(n, vals, col, row, diag, diagPos);
    adicGkoComputeRdKernel<<<gs, blockSize, 0, stream>>>(n, vals, col, row, diagPos, diag, rd);
}

void adicGkoApplyCuda(
    std::size_t n,
    const scalar* vals,
    const localIdx* col,
    const localIdx* row,
    const localIdx* diagPos,
    const scalar* rd,
    const scalar* b,
    scalar* x,
    scalar* work,
    CUstream_st* stream
)
{
    if (n == 0) return;
    const auto gs = gridSize(n);
    adicGkoDiagScaleKernel<<<gs, blockSize, 0, stream>>>(n, rd, b, x);
    adicGkoForwardKernel<<<gs, blockSize, 0, stream>>>(n, vals, col, row, diagPos, rd, x, work);
    adicGkoBackwardKernel<<<gs, blockSize, 0, stream>>>(n, vals, col, row, diagPos, rd, x, work);
}

#else // host build: launchers are never invoked (no CudaExecutor), provide no-op stubs.

void adicGkoGenerateCuda(std::size_t, const scalar*, const localIdx*, const localIdx*, scalar*, scalar*, localIdx*, CUstream_st*)
{}

void adicGkoApplyCuda(std::size_t, const scalar*, const localIdx*, const localIdx*, const localIdx*, const scalar*, const scalar*, scalar*, scalar*, CUstream_st*)
{}

#endif

} // namespace NeoN::la::ginkgo

#endif
