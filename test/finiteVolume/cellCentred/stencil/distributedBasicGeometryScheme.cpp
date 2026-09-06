// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoN
{

// Distributed processor-boundary regression test for BasicGeometryScheme.
//
// The geometry-scheme producer kernels must write processor-boundary entries
// (weights, nonOrthDeltaCoeffs, nonOrthCorrectionVec3s); left at zero, every
// diffusive flux across a rank boundary silently vanishes. This test partitions a
// 1D uniform mesh across ranks and asserts the processor-boundary values directly
// against their analytical values.
//
// CPUExecutor only: a multi-rank test must not place data on the GPU, since on a
// single-GPU machine the ranks would contend for the one device.
TEST_CASE("BasicGeometryScheme processor-boundary values")
{
    mpi::Environment mpiEnviron;
    const Executor exec = CPUExecutor {};

    const localIdx nLocal = 4;
    auto mesh = create1DUniformMeshPart(exec, nLocal);

    fvcc::GeometryScheme scheme(exec, mesh, std::make_unique<fvcc::BasicGeometryScheme>(mesh));

    const auto nBF = mesh.nBoundaryFaces();
    const auto nPF = mesh.nProcBoundaryFaces();
    REQUIRE(nPF > 0); // every rank in a 1D partition abuts at least one neighbour

    // Global domain [0, 1] is split into sizeRank slabs of nLocal cells, so the uniform
    // cell spacing is h = 1 / (sizeRank * nLocal) and the proc faces are equidistant.
    const scalar invH = static_cast<scalar>(mpiEnviron.sizeRank() * nLocal);

    auto wB = scheme.weights().boundaryData().value().copyToHost();
    auto ndcB = scheme.nonOrthDeltaCoeffs().boundaryData().value().copyToHost();
    auto cvB = scheme.nonOrthCorrectionVec3s().boundaryData().value().copyToHost();
    const auto wBv = wB.view();
    const auto ndcBv = ndcB.view();
    const auto cvBv = cvB.view();

    for (localIdx i = nBF; i < nBF + nPF; ++i)
    {
        INFO("processor-boundary face index " << i);
        REQUIRE(wBv[i] == Catch::Approx(0.5).margin(1e-12));    // equidistant neighbour
        REQUIRE(ndcBv[i] == Catch::Approx(invH).margin(1e-10)); // 1/(n.d) == 1/h (was 0 pre-fix)
        REQUIRE(mag(cvBv[i]) < 1e-12);                          // orthogonal -> no correction
    }
}
}
