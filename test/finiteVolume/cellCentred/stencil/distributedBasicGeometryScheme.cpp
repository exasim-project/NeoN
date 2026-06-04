// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoN
{

// Distributed proc-boundary regression test for BasicGeometryScheme (review T2 / H1).
//
// Before the remediation, updateNonOrthDeltaCoeffs / updateNonOrthCorrectionVec3s never wrote
// processor-boundary entries, leaving them at zero on every MPI run. This test partitions
// a 1D uniform mesh across ranks and asserts the proc-boundary values directly — it fails
// against the pre-fix code (nonOrthDeltaCoeffs == 0 at proc faces).
//
// CPUExecutor only: this machine has a single GPU, so a multi-rank test must not place data
// on the GPU (ranks would contend for the one device). See feedback memory.
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
