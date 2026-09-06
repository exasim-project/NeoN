// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoN
{

// Analytical regression test for BasicGeometryScheme on a non-1D orthogonal mesh.
//
// Every direct stencil test in the suite runs on create1DUniformMesh, the unique
// mesh on which most geometric-scheme bugs are invisible. This test pins the
// producer values on a 3D uniform hex mesh where every quantity is analytically
// known, so a regression in any of the producer kernels (weights,
// nonOrthDeltaCoeffs, nonOrthCorrectionVec3s) fails loudly.
TEST_CASE("BasicGeometryScheme analytical 3D cube")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    const localIdx n = 4;
    const scalar l = 1.0;
    auto mesh = create3DUniformMesh(exec, n, n, n, l, l, l);
    const scalar h = l / static_cast<scalar>(n);
    const scalar invH = 1.0 / h;             // owner-to-neighbour: 1/h
    const scalar invHalfH = 1.0 / (0.5 * h); // cell-centre-to-boundary-face: 2/h

    fvcc::GeometryScheme scheme(exec, mesh, std::make_unique<fvcc::BasicGeometryScheme>(mesh));

    // --- internal faces ---------------------------------------------------
    {
        auto w = scheme.weights().internalVector().copyToHost();
        const auto wv = w.view();
        for (localIdx i = 0; i < w.size(); ++i)
            REQUIRE(wv[i] == Catch::Approx(0.5).margin(1e-14)); // symmetry

        auto ndc = scheme.nonOrthDeltaCoeffs().internalVector().copyToHost();
        const auto ndcv = ndc.view();
        for (localIdx i = 0; i < ndc.size(); ++i)
            REQUIRE(ndcv[i] == Catch::Approx(invH).margin(1e-12)); // 1/(n.d) == 1/|d| orthogonal

        auto cv = scheme.nonOrthCorrectionVec3s().internalVector().copyToHost();
        const auto cvv = cv.view();
        for (localIdx i = 0; i < cv.size(); ++i)
            REQUIRE(mag(cvv[i]) < 1e-14); // zero correction on an orthogonal mesh
    }

    // --- (non-processor) boundary faces -----------------------------------
    {
        const auto nBF = mesh.nBoundaryFaces();

        auto wB = scheme.weights().boundaryData().value().copyToHost();
        const auto wBv = wB.view();
        for (localIdx i = 0; i < nBF; ++i)
            REQUIRE(wBv[i] == Catch::Approx(1.0).margin(1e-14));

        auto ndcB = scheme.nonOrthDeltaCoeffs().boundaryData().value().copyToHost();
        const auto ndcBv = ndcB.view();
        for (localIdx i = 0; i < nBF; ++i)
            REQUIRE(ndcBv[i] == Catch::Approx(invHalfH).margin(1e-12));

        auto cvB = scheme.nonOrthCorrectionVec3s().boundaryData().value().copyToHost();
        const auto cvBv = cvB.view();
        for (localIdx i = 0; i < nBF; ++i)
            REQUIRE(mag(cvBv[i]) < 1e-14);
    }
}

// Non-orthogonal mesh: nonOrthDeltaCoeffs uses the face-normal
// projection 1/(n.d), so a transverse shear (which grows |d| but leaves n.d unchanged)
// must NOT change it — it would drop below 1/h if the kernel used the Euclidean 1/|d|.
// The non-orthogonality must instead surface in the correction vector. nonOrthDeltaCoeffs
// is what every snGrad scheme uses to match OpenFOAM, so this pins the projection
// behaviour. A sheared cube introduces the non-orthogonality on x-normal faces while
// leaving y/z faces orthogonal.
TEST_CASE("BasicGeometryScheme on a sheared (non-orthogonal) mesh")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    const localIdx n = 4;
    auto mesh = create3DUniformMesh(exec, n, n, n);
    const scalar invH = static_cast<scalar>(n); // h = 1/n, so 1/h = n

    // Shear cell centres: C.y += s * C.x. The cell-to-cell vector across an x-normal
    // face gains a transverse y-component (|d| grows) but its face-normal projection
    // n.d is unchanged; face geometry is untouched.
    const scalar s = 0.5;
    {
        auto ccH = mesh.cellCenters().copyToHost();
        auto v = ccH.view();
        for (localIdx i = 0; i < ccH.size(); ++i)
        {
            const Vec3 p = v[i];
            v[i] = Vec3 {p[0], p[1] + s * p[0], p[2]};
        }
        mesh.cellCenters() = ccH.copyToExecutor(exec);
    }

    fvcc::GeometryScheme scheme(exec, mesh, std::make_unique<fvcc::BasicGeometryScheme>(mesh));

    auto ndc = scheme.nonOrthDeltaCoeffs().internalVector().copyToHost();
    auto cv = scheme.nonOrthCorrectionVec3s().internalVector().copyToHost();
    const auto ndcv = ndc.view();
    const auto cvv = cv.view();

    scalar maxCorr = 0.0;
    for (localIdx i = 0; i < ndc.size(); ++i)
    {
        // 1/(n.d) stays at 1/h under a transverse shear (n.d == h); it would be smaller
        // if the kernel used the Euclidean 1/|d|.
        REQUIRE(ndcv[i] == Catch::Approx(invH).margin(1e-12));
        const scalar c = mag(cvv[i]);
        if (c > maxCorr) maxCorr = c;
    }
    // the sheared x-faces produce a non-zero non-orthogonal correction vector
    REQUIRE(maxCorr > 1e-3);
}

// 'uncorrected' must expose nonOrthDeltaCoeffs (1/(n.d)), matching OpenFOAM's
// uncorrectedSnGrad which returns mesh().nonOrthDeltaCoeffs() — NOT the Euclidean
// 1/|d| deltaCoeffs. Checked by field identity, so it holds on any mesh and fails
// loudly if anyone rewires the accessor.
TEST_CASE("uncorrected snGrad exposes nonOrthDeltaCoeffs")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    auto mesh = create3DUniformMesh(exec, 3, 3, 3);
    auto scheme = fvcc::GeometryScheme::readOrCreate(mesh);
    fvcc::Uncorrected<scalar> uncorrected(exec, mesh);

    REQUIRE(&uncorrected.deltaCoeffs() == &scheme->nonOrthDeltaCoeffs());
}
}
