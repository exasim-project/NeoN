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
// known, so a regression in any of the producer kernels (weights, deltaCoeffs,
// nonOrthDeltaCoeffs, nonOrthCorrectionVec3s) fails loudly. Covers review
// findings H2 (doc value), H3 (no 1/0), M3/M4 (corrVec formula), N1 (deltaCoeffs
// is produced and asserted).
TEST_CASE("BasicGeometryScheme analytical 3D cube")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    const localIdx n = 4;
    const scalar L = 1.0;
    auto mesh = create3DUniformMesh(exec, n, n, n, L, L, L);
    const scalar h = L / static_cast<scalar>(n);
    const scalar invH = 1.0 / h;             // owner-to-neighbour: 1/h
    const scalar invHalfH = 1.0 / (0.5 * h); // cell-centre-to-boundary-face: 2/h

    fvcc::GeometryScheme scheme(exec, mesh, std::make_unique<fvcc::BasicGeometryScheme>(mesh));

    // --- internal faces ---------------------------------------------------
    {
        auto w = scheme.weights().internalVector().copyToHost();
        const auto wv = w.view();
        for (localIdx i = 0; i < w.size(); ++i)
            REQUIRE(wv[i] == Catch::Approx(0.5).margin(1e-14)); // symmetry

        auto dc = scheme.deltaCoeffs().internalVector().copyToHost();
        const auto dcv = dc.view();
        for (localIdx i = 0; i < dc.size(); ++i)
            REQUIRE(dcv[i] == Catch::Approx(invH).margin(1e-12)); // 1/|d|

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

        auto dcB = scheme.deltaCoeffs().boundaryData().value().copyToHost();
        const auto dcBv = dcB.view();
        for (localIdx i = 0; i < nBF; ++i)
            REQUIRE(dcBv[i] == Catch::Approx(invHalfH).margin(1e-12));

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

// Non-orthogonal mesh (review T3 / N2): the orthogonal deltaCoeffs (1/|d|) and the
// over-relaxed nonOrthDeltaCoeffs (1/(n.d)) must genuinely diverge, and the non-orth
// correction must be non-zero. nonOrthDeltaCoeffs is what every snGrad scheme uses to
// match OpenFOAM, so this pins that the producer actually distinguishes the two off the
// orthogonal limit. A sheared cube introduces the non-orthogonality on x-normal faces
// while leaving y/z faces orthogonal.
TEST_CASE("BasicGeometryScheme on a sheared (non-orthogonal) mesh")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    const localIdx n = 4;
    auto mesh = create3DUniformMesh(exec, n, n, n);

    // Shear cell centres: C.y += s * C.x. The cell-to-cell vector across an x-normal
    // face gains a transverse y-component (n.d < |d|); face geometry is untouched.
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

    auto dc = scheme.deltaCoeffs().internalVector().copyToHost();
    auto ndc = scheme.nonOrthDeltaCoeffs().internalVector().copyToHost();
    auto cv = scheme.nonOrthCorrectionVec3s().internalVector().copyToHost();
    const auto dcv = dc.view();
    const auto ndcv = ndc.view();
    const auto cvv = cv.view();

    scalar maxDiff = 0.0;
    scalar maxCorr = 0.0;
    for (localIdx i = 0; i < dc.size(); ++i)
    {
        // mathematical invariant: 1/(n.d) >= 1/|d| since the projection <= magnitude
        REQUIRE(ndcv[i] >= dcv[i] - 1e-12);
        const scalar diff = ndcv[i] - dcv[i];
        if (diff > maxDiff) maxDiff = diff;
        const scalar c = mag(cvv[i]);
        if (c > maxCorr) maxCorr = c;
    }
    // the sheared x-faces make the two delta fields genuinely diverge ...
    REQUIRE(maxDiff > 1e-3);
    // ... and produce a non-zero non-orthogonal correction vector
    REQUIRE(maxCorr > 1e-3);
}

// Pins the N1/N2 wiring (review T5, corrected): 'uncorrected' must expose
// nonOrthDeltaCoeffs (1/(n.d)), matching OpenFOAM's uncorrectedSnGrad which returns
// mesh().nonOrthDeltaCoeffs(). The review's original N2 premise (that OF uses the
// orthogonal 1/|d|) was verified false against the OF source, so this asserts the
// opposite identity from the pre-revert version. The orthogonal deltaCoeffs is kept
// on the GeometryScheme for API completeness but has no snGrad consumer, so it must
// be a distinct field from what 'uncorrected' exposes. Checked by field identity, so
// it holds on any mesh and fails loudly if anyone rewires the accessor.
TEST_CASE("uncorrected snGrad exposes nonOrthDeltaCoeffs (N1/N2)")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    auto mesh = create3DUniformMesh(exec, 3, 3, 3);
    auto scheme = fvcc::GeometryScheme::readOrCreate(mesh);
    fvcc::Uncorrected<scalar> uncorrected(exec, mesh);

    REQUIRE(&uncorrected.deltaCoeffs() == &scheme->nonOrthDeltaCoeffs());
    REQUIRE(&uncorrected.deltaCoeffs() != &scheme->deltaCoeffs());
}
}
