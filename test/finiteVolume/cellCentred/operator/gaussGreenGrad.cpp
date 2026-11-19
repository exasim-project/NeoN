// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoN
{

// The boundary value of grad(U) on a symmetry plane, as OpenFOAM computes it: the gradient is
// itself a symmetry field, so the cell gradient is reflected, (g + R g R)/2 with R = I - 2 n n, and
// its normal derivative is then replaced by the patch's snGrad, deltaCoeffs*(value - cell value).
// With the default implicit treatment the symmetry patch leaves refGrad at zero, so the snGrad
// must come from the patch value, not from refGrad.
TEST_CASE("gaussGreenGrad tensor gradient on symmetry planes")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    const localIdx n = 3;
    const scalar l = 1.0;
    auto mesh = create3DUniformMesh(exec, n, n, n, l, l, l);

    std::vector<fvcc::VolumeBoundary<Vec3>> bcs {};
    for (localIdx patchi = 0; patchi < toLocalIdx(mesh.nBoundaries()); ++patchi)
    {
        Dictionary dict;
        dict.insert("type", std::string("symmetry"));
        bcs.push_back(fvcc::VolumeBoundary<Vec3>(mesh, dict, patchi));
    }

    // A linear field in which every component varies in every direction, so the cell gradient
    // has no zero entries and both the reflection and the snGrad correction change it.
    auto ccH = mesh.cellCenters().copyToHost();
    std::vector<Vec3> values(static_cast<sizeType>(mesh.nCells()));
    for (localIdx i = 0; i < mesh.nCells(); ++i)
    {
        const Vec3 p = ccH.view()[i];
        values[static_cast<sizeType>(i)] = Vec3 {
            1.5 * p[0] - 2.0 * p[1] + 0.75 * p[2] + 0.3,
            -0.5 * p[0] + 1.25 * p[1] + 0.5 * p[2] - 0.2,
            0.25 * p[0] + 0.5 * p[1] - 1.75 * p[2] + 0.1
        };
    }
    fvcc::VolumeField<Vec3> u(exec, "U", mesh, Vector<Vec3>(exec, values), bcs);
    u.correctBoundaryConditions();

    fvcc::GaussGreenGrad gradOp(exec, mesh);
    const auto gradU = gradOp.gradTensor(u);

    auto [gB, uB, uInt, faceCells, deltaCoeffs, normals] = copyToHosts(
        gradU.boundaryData().value(),
        u.boundaryData().value(),
        u.internalVector(),
        mesh.boundaryMesh().faceOwners(),
        mesh.boundaryMesh().deltaCoeffs(),
        mesh.boundaryMesh().faceUnitNormals()
    );

    REQUIRE(gB.size() > 0);
    for (localIdx i = 0; i < gB.size(); ++i)
    {
        INFO("boundary face " << i);
        const Tensor g = gB.view()[i];
        const Vec3 nf = normals.view()[i];
        const auto owner = faceCells.view()[i];
        const Vec3 snGrad = (uB.view()[i] - uInt.view()[owner]) * deltaCoeffs.view()[i];

        // The faces of the uniform mesh are axis aligned: k is the normal direction.
        sizeType k = 0;
        for (sizeType d = 1; d < 3; ++d)
        {
            if (std::abs(nf[d]) > std::abs(nf[k])) k = d;
        }
        REQUIRE(std::abs(nf[k]) == Catch::Approx(1.0));

        for (sizeType row = 0; row < 3; ++row)
        {
            // The normal derivative of every component is the patch's snGrad.
            const scalar nDotG = nf[0] * g(row, 0) + nf[1] * g(row, 1) + nf[2] * g(row, 2);
            REQUIRE(nDotG == Catch::Approx(snGrad[row]).margin(1e-10));
        }
        // The normal component of U is removed on the plane, so its normal derivative is not zero
        // here; refGrad would have given zero.
        REQUIRE(std::abs(snGrad[k]) > 1e-6);

        // The reflection removes the tangential derivatives of the normal component.
        for (sizeType col = 0; col < 3; ++col)
        {
            if (col == k) continue;
            REQUIRE(g(k, col) == Catch::Approx(0.0).margin(1e-10));
        }
    }
}

// gradVolIntegrated accumulates operatorScaling * V * grad(phi), i.e. the explicit gradient before
// the division by the cell volume, into its target. It must equal grad() times V cell by cell,
// with a per-cell operator scaling, and add to what the target already holds.
TEST_CASE("gaussGreenGrad volume-integrated gradient")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    const localIdx n = 3;
    auto mesh = create3DUniformMesh(exec, n, n, n, 2.0, 1.0, 0.5);
    const auto nCells = mesh.nCells();

    auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    auto ccH = mesh.cellCenters().copyToHost();
    std::vector<scalar> phiValues(static_cast<sizeType>(nCells));
    std::vector<scalar> scaling(static_cast<sizeType>(nCells));
    for (localIdx i = 0; i < nCells; ++i)
    {
        const Vec3 p = ccH.view()[i];
        phiValues[static_cast<sizeType>(i)] = 1.5 * p[0] * p[0] - 2.0 * p[1] + 0.75 * p[2] + 0.3;
        scaling[static_cast<sizeType>(i)] = 1.0 + 0.1 * static_cast<scalar>(i);
    }
    fvcc::VolumeField<scalar> phi(exec, "phi", mesh, Vector<scalar>(exec, phiValues), bcs);
    phi.correctBoundaryConditions();
    const Vector<scalar> scalingField(exec, scaling); // dsl::Coeff only views the field
    const dsl::Coeff coeff(scalingField);

    fvcc::GaussGreenGrad gradOp(exec, mesh);

    Vector<Vec3> grad(exec, nCells, Vec3(0.0, 0.0, 0.0));
    gradOp.grad(phi, coeff, grad);

    const Vec3 offset(1.0, -2.0, 3.0);
    Vector<Vec3> volIntegrated(exec, nCells, offset);
    gradOp.gradVolIntegrated(phi, coeff, volIntegrated);

    auto [gradH, volIntegratedH, volH] = copyToHosts(grad, volIntegrated, mesh.cellVolumes());
    for (localIdx i = 0; i < nCells; ++i)
    {
        INFO("cell " << i);
        const Vec3 expected = gradH.view()[i] * volH.view()[i];
        REQUIRE(std::abs(expected[0]) + std::abs(expected[1]) > 1e-6);
        for (sizeType d = 0; d < 3; ++d)
        {
            REQUIRE(
                volIntegratedH.view()[i][d] - offset[d] == Catch::Approx(expected[d]).margin(1e-12)
            );
        }
    }
}

// GaussGreenGrad reuses one face-interpolation buffer across calls; a second gradient of a
// different field must not see values left from the first.
TEST_CASE("gaussGreenGrad reuses its face buffer without stale values")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    auto mesh = create3DUniformMesh(exec, 3, 3, 3, 1.0, 1.0, 1.0);
    auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    auto ccH = mesh.cellCenters().copyToHost();
    std::vector<scalar> a(static_cast<sizeType>(mesh.nCells()));
    std::vector<scalar> b(static_cast<sizeType>(mesh.nCells()));
    for (localIdx i = 0; i < mesh.nCells(); ++i)
    {
        const Vec3 p = ccH.view()[i];
        a[static_cast<sizeType>(i)] = 3.0 * p[0] + p[1];
        b[static_cast<sizeType>(i)] = -p[1] + 2.0 * p[2] * p[2];
    }
    fvcc::VolumeField<scalar> phiA(exec, "a", mesh, Vector<scalar>(exec, a), bcs);
    fvcc::VolumeField<scalar> phiB(exec, "b", mesh, Vector<scalar>(exec, b), bcs);
    phiA.correctBoundaryConditions();
    phiB.correctBoundaryConditions();

    fvcc::GaussGreenGrad reused(exec, mesh);
    auto first = reused.grad(phiA);
    auto second = reused.grad(phiB);
    auto fresh = fvcc::GaussGreenGrad(exec, mesh).grad(phiB);

    auto [secondH, freshH] = copyToHosts(second.internalVector(), fresh.internalVector());
    for (localIdx i = 0; i < mesh.nCells(); ++i)
    {
        for (sizeType d = 0; d < 3; ++d)
        {
            REQUIRE(secondH.view()[i][d] == Catch::Approx(freshH.view()[i][d]).margin(1e-12));
        }
    }
}

}
