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


// The scalar gradient sums each cell's internal faces without atomics, using a cell-to-face
// stencil cached on the mesh. It must match the face-based Gauss sum, with a per-cell operator
// scaling, on a 3D mesh where every cell has several internal faces.
TEST_CASE("gaussGreenGrad cell-based matches face-based sum")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    auto mesh = create3DUniformMesh(exec, 4, 3, 2, 1.0, 1.0, 1.0);
    auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> phi(exec, "phi", mesh, bcs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    std::vector<scalar> scaling(static_cast<sizeType>(mesh.nCells()));
    for (sizeType i = 0; i < scaling.size(); ++i)
    {
        scaling[i] = 1.0 + 0.25 * static_cast<scalar>(i % 5);
    }
    const Vector<scalar> scalingField(exec, scaling); // dsl::Coeff only views the field
    const dsl::Coeff coeff(scalingField);

    // face-based reference on the host
    fvcc::SurfaceField<scalar> phif(
        exec, "phif", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh)
    );
    fvcc::Linear<scalar>(exec, mesh, Dictionary()).interpolate(phi, phif);
    auto [own, nei, sf, bOwn, bSf, vol, phifI, phifB] = copyToHosts(
        mesh.faceOwners(),
        mesh.faceNeighbors(),
        mesh.faceNormals(),
        mesh.boundaryMesh().faceOwners(),
        mesh.boundaryMesh().faceNormals(),
        mesh.cellVolumes(),
        phif.internalVector(),
        phif.boundaryData().value()
    );
    std::vector<Vec3> expected(static_cast<sizeType>(mesh.nCells()), zero<Vec3>());
    for (localIdx f = 0; f < mesh.nInternalFaces(); ++f)
    {
        const Vec3 flux = sf.view()[f] * phifI.view()[f];
        expected[static_cast<sizeType>(own.view()[f])] += flux;
        expected[static_cast<sizeType>(nei.view()[f])] -= flux;
    }
    for (localIdx f = 0; f < mesh.nBoundaryFaces(); ++f)
    {
        expected[static_cast<sizeType>(bOwn.view()[f])] += bSf.view()[f] * phifB.view()[f];
    }
    for (sizeType c = 0; c < expected.size(); ++c)
    {
        expected[c] *= scaling[c] / vol.view()[static_cast<localIdx>(c)];
    }
    const Vector<Vec3> expectedField(exec, expected);

    fvcc::GaussGreenGrad gradOp(exec, mesh);
    REQUIRE_FALSE(mesh.stencilDB().contains("GaussGreenGradCellFaces"));
    for (int call = 0; call < 2; ++call) // the second call reuses the cached stencil
    {
        Vector<Vec3> out(exec, mesh.nCells(), zero<Vec3>());
        gradOp.grad(phi, coeff, out);
        REQUIRE(mesh.stencilDB().contains("GaussGreenGradCellFaces"));
        REQUIRE_THAT(out, Equals(expectedField, Approx {1e-12}));
    }
}

}
