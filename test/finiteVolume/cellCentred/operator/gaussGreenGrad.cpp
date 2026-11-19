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

// per-cell operator scaling 1, 1.25, ..., 2 repeating, so a missing or misplaced scaling shows
std::vector<scalar> cellScaling(localIdx nCells)
{
    std::vector<scalar> scaling(static_cast<sizeType>(nCells));
    for (sizeType i = 0; i < scaling.size(); ++i)
    {
        scaling[i] = 1.0 + 0.25 * static_cast<scalar>(i % 5);
    }
    return scaling;
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

    const auto scaling = cellScaling(mesh.nCells());
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

// The tensor gradient grad(U)(row,col) = sum_f Sf[col] U_f[row] / V, summed cell by cell like the
// scalar one, must match the face-based sum, with the per-cell operator scaling.
TEST_CASE("gaussGreenGrad tensor cell-based matches face-based sum")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    auto mesh = create3DUniformMesh(exec, 4, 3, 2, 1.0, 1.0, 1.0);
    auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Vec3>>(mesh);
    fvcc::VolumeField<Vec3> u(exec, "U", mesh, bcs);
    std::vector<Vec3> uValues(static_cast<sizeType>(mesh.nCells()));
    for (sizeType c = 0; c < uValues.size(); ++c) // every component differs from cell to cell
    {
        const auto x = static_cast<scalar>(c);
        uValues[c] = Vec3(1.0 + 0.3 * x, 2.0 - 0.1 * x * x, 0.5 + 0.07 * x * x * x);
    }
    u.internalVector() = Vector<Vec3>(exec, uValues);
    u.correctBoundaryConditions();

    const auto scaling = cellScaling(mesh.nCells());
    const Vector<scalar> scalingField(exec, scaling);
    const dsl::Coeff coeff(scalingField);

    fvcc::SurfaceField<Vec3> uf(
        exec, "Uf", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<Vec3>>(mesh)
    );
    fvcc::Linear<Vec3>(exec, mesh, Dictionary()).interpolate(u, uf);
    auto [own, nei, sf, bOwn, bSf, vol, ufI, ufB] = copyToHosts(
        mesh.faceOwners(),
        mesh.faceNeighbors(),
        mesh.faceNormals(),
        mesh.boundaryMesh().faceOwners(),
        mesh.boundaryMesh().faceNormals(),
        mesh.cellVolumes(),
        uf.internalVector(),
        uf.boundaryData().value()
    );
    std::vector<Tensor> expected(static_cast<sizeType>(mesh.nCells()), zero<Tensor>());
    auto addFlux = [](Tensor& t, const Vec3& s, const Vec3& uFace, scalar sign)
    {
        for (sizeType row = 0; row < 3; ++row)
        {
            for (sizeType col = 0; col < 3; ++col)
            {
                t(row, col) += sign * s[col] * uFace[row];
            }
        }
    };
    for (localIdx f = 0; f < mesh.nInternalFaces(); ++f)
    {
        addFlux(expected[static_cast<sizeType>(own.view()[f])], sf.view()[f], ufI.view()[f], 1.0);
        addFlux(expected[static_cast<sizeType>(nei.view()[f])], sf.view()[f], ufI.view()[f], -1.0);
    }
    for (localIdx f = 0; f < mesh.nBoundaryFaces(); ++f)
    {
        addFlux(expected[static_cast<sizeType>(bOwn.view()[f])], bSf.view()[f], ufB.view()[f], 1.0);
    }
    for (sizeType c = 0; c < expected.size(); ++c)
    {
        expected[c] *= scaling[c] / vol.view()[static_cast<localIdx>(c)];
    }

    fvcc::GaussGreenGrad gradOp(exec, mesh);
    auto gradBcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Tensor>>(mesh);
    fvcc::VolumeField<Tensor> gradU(exec, "gradU", mesh, gradBcs);
    for (int call = 0; call < 2; ++call) // the second call reuses the cached stencil
    {
        gradOp.gradTensor(u, gradU, coeff);
        REQUIRE(mesh.stencilDB().contains("GaussGreenGradCellFaces"));
        auto result = gradU.internalVector().copyToHost();
        for (sizeType c = 0; c < expected.size(); ++c)
        {
            for (sizeType row = 0; row < 3; ++row)
            {
                for (sizeType col = 0; col < 3; ++col)
                {
                    INFO("cell " << c << " (" << row << "," << col << ")");
                    REQUIRE_THAT(
                        result.view()[static_cast<localIdx>(c)](row, col),
                        Catch::Matchers::WithinAbs(expected[c](row, col), 1e-11)
                    );
                }
            }
        }
    }
}

}
