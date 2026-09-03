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

// reconstruct(ssf) inverts sum_f (Sf (x) Sf / |Sf|) against sum_f (Sf / |Sf|) ssf_f. For a flux
// ssf_f = U . Sf of a spatially uniform U the least-squares system is exact, so the reconstruction
// must return U back — that is the property these tests pin. A 1D uniform mesh is x-extruded with
// zero-area y/z directions, so it also exercises the empty-direction regularisation.
TEST_CASE("reconstruct")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("recovers a uniform velocity from its flux" + execName)
    {
        auto mesh = create1DUniformMesh(exec, 10);
        auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
        fvcc::SurfaceField<scalar> phi(exec, "phi", mesh, surfaceBCs);

        // U = (3, 0, 0); the mesh is x-normal, so phi_f = U . Sf = 3 * Sf_x on every face.
        const scalar ux = 3.0;
        auto [phiI, phiB] = views(phi.internalVector(), phi.boundaryData().value());
        const auto sf = mesh.faceNormals().view();
        const auto bSf = mesh.boundaryMesh().faceNormals().view();
        parallelFor(
            exec,
            {0, mesh.nInternalFaces()},
            NEON_LAMBDA(const localIdx f) { phiI[f] = ux * sf[f][0]; }
        );
        parallelFor(
            exec,
            {0, mesh.nBoundaryFaces()},
            NEON_LAMBDA(const localIdx bf) { phiB[bf] = ux * bSf[bf][0]; }
        );

        auto res = fvcc::reconstruct(phi);
        REQUIRE(res.size() == mesh.nCells());

        auto hostRes = res.internalVector().copyToHost();
        auto resView = hostRes.view();
        for (localIdx i = 0; i < hostRes.size(); ++i)
        {
            REQUIRE_THAT(resView[i][0], Catch::Matchers::WithinAbs(ux, 1e-12));
            // The y/z directions carry no face area on this mesh; the empty-direction
            // regularisation must keep them at ~0 rather than blowing up or zeroing the
            // whole cell by a singular inverse.
            REQUIRE_THAT(resView[i][1], Catch::Matchers::WithinAbs(0.0, 1e-12));
            REQUIRE_THAT(resView[i][2], Catch::Matchers::WithinAbs(0.0, 1e-12));
        }

        // The returned field carries extrapolated/processor BCs and has been corrected, so its
        // boundary values are the owner values rather than the default zero.
        auto hostBnd = res.boundaryData().value().copyToHost();
        auto bndView = hostBnd.view();
        for (localIdx i = 0; i < hostBnd.size(); ++i)
        {
            REQUIRE_THAT(bndView[i][0], Catch::Matchers::WithinAbs(ux, 1e-12));
        }
    }

    SECTION("a zero flux reconstructs to zero" + execName)
    {
        auto mesh = create1DUniformMesh(exec, 10);
        auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
        fvcc::SurfaceField<scalar> phi(exec, "phi", mesh, surfaceBCs);
        fill(phi.internalVector(), 0.0);
        fill(phi.boundaryData().value(), 0.0);

        auto res = fvcc::reconstruct(phi);
        auto hostRes = res.internalVector().copyToHost();
        auto resView = hostRes.view();
        for (localIdx i = 0; i < hostRes.size(); ++i)
        {
            REQUIRE_THAT(resView[i][0], Catch::Matchers::WithinAbs(0.0, 1e-12));
            REQUIRE_THAT(resView[i][1], Catch::Matchers::WithinAbs(0.0, 1e-12));
            REQUIRE_THAT(resView[i][2], Catch::Matchers::WithinAbs(0.0, 1e-12));
        }
    }

    SECTION("a 3D mesh recovers a fully 3D velocity" + execName)
    {
        // Face normals span all three axes here, so the tensor is full rank and no
        // regularisation path is taken: every component must come back exactly.
        auto mesh = create3DUniformMesh(exec, 3, 3, 3);
        auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
        fvcc::SurfaceField<scalar> phi(exec, "phi", mesh, surfaceBCs);

        const Vec3 u {1.0, -2.0, 0.5};
        auto [phiI, phiB] = views(phi.internalVector(), phi.boundaryData().value());
        const auto sf = mesh.faceNormals().view();
        const auto bSf = mesh.boundaryMesh().faceNormals().view();
        parallelFor(
            exec,
            {0, mesh.nInternalFaces()},
            NEON_LAMBDA(const localIdx f) {
                phiI[f] = u[0] * sf[f][0] + u[1] * sf[f][1] + u[2] * sf[f][2];
            }
        );
        parallelFor(
            exec,
            {0, mesh.nBoundaryFaces()},
            NEON_LAMBDA(const localIdx bf) {
                phiB[bf] = u[0] * bSf[bf][0] + u[1] * bSf[bf][1] + u[2] * bSf[bf][2];
            }
        );

        auto res = fvcc::reconstruct(phi);
        auto hostRes = res.internalVector().copyToHost();
        auto resView = hostRes.view();
        for (localIdx i = 0; i < hostRes.size(); ++i)
        {
            REQUIRE_THAT(resView[i][0], Catch::Matchers::WithinAbs(u[0], 1e-12));
            REQUIRE_THAT(resView[i][1], Catch::Matchers::WithinAbs(u[1], 1e-12));
            REQUIRE_THAT(resView[i][2], Catch::Matchers::WithinAbs(u[2], 1e-12));
        }
    }
}

}
