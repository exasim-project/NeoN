// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <memory>
#include <string>

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/surface/ddtFluxCorrBoundary.hpp"

namespace NeoN::finiteVolume::cellCentred
{

using VolVectorField = VolumeField<Vec3>;
using SurfScalarField = SurfaceField<scalar>;

namespace detail
{

// ddtCouplingCoeff of OpenFOAM's ddtScheme::fvcDdtPhiCoeff: 1 - min(|phiCorr| / (|phi| + SMALL), 1)
KOKKOS_INLINE_FUNCTION
scalar ddtFluxCorrLimiter(const scalar fluxMag, const scalar corrMag)
{
    constexpr scalar small = 1.0e-15; // OpenFOAM's SMALL for double precision
    const auto ratio = corrMag / (fluxMag + small);
    return scalar(1) - Kokkos::min(ratio, scalar(1));
}

// ------------------------------------------------------------------
// BDF1 kernel (also used for BDF2 startup)
// ------------------------------------------------------------------
inline void ddtFluxCorrBDF1Kernel(
    const Executor& exec,
    const UnstructuredMesh& mesh,
    const SurfScalarField& flux0,
    const SurfaceField<Vec3>& uf0,
    SurfScalarField& fluxCorr,
    scalar dt
)
{
    const scalar a1 = scalar(1) / dt;
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    const auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();

    // Internal faces. mesh.faceNormals() is OF-full but its leading nInternalFaces
    // entries coincide with the compressed internal-face Sf, so indexing by i is fine.
    auto [outV, flux0V, uf0V, SfV] = views(
        fluxCorr.internalVector(), flux0.internalVector(), uf0.internalVector(), mesh.faceNormals()
    );
    parallelFor(
        exec,
        {sizeType(0), nInternalFaces},
        NEON_LAMBDA(const localIdx i) {
            const auto d = (SfV[i] & uf0V[i]);
            const auto corr = flux0V[i] - d;
            const scalar limiter = ddtFluxCorrLimiter(mag(flux0V[i]), mag(corr));
            outV[i] = limiter * a1 * corr;
        },
        "ddtFluxCorr::BDF1::internal"
    );

    // Boundary + processor faces.
    // boundaryData().value() stores [physical boundary | processor] faces; the
    // boundary-mesh Sf (mesh.boundaryMesh().faceNormals()) is the matching COMPRESSED
    // face-area-normal field spanning the same range, so it is indexed directly by bfi.
    // The previous code used the OF-full mesh.faceNormals()[nInternalFaces + bfi] and
    // only looped over nBoundaryFaces: that reads the wrong face when empty/wedge
    // patches are present, and never touched the processor tail at all -- leaving the
    // ddt flux correction unset on processor faces and corrupting phiHbyA (hence the
    // inflated Courant number) on distributed runs. The processor-patch BC is
    // 'calculated' (no-op correctBoundaryCondition), so this locally computed proc-tail
    // value survives the fluxCorr.correctBoundaryConditions() call below. See
    // project_neon_compressed_face_indexing and the matching proc-face loop in
    // pressureVelocityCoupling::flux.
    auto [outBV, flux0BV, uf0BV] = views(
        fluxCorr.boundaryData().value(), flux0.boundaryData().value(), uf0.boundaryData().value()
    );
    const auto bFaceNormals = mesh.boundaryMesh().faceNormals().view();
    parallelFor(
        exec,
        {sizeType(0), nBoundaryFaces + nProcBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            const auto d = (bFaceNormals[bfi] & uf0BV[bfi]);
            const auto corr = flux0BV[bfi] - d;
            const scalar limiter = ddtFluxCorrLimiter(mag(flux0BV[bfi]), mag(corr));
            outBV[bfi] = limiter * a1 * corr;
        },
        "ddtFluxCorr::BDF1::boundary"
    );
}

// ------------------------------------------------------------------
// BDF2 kernel
//
// Same as OpenFOAM's backwardDdtScheme::fvcDdtPhiCorr: a single coupling coefficient, computed from
// the single-level correction phi^0 - Sf & uf^0, scales the combined BDF2 correction
//   (2 phi^0 - phi^00 / 2) - Sf & (2 uf^0 - uf^00 / 2).
// ------------------------------------------------------------------
inline void ddtFluxCorrBDF2Kernel(
    const Executor& exec,
    const UnstructuredMesh& mesh,
    const SurfScalarField& flux0,
    const SurfScalarField& flux00,
    const SurfaceField<Vec3>& uf0,
    const SurfaceField<Vec3>& uf00,
    SurfScalarField& fluxCorr,
    scalar dt
)
{
    const scalar a1 = 2.0 / dt;
    const scalar a2 = -0.5 / dt;
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    const auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();

    // Internal faces
    {
        auto [outV, flux0V, flux00V, uf0V, uf00V, SfV] = views(
            fluxCorr.internalVector(),
            flux0.internalVector(),
            flux00.internalVector(),
            uf0.internalVector(),
            uf00.internalVector(),
            mesh.faceNormals()
        );
        parallelFor(
            exec,
            {sizeType(0), nInternalFaces},
            NEON_LAMBDA(const localIdx i) {
                const auto d1 = (SfV[i] & uf0V[i]);
                const auto corr1 = flux0V[i] - d1;

                const auto d2 = (SfV[i] & uf00V[i]);
                const auto corr2 = flux00V[i] - d2;

                const scalar limiter = ddtFluxCorrLimiter(mag(flux0V[i]), mag(corr1));

                outV[i] = limiter * (a1 * corr1 + a2 * corr2);
            },
            "ddtFluxCorr::BDF2::internal"
        );
    }

    // Boundary + processor faces. Compressed boundary-mesh Sf indexed by bfi over the
    // [physical | processor] range; mirrors the BDF1 kernel (see its comment for why the
    // OF-full mesh.faceNormals() must not be used here and why the processor tail must be
    // included for correct distributed phiHbyA / Courant number).
    {
        auto outBV = fluxCorr.boundaryData().value().view();
        auto flux0BV = flux0.boundaryData().value().view();
        auto flux00BV = flux00.boundaryData().value().view();
        auto uf0BV = uf0.boundaryData().value().view();
        auto uf00BV = uf00.boundaryData().value().view();
        const auto bFaceNormals = mesh.boundaryMesh().faceNormals().view();
        parallelFor(
            exec,
            {sizeType(0), nBoundaryFaces + nProcBoundaryFaces},
            NEON_LAMBDA(const localIdx bfi) {
                const auto d1 = (bFaceNormals[bfi] & uf0BV[bfi]);
                const auto corr1 = flux0BV[bfi] - d1;

                const auto d2 = (bFaceNormals[bfi] & uf00BV[bfi]);
                const auto corr2 = flux00BV[bfi] - d2;

                const scalar limiter = ddtFluxCorrLimiter(mag(flux0BV[bfi]), mag(corr1));

                outBV[bfi] = limiter * (a1 * corr1 + a2 * corr2);
            },
            "ddtFluxCorr::BDF2::boundary"
        );
    }
}

} // namespace detail

inline SurfScalarField
ddtFluxCorr(const VolVectorField& u, const SurfScalarField& phi, scalar dt, DdtScheme scheme)
{
    const auto& mesh = u.mesh();
    const auto& exec = phi.exec();

    // --- interpolation
    SurfaceInterpolation<Vec3> interp(exec, mesh, TokenList({std::string("linear")}));

    // --- boundary conditions consistent with U
    auto surfaceBCs = createFluxCorrBCsFromU(mesh, u);

    SurfScalarField fluxCorr(exec, "ddtFluxCorr", mesh, surfaceBCs);

    const int level = oldTimeLevel(u);

    // --- BDF1 / startup
    const auto& u0 = oldTime(u);
    const auto& phi0 = oldTime(phi);
    auto uf0 = interp.interpolate(u0);

    if (scheme == DdtScheme::BDF2 && level >= 2) // --- BDF2 contribution
    {
        const auto& u00 = oldTime(u0);
        const auto& phi00 = oldTime(phi0);
        auto uf00 = interp.interpolate(u00);

        detail::ddtFluxCorrBDF2Kernel(exec, mesh, phi0, phi00, uf0, uf00, fluxCorr, dt);
    }
    else
    {
        detail::ddtFluxCorrBDF1Kernel(exec, mesh, phi0, uf0, fluxCorr, dt);
    }

    fluxCorr.correctBoundaryConditions();
    return fluxCorr;
}

namespace detail
{

// Fills `rhoU` with the cell-centred product rho * U, boundary included (boundary value =
// rho_b * U_b), which is what OpenFOAM interpolates in ddtScheme::fvcDdtPhiCorr(rho, U, phi)
// (rhoU0 = rho.oldTime()*U.oldTime(), then dotInterpolate(Sf, rhoU0)). It is not the same as
// interpolate(rho) * interpolate(U): the two differ wherever rho and U vary between the cells.
// `rhoU` is an existing field with 'calculated' patches; nothing is allocated here.
inline void densityWeightedVelocity(
    const VolumeField<scalar>& rho, const VolVectorField& u, VolVectorField& rhoU
)
{
    const auto& exec = u.exec();
    {
        auto [outV, rhoV, uV] =
            views(rhoU.internalVector(), rho.internalVector(), u.internalVector());
        parallelFor(
            exec,
            {sizeType(0), rhoU.internalVector().size()},
            NEON_LAMBDA(const localIdx i) { outV[i] = rhoV[i] * uV[i]; },
            "ddtFluxCorr::rhoU::internal"
        );
    }
    {
        // [physical boundary | processor] faces, both already carry valid values in rho and U
        auto [outBV, rhoBV, uBV] = views(
            rhoU.boundaryData().value(), rho.boundaryData().value(), u.boundaryData().value()
        );
        parallelFor(
            exec,
            {sizeType(0), rhoU.boundaryData().value().size()},
            NEON_LAMBDA(const localIdx bfi) { outBV[bfi] = rhoBV[bfi] * uBV[bfi]; },
            "ddtFluxCorr::rhoU::boundary"
        );
    }
}

// Everything the density-weighted ddtFluxCorr needs besides its inputs and its result: the linear
// interpolation, the rho * U scratch field and the two interpolated face fields (rho^0 * U^0 and,
// for BDF2, rho^00 * U^00). They depend on the mesh only, so they are built once and reused by
// every call on that mesh. Rebuilding them per call allocated two volume/surface fields, their
// boundary-condition objects and the interpolation scheme each time step, which is expensive on
// GPUs. Same per-mesh cache as the linearUpwindV gradient operator (GeometryScheme::readOrCreate).
struct DensityWeightedFluxCorrWorkspace
{
    SurfaceInterpolation<Vec3> interpolation;
    VolVectorField rhoU;
    SurfaceField<Vec3> rhoUf0;
    SurfaceField<Vec3> rhoUf00;

    DensityWeightedFluxCorrWorkspace(const Executor& exec, const UnstructuredMesh& mesh)
        : interpolation(exec, mesh, TokenList({std::string("linear")})),
          rhoU(
              exec, "ddtFluxCorr::rhoU", mesh, createCalculatedProcBCs<VolumeBoundary<Vec3>>(mesh)
          ),
          rhoUf0(
              exec, "ddtFluxCorr::rhoUf0", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh)
          ),
          rhoUf00(
              exec, "ddtFluxCorr::rhoUf00", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh)
          )
    {}
};

inline DensityWeightedFluxCorrWorkspace&
densityWeightedFluxCorrWorkspace(const Executor& exec, const UnstructuredMesh& mesh)
{
    auto& db = mesh.stencilDB();
    const std::string key = "ddtFluxCorr::DensityWeightedFluxCorrWorkspace";
    if (!db.contains(key))
    {
        db.insert(key, std::make_shared<DensityWeightedFluxCorrWorkspace>(exec, mesh));
    }
    return *db.get<std::shared_ptr<DensityWeightedFluxCorrWorkspace>>(key);
}

} // namespace detail

// Density-weighted overload for compressible phi = rho_f * uf, the counterpart of OpenFOAM's
// fvc::ddtCorr(rho, U, phi) (ddtScheme::fvcDdtPhiCorr(rho, U, phi)). It reuses the BDF1/BDF2
// kernels on the interpolated rho^k * U^k (see detail::densityWeightedVelocity); rho and U carry
// their own old-time storage, so no persistent rhoUf field is needed.
// As in OpenFOAM, the coupling coefficient is evaluated on the rho*U product, whose patches are
// 'calculated' and therefore fix their value: the correction is zero on every physical boundary
// face (see createDensityWeightedFluxCorrBCs), unlike the incompressible overload, which only
// zeroes it on patches where U itself fixes the value.
//
// BDF2 needs a second old-time level of U, rho and phi alike: the scheme falls back to BDF1
// (OpenFOAM's start-up step) as soon as any of the three has only one.
//
// This overload writes into `fluxCorr`, which the caller owns and which must carry the patches of
// createDensityWeightedFluxCorrBCs(mesh); after the first call on a mesh it allocates nothing, so
// a time loop should keep one `fluxCorr` and call this. The overload below returns a new field.
inline void ddtFluxCorr(
    SurfScalarField& fluxCorr,
    const VolumeField<scalar>& rho,
    const VolVectorField& u,
    const SurfScalarField& phi,
    scalar dt,
    DdtScheme scheme
)
{
    const auto& mesh = u.mesh();
    const auto& exec = phi.exec();
    auto& work = detail::densityWeightedFluxCorrWorkspace(exec, mesh);

    // --- BDF1 / startup
    const auto& u0 = oldTime(u);
    const auto& rho0 = oldTime(rho);
    const auto& phi0 = oldTime(phi);
    detail::densityWeightedVelocity(rho0, u0, work.rhoU);
    work.interpolation.interpolate(work.rhoU, work.rhoUf0);

    const int level = std::min({oldTimeLevel(u), oldTimeLevel(rho), oldTimeLevel(phi)});
    if (scheme == DdtScheme::BDF2 && level >= 2) // --- BDF2 contribution
    {
        const auto& u00 = oldTime(u0);
        const auto& rho00 = oldTime(rho0);
        const auto& phi00 = oldTime(phi0);
        detail::densityWeightedVelocity(rho00, u00, work.rhoU);
        work.interpolation.interpolate(work.rhoU, work.rhoUf00);

        detail::ddtFluxCorrBDF2Kernel(
            exec, mesh, phi0, phi00, work.rhoUf0, work.rhoUf00, fluxCorr, dt
        );
    }
    else
    {
        detail::ddtFluxCorrBDF1Kernel(exec, mesh, phi0, work.rhoUf0, fluxCorr, dt);
    }

    fluxCorr.correctBoundaryConditions();
}

inline SurfScalarField ddtFluxCorr(
    const VolumeField<scalar>& rho,
    const VolVectorField& u,
    const SurfScalarField& phi,
    scalar dt,
    DdtScheme scheme
)
{
    // --- zero correction on physical boundary faces, see above
    SurfScalarField fluxCorr(
        phi.exec(), "ddtFluxCorr", u.mesh(), createDensityWeightedFluxCorrBCs(u.mesh())
    );
    ddtFluxCorr(fluxCorr, rho, u, phi, dt, scheme);
    return fluxCorr;
}

} // namespace NeoN::finiteVolume::cellCentred
