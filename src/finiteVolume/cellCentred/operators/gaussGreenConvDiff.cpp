// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenConvDiff.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* -------------------------------------------------------------------------- */
/*  Explicit fused operator                                                   */
/* -------------------------------------------------------------------------- */

template<typename ValueType>
void computeConvDiffExp(
    const SurfaceField<scalar>& faceFlux,
    const SurfaceField<scalar>& gamma, // currently unused in explicit Laplacian
    const VolumeField<ValueType>& phi,
    const SurfaceInterpolation<ValueType>& surfInterp,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    Vector<ValueType>& result,
    const dsl::Coeff operatorScaling
)
{
    (void)gamma; // not used in current explicit Laplacian formulation

    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();

    // Interpolated values at faces (Gauss-Green convection)
    SurfaceField<ValueType> phif(
        exec, "phif", mesh, createCalculatedBCs<SurfaceBoundary<ValueType>>(mesh)
    );
    surfInterp.interpolate(faceFlux, phi, phif);
    // simple boundary treatment: copy from cell field (same as computeDivExp)
    phif.boundaryData().value() = phi.boundaryData().value();

    // Face-normal gradient for Laplacian (Gauss-Green diffusion)
    SurfaceField<ValueType> faceNormalGrad = faceNormalGradient.faceNormalGrad(phi);

    const auto [owner, neighbour, surfFaceCells] =
        views(mesh.faceOwner(), mesh.faceNeighbour(), mesh.boundaryMesh().faceCells());

    const auto [res, faceArea, fnGrad, vol, fluxV, phiF] = views(
        result,
        mesh.magFaceAreas(),
        faceNormalGrad.internalVector(),
        mesh.cellVolumes(),
        faceFlux.internalVector(),
        phif.internalVector()
    );

    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nTotalFaces = fluxV.size();
    const auto nCells = mesh.nCells();

    // internal faces
    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx i) {
            // convection flux
            ValueType fluxConv = fluxV[i] * phiF[i];
            // diffusion flux
            ValueType fluxDiff = faceArea[i] * fnGrad[i];

            ValueType fluxTotal = fluxConv + fluxDiff;

            Kokkos::atomic_add(&res[owner[i]], fluxTotal);
            Kokkos::atomic_sub(&res[neighbour[i]], fluxTotal);
        },
        "computeConvDiffExplicitInternal"
    );

    // boundary faces
    parallelFor(
        exec,
        {nInternalFaces, nTotalFaces},
        KOKKOS_LAMBDA(const localIdx i) {
            auto own = surfFaceCells[i - nInternalFaces];

            ValueType fluxConv = fluxV[i] * phiF[i];
            ValueType fluxDiff = faceArea[i] * fnGrad[i];

            ValueType fluxTotal = fluxConv + fluxDiff;

            Kokkos::atomic_add(&res[own], fluxTotal);
        },
        "computeConvDiffExplicitBoundary"
    );

    // normalize by cell volume and operator scaling (same pattern as div/laplacian)
    parallelFor(
        exec,
        {0, nCells},
        KOKKOS_LAMBDA(const localIdx celli) { res[celli] *= operatorScaling[celli] / vol[celli]; },
        "normalizeConvDiffExplicit"
    );
}

#define NF_DECLARE_COMPUTE_EXP_CONVDIFF(TYPENAME)                                                  \
    template void computeConvDiffExp<TYPENAME>(                                                    \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceField<scalar>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceInterpolation<TYPENAME>&,                                                     \
        const FaceNormalGradient<TYPENAME>&,                                                       \
        Vector<TYPENAME>&,                                                                         \
        const dsl::Coeff                                                                           \
    )

NF_DECLARE_COMPUTE_EXP_CONVDIFF(scalar);
NF_DECLARE_COMPUTE_EXP_CONVDIFF(Vec3);


/* -------------------------------------------------------------------------- */
/*  Implicit fused operator (matrix assembly)                                 */
/* -------------------------------------------------------------------------- */

template<typename ValueType>
void computeConvDiffImp(
    la::LinearSystem<ValueType, localIdx>& ls,
    const SurfaceField<scalar>& faceFlux,
    const SurfaceField<scalar>& gamma,
    const VolumeField<ValueType>& phi,
    const SurfaceInterpolation<ValueType>& surfInterp,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    const dsl::Coeff operatorScaling,
    const la::SparsityPattern& sparsityPattern
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();
    const auto nInternalFaces = mesh.nInternalFaces();

    // interpolation weights for convection (same as GaussGreenDiv)
    const auto weights = surfInterp.weight(faceFlux, phi);

    const auto
        [faceFluxV,
         weightsV,
         sGamma,
         deltaCoeffsInt,
         magFaceArea,
         owner,
         neighbour,
         surfFaceCells,
         diagOffs,
         ownOffs,
         neiOffs] =
            views(
                faceFlux.internalVector(),
                weights.internalVector(),
                gamma.internalVector(),
                faceNormalGradient.deltaCoeffs().internalVector(),
                mesh.magFaceAreas(),
                mesh.faceOwner(),
                mesh.faceNeighbour(),
                mesh.boundaryMesh().faceCells(),
                sparsityPattern.diagOffset(),
                sparsityPattern.ownerOffset(),
                sparsityPattern.neighbourOffset()
            );

    // matrix + rhs access (same style as Laplacian)
    auto [values, colIdxs, rowOffs] = ls.matrix().view();
    auto rhs = ls.rhs().view();

    // internal faces: fused convection + diffusion
    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            const scalar fluxConv = faceFluxV[facei];
            const scalar weight = weightsV[facei];

            const scalar fluxDiff = deltaCoeffsInt[facei] * sGamma[facei] * magFaceArea[facei];

            const localIdx own = owner[facei];
            const localIdx nei = neighbour[facei];

            const localIdx rowOwnStart = rowOffs[own];
            const localIdx rowNeiStart = rowOffs[nei];

            const scalar scaleOwn = operatorScaling[own];
            const scalar scaleNei = operatorScaling[nei];

            // ----------------- Convection (Gauss interpolation) -----------------
            // Same coefficients as computeDivImp

            // neighbour upper coefficient
            ValueType value = -weight * fluxConv * one<ValueType>();
            values[rowNeiStart + neiOffs[facei]] += value * scaleNei;
            Kokkos::atomic_sub(&values[rowOwnStart + diagOffs[own]], value * scaleOwn);

            // owner lower coefficient
            value = fluxConv * (1.0 - weight) * one<ValueType>();
            values[rowOwnStart + ownOffs[facei]] += value * scaleOwn;
            Kokkos::atomic_sub(&values[rowNeiStart + diagOffs[nei]], value * scaleNei);

            // ----------------- Diffusion (Gauss-Green Laplacian) ----------------
            ValueType diffVal = fluxDiff * one<ValueType>();

            // upper (neighbour row)
            values[rowNeiStart + neiOffs[facei]] += diffVal * scaleNei;
            Kokkos::atomic_sub(&values[rowOwnStart + diagOffs[own]], diffVal * scaleOwn);

            // lower (owner row)
            values[rowOwnStart + ownOffs[facei]] += diffVal * scaleOwn;
            Kokkos::atomic_sub(&values[rowNeiStart + diagOffs[nei]], diffVal * scaleNei);
        },
        "computeLocalGaussGreenConvDiffCoefficients"
    );

    // ----------------- Boundary faces ----------------------------------------

    auto [bweights, refGradient, value, valueFraction, refValue, deltaCoeffsB] = views(
        weights.boundaryData().value(),
        phi.boundaryData().refGrad(),
        phi.boundaryData().value(),
        phi.boundaryData().valueFraction(),
        phi.boundaryData().refValue(),
        mesh.boundaryMesh().deltaCoeffs()
    );

    auto& bcCoeffs =
        ls.auxiliaryCoefficients().template get<la::BoundaryCoefficients<ValueType, localIdx>>(
            "boundaryCoefficients"
        );

    auto [boundValues, rhsBoundValues] = views(bcCoeffs.matrixValues, bcCoeffs.rhsValues);

    const auto nFacesTotal = sGamma.size();

    parallelFor(
        exec,
        {nInternalFaces, nFacesTotal},
        KOKKOS_LAMBDA(const localIdx facei) {
            const localIdx bcfacei = facei - nInternalFaces;

            const scalar fluxConv = bweights[bcfacei] * faceFluxV[facei];
            const scalar fluxDiffBase = sGamma[facei] * magFaceArea[facei];

            const localIdx own = surfFaceCells[bcfacei];
            const localIdx rowOwnStart = rowOffs[own];
            const scalar scaleOwn = operatorScaling[own];

            const scalar valFrac = valueFraction[bcfacei];
            const scalar valFrac2 = 1.0 - valFrac;
            const scalar deltaB = deltaCoeffsB[bcfacei];
            const scalar deltaInt = deltaCoeffsInt[facei];

            // ---------- Convection boundary contribution (same as div) ----------
            ValueType valueMatConv = fluxConv * scaleOwn * valFrac2 * one<ValueType>();

            ValueType valueRhsConv = fluxConv * scaleOwn * (valFrac * refValue[bcfacei])
                                   + valFrac2 * refGradient[bcfacei] * (1.0 / deltaB);

            // ---------- Diffusion boundary contribution (same as laplacian) -----
            ValueType valueMatDiff =
                fluxDiffBase * scaleOwn * valFrac * deltaInt * one<ValueType>();

            ValueType valueRhsDiff =
                fluxDiffBase * scaleOwn
                * (valFrac * deltaInt * refValue[bcfacei] + (1.0 - valFrac) * refGradient[bcfacei]);

            // ---------- Fused effect on matrix & rhs ----------------------------
            // diag += conv - diff
            Kokkos::atomic_add(&values[rowOwnStart + diagOffs[own]], valueMatConv);
            Kokkos::atomic_sub(&values[rowOwnStart + diagOffs[own]], valueMatDiff);

            ValueType valueMatTotal = valueMatConv - valueMatDiff;
            boundValues[bcfacei] = valueMatTotal;

            // rhs -= (conv + diff)
            ValueType valueRhsTotal = valueRhsConv + valueRhsDiff;
            Kokkos::atomic_sub(&rhs[own], valueRhsTotal);
            rhsBoundValues[bcfacei] = valueRhsTotal;
        },
        "computeInterfaceGaussGreenConvDiffCoefficients"
    );
}

#define NN_DECLARE_COMPUTE_IMP_CONVDIFF(TYPENAME)                                                  \
    template void computeConvDiffImp<                                                              \
        TYPENAME>(la::LinearSystem<TYPENAME, localIdx>&, const SurfaceField<scalar>&, const SurfaceField<scalar>&, const VolumeField<TYPENAME>&, const SurfaceInterpolation<TYPENAME>&, const FaceNormalGradient<TYPENAME>&, const dsl::Coeff, const la::SparsityPattern&)

NN_DECLARE_COMPUTE_IMP_CONVDIFF(scalar);
NN_DECLARE_COMPUTE_IMP_CONVDIFF(Vec3);

} // namespace NeoN::finiteVolume::cellCentred
