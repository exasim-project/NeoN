// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenLaplacian.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeLaplacianExp(
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    const SurfaceField<scalar>&, // gamma,
    const VolumeField<ValueType>& phi,
    Vector<ValueType>& lapPhi,
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();

    SurfaceField<ValueType> faceNormalGrad = faceNormalGradient.faceNormalGrad(phi);

    const auto [owner, neighbour, surfFaceCells] =
        views(mesh.faceOwner(), mesh.faceNeighbour(), mesh.boundaryMesh().faceCells());

    const auto [result, faceArea, fnGrad, vol] =
        views(lapPhi, mesh.magFaceAreas(), faceNormalGrad.internalVector(), mesh.cellVolumes());
    // Boundary-tail face area (compressed indexing matching bcfacei). Used for
    // proc faces because mesh.magFaceAreas() is in OpenFOAM's full face list
    // (incl. empty patches like defaultFaces) — indexing it with the compressed
    // proc-face index reads the wrong face's area.
    const auto bcMagSf = mesh.boundaryMesh().magSf().view();

    auto nInternalFaces = mesh.nInternalFaces();
    auto nBoundaryFaces = mesh.nBoundaryFaces();

    // Green-Gauss Laplacian: ∇·(γ∇φ)_C = (1/V_C) * sum_f γ_f * |S_f| * (∂φ/∂n)_f
    //
    // fnGrad[f] = nonOrthDeltaCoeffs[f] * (phi[nei] − phi[own])  (computed by FaceNormalGradient)
    //   S_f points from owner to neighbour by construction, so fnGrad is the gradient
    //   component in the outward direction from the owner cell.
    //   fnGrad > 0  when phi_N > phi_P (gradient points outward from owner)
    //             → diffusion brings φ into owner → positive Laplacian at owner (owner gains φ)
    //             → diffusion takes φ from neighbour → negative Laplacian at neighbour
    //
    // This computes +∇·(γ∇φ) (positive Laplacian form).
    // TODO use NeoN::add and sub
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx i) {
            ValueType flux = faceArea[i] * fnGrad[i];
            Kokkos::atomic_add(
                &result[owner[i]], flux
            ); // +|S_f| * fnGrad (outward gradient from owner)
            Kokkos::atomic_sub(
                &result[neighbour[i]], flux
            ); // −|S_f| * fnGrad (inward gradient for neighbour)
        },
        "computeLaplacianExplicitInternal"
    );

<<<<<<< HEAD
    // Physical (non-proc) boundary faces: only the owner cell is on this rank.
    // For non-proc patches, OpenFOAM's full face index and NeoN's compressed
    // index agree (empty patches like defaultFaces have size()==0 in fvPatch),
    // so faceArea[i] = mesh.magFaceAreas()[i] is correct here.
=======
    // Boundary faces: only the owner cell is on this rank.
>>>>>>> origin/develop
    parallelFor(
        exec,
        {nInternalFaces, nInternalFaces + nBoundaryFaces},
        NEON_LAMBDA(const localIdx i) {
            auto own = surfFaceCells[i - nInternalFaces];
            ValueType valueOwn =
                faceArea[i] * fnGrad[i]; // +|S_f| * fnGrad (S_f outward from owner)
            Kokkos::atomic_add(&result[own], valueOwn);
        },
        "computeLaplacianExplicitBoundary"
    );

    // Processor boundary faces: same operation but read the face area from the
    // boundary-mesh (compressed) view because mesh.magFaceAreas() is in OF's
    // full face list including empty patches.
    parallelFor(
        exec,
        {nInternalFaces + nBoundaryFaces, fnGrad.size()},
        NEON_LAMBDA(const localIdx i) {
            const auto bcfacei = i - nInternalFaces;
            auto own = surfFaceCells[bcfacei];
            ValueType valueOwn = bcMagSf[bcfacei] * fnGrad[i];
            Kokkos::atomic_add(&result[own], valueOwn);
        },
        "computeLaplacianExplicitProcBoundary"
    );

    parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx celli) { result[celli] *= operatorScaling[celli] / vol[celli]; },
        "computeLaplacianExplicitCells"
    );
}

#define NF_DECLARE_COMPUTE_EXP_LAP(TYPENAME)                                                       \
    template void computeLaplacianExp<TYPENAME>(                                                   \
        const FaceNormalGradient<TYPENAME>&,                                                       \
        const SurfaceField<scalar>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        Vector<TYPENAME>&,                                                                         \
        const dsl::Coeff                                                                           \
    )

NF_DECLARE_COMPUTE_EXP_LAP(scalar);
NF_DECLARE_COMPUTE_EXP_LAP(Vec3);

template<typename ValueType>
void computeLaplacianProcBoundImpl(
    la::LinearSystem<ValueType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<ValueType>& phi,
    const dsl::Coeff operatorScaling,
    const FaceNormalGradient<ValueType>& faceNormalGradient
)
{
    const auto exec = phi.exec();
    const auto& mesh = phi.mesh();

    auto gammaV = gamma.internalVector().view();

    // bm.magSf() is in the compressed boundary-tail layout matching bcfacei.
    // mesh.magFaceAreas() is in OpenFOAM's full face list (incl. empty patches);
    // indexing it with the compressed proc-face index reads the wrong face's
    // area, which produces wrong Laplacian matrix coefficients at proc patches
    // and a visible pressure discontinuity across the cut.
    const auto [deltaCoeffs, surfFaceCells] = views(
        faceNormalGradient.deltaCoeffs().internalVector(),
        mesh.boundaryMesh().faceCells() //,
                                        // mesh.boundaryMesh().deltaCoeffs()
    );
    const auto bcMagSf = mesh.boundaryMesh().magSf().view();

    const auto matIt = ls.faceToMatrixAddress();
    auto const rowOffs = matIt->sparsityPattern()->rowOffs().view();
    auto const diagOffs = matIt->diagOffset().view();

    auto values = ls.matrix().values().view();

    auto [/*bweights,*/ refGradient, value, refValue] = views(
        // weights.boundaryData().value(),
        phi.boundaryData().refGrad(),
        phi.boundaryData().value(),
        phi.boundaryData().refValue()
    );

    auto rhs = ls.rhs().view();
    auto bRhs = ls.boundaryRhs().view();
    auto bValues = ls.nonLocalMatrix().values().view();

    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    auto totalFaces = gammaV.size();
    NeoN::mpi::Environment mpiEnviron;
    parallelFor(
        exec,
        {nInternalFaces + nBoundaryFaces, totalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto bcfacei = facei - (nInternalFaces);
            auto bcfaceii = facei - (nInternalFaces + nBoundaryFaces);
            auto cell = surfFaceCells[bcfacei];
            auto rowStart = rowOffs[cell];
            auto c = operatorScaling[cell];

            auto flux = gammaV[facei] * bcMagSf[bcfacei] * deltaCoeffs[facei];
            auto value = flux * c * one<ValueType>();

            Kokkos::atomic_sub(&values[rowStart + diagOffs[cell]], value);
            bValues[bcfaceii] += value;

            // FIXME
            // std::cout << __FILE__ << ":" << __LINE__
            //           << " proc "
            //           << " rank " << mpiEnviron.rank()
            //           << " facei " << facei
            //           << " flux " << flux
            //           << " deltaCoeff " << deltaCoeffs[facei]
            //           << " magFaceArea " << magFaceArea[facei]
            //           << " sGamma " << gammaV[facei]
            //           << "\n";
        },
        "computeInterfaceLaplacianCoefficients"
    );
}

template<typename ValueType>
void computeLaplacianBoundImpl(
    la::LinearSystem<ValueType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<ValueType>& phi,
    const dsl::Coeff operatorScaling,
    const FaceNormalGradient<ValueType>& faceNormalGradient
)
{
    const auto exec = phi.exec();
    const auto& mesh = phi.mesh();

    auto gammaV = gamma.internalVector().view();


    const auto [magFaceArea, surfFaceCells, deltaCoeffs] = views(
        mesh.magFaceAreas(),
        mesh.boundaryMesh().faceCells(),
        faceNormalGradient.deltaCoeffs().internalVector()
    );

    const auto matIt = ls.faceToMatrixAddress();
    auto const rowOffs = matIt->sparsityPattern()->rowOffs().view();
    auto const diagOffs = matIt->diagOffset().view();

    auto values = ls.matrix().values().view();

    auto [/*bweights,*/ refGradient, value, valueFraction, refValue] = views(
        // weights.boundaryData().value(),
        phi.boundaryData().refGrad(),
        phi.boundaryData().value(),
        phi.boundaryData().valueFraction(),
        phi.boundaryData().refValue()
    );

    auto rhs = ls.rhs().view();
    auto bRhs = ls.boundaryRhs().view();
    auto bValues = ls.boundaryMatrix().values().view();


    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    auto totalFaces = nInternalFaces + nBoundaryFaces;
    NeoN::mpi::Environment mpiEnviron;
    parallelFor(
        exec,
        {nInternalFaces, totalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto bcfacei = facei - nInternalFaces;
            auto flux = gammaV[facei] * magFaceArea[facei];

            auto own = surfFaceCells[bcfacei];
            auto rowOwnStart = rowOffs[own];
            auto operatorScalingOwn = operatorScaling[own];

            auto valFrac1 = valueFraction[bcfacei];
            auto valFrac2 = 1.0 - valFrac1;

            // FIXME deltaCoeffs was previously indexed by facei?
            auto valueMat =
                flux * operatorScalingOwn * valFrac1 * deltaCoeffs[facei] * one<ValueType>();

            Kokkos::atomic_sub(&values[rowOwnStart + diagOffs[own]], valueMat);
            bValues[bcfacei] += valueMat;
            ValueType valueRhs = flux * operatorScalingOwn
                               * (valueFraction[bcfacei] * deltaCoeffs[facei] * refValue[bcfacei]
                                  + (1.0 - valueFraction[bcfacei]) * refGradient[bcfacei]);
            Kokkos::atomic_sub(&rhs[own], valueRhs);
            bRhs[bcfacei] += valueRhs;
        },
        "computeInterfaceLaplacianCoefficients"
    );
}

template<typename ValueType>
void computeLaplacianBoundImpl(
    la::LinearSystem<ValueType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<ValueType>& phi,
    const dsl::Coeff operatorScaling,
    const FaceNormalGradient<ValueType>& faceNormalGradient
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();
    const auto matIt = ls.faceToMatrixAddress();
    const auto [ownV, neiV, surfFaceCells, diagOffs, ownOffs, neiOffs, rowOffs] = views(
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        mesh.boundaryMesh().faceCells(),
        matIt->diagOffset(),
        matIt->ownerOffset(),
        matIt->neighbourOffset(),
        matIt->sparsityPattern()->rowOffs()
    );

    const auto [gammaV, deltaCoeffs, magFaceArea] = views(
        gamma.internalVector(),
        faceNormalGradient.deltaCoeffs().internalVector(),
        mesh.magFaceAreas()
    );

    auto rhs = ls.rhs().view();
    auto values = ls.matrix().values().view();

    const auto nInternalFaces = mesh.nInternalFaces();
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            // row and column indices
            auto ownRow = ownV[facei];
            auto neiRow = neiV[facei];

            auto neiRowStart = rowOffs[neiRow];
            auto ownRowStart = rowOffs[ownRow];

            // operator sign coefficient  handles: = +/- laplacian
            auto ownCoeff = coeff[ownRow];
            auto neiCoeff = coeff[neiRow];

            // matrix value diagonal and column offsets
            // NOTE TODO these are currently hardcode COO/CSR offsets
            auto ownDiagOffs = ownRowStart + static_cast<localIdx>(diagOffs[ownRow]);
            auto neiDiagOffs = neiRowStart + static_cast<localIdx>(diagOffs[neiRow]);
            auto upperColOffs = ownRowStart + ownOffs[facei];
            auto lowerColOffs = neiRowStart + neiOffs[facei];

            // Laplacian face coefficient: δ_f · γ_f · |S_f|
            // The Laplacian is symmetric — the same flux value enters both owner and neighbour rows
            // with opposite signs (diffusion out of one cell = diffusion into the other).
            // S_f points from owner to neighbour by construction.
            auto flux = deltaCoeffs[facei] * gammaV[facei] * magFaceArea[facei] * one<ValueType>();

            // triangular coefficients - neighbour -> lower, owner -> upper
            values[upperColOffs] += flux * ownCoeff;
            values[lowerColOffs] += flux * neiCoeff;

            // diagonal contribution is negative sum of offdiagonal coefficients
            Kokkos::atomic_sub(&values[ownDiagOffs], flux * ownCoeff);
            Kokkos::atomic_sub(&values[neiDiagOffs], flux * neiCoeff);
        },
        "computeLocalLaplacianCoefficients"
    );
}

#define NN_DECLARE_COMPUTE_IMP_LAP(TYPENAME)                                                                                                                      \
    template void computeLaplacianIntImpl<                                                                                                                        \
        TYPENAME>(la::LinearSystem<TYPENAME>&, const SurfaceField<scalar>&, const VolumeField<TYPENAME>&, const dsl::Coeff, const FaceNormalGradient<TYPENAME>&); \
    template void computeLaplacianBoundImpl<                                                                                                                      \
        TYPENAME>(la::LinearSystem<TYPENAME>&, const SurfaceField<scalar>&, const VolumeField<TYPENAME>&, const dsl::Coeff, const FaceNormalGradient<TYPENAME>&); \
    template void computeLaplacianProcBoundImpl<                                                                                                                  \
        TYPENAME>(la::LinearSystem<TYPENAME>&, const SurfaceField<scalar>&, const VolumeField<TYPENAME>&, const dsl::Coeff, const FaceNormalGradient<TYPENAME>&)

NN_DECLARE_COMPUTE_IMP_LAP(scalar);
NN_DECLARE_COMPUTE_IMP_LAP(Vec3);

};
