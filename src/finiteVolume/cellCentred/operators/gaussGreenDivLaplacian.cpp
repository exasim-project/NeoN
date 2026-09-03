// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDivLaplacian.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenLaplacian.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/boundedDiv.hpp"
#include "NeoN/linearAlgebra/meshIterationStrategies.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename FieldValueType, typename AssemblyType>
static void computeDivLaplacianIntImpl(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const VolumeField<FieldValueType>& /*U*/,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<FieldValueType>& /*divSurfInterp*/,
    const FaceNormalGradient<FieldValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto exec = phi.exec();
    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    const auto [gammaV, deltaV] =
        views(gamma.internalVector(), faceNormalGradient.deltaCoeffs().internalVector());
    const auto [phiV, ownV, neiV, magFaceAreaV] =
        views(phi.internalVector(), mesh.faceOwners(), mesh.faceNeighbors(), mesh.faceAreas());

    auto values = ls.matrix().values().view();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto own = ownV[facei];
            auto nei = neiV[facei];

            auto fluxDiv = phiV[facei];
            const auto weight = (phiV[facei] >= 0) ? 1.0 : 0.0;
            auto fluxLap = deltaV[facei] * gammaV[facei] * magFaceAreaV[facei];

            auto coeffNeiA = coeffA[nei];
            auto coeffOwnA = coeffA[own];
            auto coeffNeiB = coeffB[nei];
            auto coeffOwnB = coeffB[own];

            // lower triangular (nei row, own col)
            auto valueA =
                (-weight * coeffNeiA * fluxDiv + coeffNeiB * fluxLap) * one<AssemblyType>();
            values[ma.lowerIdx(nei, facei)] += valueA;
            Kokkos::atomic_sub(&values[ma.diagIdx(own)], valueA);

            // upper triangular (own row, nei col)
            auto valueB =
                ((1.0 - weight) * coeffOwnA * fluxDiv + coeffOwnB * fluxLap) * one<AssemblyType>();
            values[ma.upperIdx(own, facei)] += valueB;
            Kokkos::atomic_sub(&values[ma.diagIdx(nei)], valueB);
        },
        "computeLocalGaussGreenDivLaplacianCoefficients"
    );
}

template<typename FieldValueType, typename AssemblyType>
static void computeDivLaplacianIntCellBasedImpl(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const VolumeField<FieldValueType>& /*U*/,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<FieldValueType>& /*divSurfInterp*/,
    const FaceNormalGradient<FieldValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();

    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());
    auto iterator = std::dynamic_pointer_cast<la::CellBasedIterator>(ls.getMeshIterator()->get());

    const auto [phiV, gammaV, deltaV, magFaceAreaV] = views(
        phi.internalVector(),
        gamma.internalVector(),
        faceNormalGradient.deltaCoeffs().internalVector(),
        mesh.faceAreas()
    );

    auto cellBasedData = iterator->getCellBasedData();
    NF_ASSERT(
        cellBasedData != nullptr,
        "CellBasedData not initialized - call setComputeCellBasedData before invoking the "
        "cell-based kernel"
    );
    auto [cellFacesValues, cellFacesSegments] = cellBasedData->cellFaces.views();
    auto faceSignV = cellBasedData->faceSign.view();
    auto matrixColumnIdxV = cellBasedData->matrixColumnIdx.view();

    auto values = ls.matrix().values().view();

    parallelFor(
        exec,
        {0, iterator->size()},
        NEON_LAMBDA(const localIdx celli) {
            auto diagValue = zero<AssemblyType>();
            const auto numFaces = cellFacesSegments[celli + 1] - cellFacesSegments[celli];
            const auto startIdx = cellFacesSegments[celli];
            const auto cellCoeffA = coeffA[celli];
            const auto cellCoeffB = coeffB[celli];

            for (localIdx i = 0; i < numFaces; ++i)
            {
                const auto faceIdx = cellFacesValues[startIdx + i];
                const auto sign = faceSignV[startIdx + i];

                // upwind weight: 1 if flux leaves owner (flux > 0), 0 otherwise
                const auto flux = phiV[faceIdx];
                const auto w = (flux >= 0) ? scalar(1) : scalar(0);

                // Laplacian face coefficient: δ_f · γ_f · |S_f|
                const auto fluxLap = deltaV[faceIdx] * gammaV[faceIdx] * magFaceAreaV[faceIdx];

                AssemblyType offDiag;
                AssemblyType diagContrib;

                if (sign > 0) // celli is owner: off-diagonal is upper triangular entry
                {
                    offDiag = (flux * (1.0 - w) * cellCoeffA + fluxLap * cellCoeffB)
                            * one<AssemblyType>();
                    diagContrib =
                        (flux * w * cellCoeffA - fluxLap * cellCoeffB) * one<AssemblyType>();
                }
                else // celli is neighbor: off-diagonal is lower triangular entry
                {
                    offDiag = (-flux * w * cellCoeffA + fluxLap * cellCoeffB) * one<AssemblyType>();
                    diagContrib = (-flux * (1.0 - w) * cellCoeffA - fluxLap * cellCoeffB)
                                * one<AssemblyType>();
                }

                values[matrixColumnIdxV[startIdx + i]] += offDiag;
                diagValue += diagContrib;
            }

            values[ma.diagIdx(celli)] += diagValue;
        },
        "computeDivLaplacianIntCellBasedImpl::cellLoop"
    );
}

template<typename FieldValueType, typename AssemblyType>
static void computeDivLaplacianBoundImpl(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const VolumeField<FieldValueType>& u,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<FieldValueType>& /*divSurfInterp*/,
    const FaceNormalGradient<FieldValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();

    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    const auto [bPhiV, bGammaV, bDeltaV, bMagSf, surfFaceCells] = views(
        phi.boundaryData().value(),
        gamma.boundaryData().value(),
        faceNormalGradient.deltaCoeffs().boundaryData().value(),
        mesh.boundaryMesh().faceAreas(),
        mesh.boundaryMesh().faceOwners()
    );

    const auto [refGradient, valueFraction, refValue, deltaCoeffsA] = views(
        u.boundaryData().refGrad(),
        u.boundaryData().valueFraction(),
        u.boundaryData().refValue(),
        mesh.boundaryMesh().deltaCoeffs()
    );

    auto values = ls.matrix().values().view();
    auto rhs = ls.rhs().view();
    auto bRhs = ls.boundaryRhs().view();
    auto bValues = ls.boundaryMatrix().values().view();

    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            auto own = surfFaceCells[bfi];
            auto coeffAOwn = coeffA[own];
            auto coeffBOwn = coeffB[own];

            auto fluxDiv = bPhiV[bfi];
            auto fluxLap = bGammaV[bfi] * bMagSf[bfi];

            auto valFrac1 = valueFraction[bfi];
            auto valFrac2 = 1.0 - valFrac1;

            const auto bweights = (fluxDiv >= 0) ? scalar(1) : scalar(0);

            auto valueDiv = -bweights * coeffAOwn * fluxDiv * valFrac2;
            auto valueLap = bDeltaV[bfi] * coeffBOwn * fluxLap * valFrac1;
            auto valueA = (valueDiv + valueLap) * one<AssemblyType>();

            Kokkos::atomic_sub(&values[ma.diagIdx(own)], valueA);
            bValues[bfi] = valueA * (-1.0);

            // div rhs
            auto valueRhsA = (fluxDiv * coeffAOwn * (valFrac1 * refValue[bfi]))
                           + valFrac2 * refGradient[bfi] * (1.0 / deltaCoeffsA[bfi]);
            // lap rhs
            auto valueRhsB =
                fluxLap * coeffBOwn
                * (valFrac1 * refValue[bfi] * bDeltaV[bfi] + valFrac2 * refGradient[bfi]);

            Kokkos::atomic_sub(&rhs[own], valueRhsA);
            Kokkos::atomic_sub(&rhs[own], valueRhsB);
            bRhs[bfi] = valueRhsA + valueRhsB;
        },
        "computeInterfaceGaussGreenDivLaplacianCoefficients"
    );
}

template<typename FieldValueType, typename AssemblyType>
static void computeDivLaplacianProcBoundImpl(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const VolumeField<FieldValueType>& /*U*/,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<FieldValueType>& /*divSurfInterp*/,
    const FaceNormalGradient<FieldValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB
)
{
    const auto exec = phi.exec();
    const auto& mesh = phi.mesh();

    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    const auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();
    if (nProcBoundaryFaces == 0) return;

    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    const auto [bPhiV, bGammaV, bDeltaCoeffs, bMagSf, boundaryFaceOwner, isOwner] = views(
        phi.boundaryData().value(),
        gamma.boundaryData().value(),
        faceNormalGradient.deltaCoeffs().boundaryData().value(),
        mesh.boundaryMesh().faceAreas(),
        mesh.boundaryMesh().faceOwners(),
        mesh.boundaryMesh().weights()
    );

    auto bOffValues = ls.offDiagonalMatrix().values().view();
    auto bndDiagValues = ls.boundaryMatrix().values().view();
    auto values = ls.matrix().values().view();
    const auto rowOrderV = mesh.boundaryMesh().getRowOrderWriteIndex().view();

    parallelFor(
        exec,
        {0, nProcBoundaryFaces},
        NEON_LAMBDA(const localIdx procFacei) {
            auto bcfacei = nBoundaryFaces + procFacei;
            auto cell = boundaryFaceOwner[bcfacei];
            auto ownCoeffA = coeffA[cell];
            auto ownCoeffB = coeffB[cell];

            // Laplacian contribution
            auto lapFlux = bGammaV[bcfacei] * bMagSf[bcfacei] * bDeltaCoeffs[bcfacei];
            auto lapValue = lapFlux * ownCoeffB * one<AssemblyType>();

            // Div upwind contribution
            auto isOwnerFace = isOwner[bcfacei] > 0.0;
            auto sign = isOwnerFace ? scalar(-1) : scalar(1);
            auto bFlux = bPhiV[bcfacei];
            auto weight = isOwnerFace ? (bFlux >= 0 ? scalar(1) : scalar(0))
                                      : (bFlux >= 0 ? scalar(0) : scalar(1));
            auto divDiag = sign * weight * bFlux * ownCoeffA * one<AssemblyType>();
            auto divOffDiag =
                -sign * (scalar(1) - weight) * bFlux * ownCoeffA * one<AssemblyType>();

            auto diagValue = divDiag + lapValue;
            Kokkos::atomic_sub(&values[ma.diagIdx(cell)], diagValue);
            bndDiagValues[bcfacei] += diagValue;
            bOffValues[rowOrderV[procFacei]] += divOffDiag + lapValue;
        },
        "computeProcInterfaceGaussGreenDivLaplacianCoefficients"
    );
}

template<typename ValueType>
GaussGreenDivLaplacian<ValueType>::GaussGreenDivLaplacian(
    const Executor& exec, Dictionary divConfig, Dictionary lapConfig
)
    : dsl::OperatorMixin<VolumeField<ValueType>>(
        exec,
        dsl::Coeff(1.0),
        divConfig.get<detail::RefHolder<VolumeField<ValueType>>>("field").c,
        dsl::Operator::Type::Implicit
    ),
      coeffA_(divConfig.get<detail::RefHolder<dsl::Coeff>>("coeff").c),
      coeffB_(lapConfig.get<detail::RefHolder<dsl::Coeff>>("coeff").c),
      gamma_(lapConfig.get<detail::RefHolder<SurfaceField<scalar>>>("gamma").c),
      flux_(divConfig.get<detail::RefHolder<SurfaceField<scalar>>>("flux").c)
{}

template<typename ValueType>
void GaussGreenDivLaplacian<ValueType>::explicitOperation(Vector<ValueType>& /*source*/) const
{}

template<typename ValueType>
void GaussGreenDivLaplacian<ValueType>::implicitOperation(la::LinearSystem<ValueType>& ls) const
{
    if (auto* cellIter = dynamic_cast<la::CellBasedIterator*>(ls.getMeshIterator()->get().get()))
    {
        if (!cellIter->getCellBasedData())
        {
            cellIter->setComputeCellBasedData(
                this->getVector().mesh(), ls.matrix().sparsity(), ls.faceToMatrixAddress()
            );
        }
        computeDivLaplacianIntCellBasedImpl(
            ls,
            this->getVector(),
            flux_,
            gamma_,
            *divSurfaceInterpolation_,
            *faceNormalGradient_,
            coeffA_,
            coeffB_
        );
    }
    else
    {
        computeDivLaplacianIntImpl(
            ls,
            this->getVector(),
            flux_,
            gamma_,
            *divSurfaceInterpolation_,
            *faceNormalGradient_,
            coeffA_,
            coeffB_
        );
    }
    computeLaplacianNonOrthCorrImpl(ls, gamma_, this->getVector(), coeffB_, *faceNormalGradient_);
    computeDivLaplacianBoundImpl(
        ls,
        this->getVector(),
        flux_,
        gamma_,
        *divSurfaceInterpolation_,
        *faceNormalGradient_,
        coeffA_,
        coeffB_
    );
    computeDivLaplacianProcBoundImpl(
        ls,
        this->getVector(),
        flux_,
        gamma_,
        *divSurfaceInterpolation_,
        *faceNormalGradient_,
        coeffA_,
        coeffB_
    );

    // Deferred correction for a corrected div scheme (e.g. linearUpwind): the fused matrix uses
    // upwind div weights, the gradient correction is added explicitly to the rhs (internal + proc).
    if (divSurfaceInterpolation_->corrected())
    {
        const auto& mesh = this->getVector().mesh();
        SurfaceField<ValueType> correction(
            this->getVector().exec(),
            "divCorrection",
            mesh,
            createCalculatedBCs<SurfaceBoundary<ValueType>>(mesh)
        );
        divSurfaceInterpolation_->correction(flux_, this->getVector(), correction);
        addDivCorrectionToRhs(ls, flux_, correction, coeffA_);
    }

    // Bounded-convection Sp diagonal term for a `bounded` div scheme (same as BoundedDiv::div).
    if (bounded_)
    {
        applyBoundedDivDiagonal(ls, flux_, this->getVector().mesh(), coeffA_);
    }
}

template<typename ValueType>
void GaussGreenDivLaplacian<ValueType>::implicitOperation(la::LinearSystem<scalar, ValueType>& ls
) const
    requires(!std::is_same_v<ValueType, scalar>)
{
    if (auto* cellIter = dynamic_cast<la::CellBasedIterator*>(ls.getMeshIterator()->get().get()))
    {
        if (!cellIter->getCellBasedData())
        {
            cellIter->setComputeCellBasedData(
                this->getVector().mesh(), ls.matrix().sparsity(), ls.faceToMatrixAddress()
            );
        }
        computeDivLaplacianIntCellBasedImpl<ValueType, scalar>(
            ls,
            this->getVector(),
            flux_,
            gamma_,
            *divSurfaceInterpolation_,
            *faceNormalGradient_,
            coeffA_,
            coeffB_
        );
    }
    else
    {
        computeDivLaplacianIntImpl<ValueType, scalar>(
            ls,
            this->getVector(),
            flux_,
            gamma_,
            *divSurfaceInterpolation_,
            *faceNormalGradient_,
            coeffA_,
            coeffB_
        );
    }
    computeLaplacianNonOrthCorrImpl<ValueType, scalar>(
        ls, gamma_, this->getVector(), coeffB_, *faceNormalGradient_
    );
    computeDivLaplacianBoundImpl<ValueType, scalar>(
        ls,
        this->getVector(),
        flux_,
        gamma_,
        *divSurfaceInterpolation_,
        *faceNormalGradient_,
        coeffA_,
        coeffB_
    );
    computeDivLaplacianProcBoundImpl<ValueType, scalar>(
        ls,
        this->getVector(),
        flux_,
        gamma_,
        *divSurfaceInterpolation_,
        *faceNormalGradient_,
        coeffA_,
        coeffB_
    );

    // Deferred correction for a corrected div scheme (e.g. linearUpwind), scalar-matrix / Vec3-rhs.
    if (divSurfaceInterpolation_->corrected())
    {
        const auto& mesh = this->getVector().mesh();
        SurfaceField<ValueType> correction(
            this->getVector().exec(),
            "divCorrection",
            mesh,
            createCalculatedBCs<SurfaceBoundary<ValueType>>(mesh)
        );
        divSurfaceInterpolation_->correction(flux_, this->getVector(), correction);
        addDivCorrectionToRhs(ls, flux_, correction, coeffA_);
    }

    // Bounded-convection Sp diagonal term for a `bounded` div scheme (same as BoundedDiv::div).
    if (bounded_)
    {
        applyBoundedDivDiagonal(ls, flux_, this->getVector().mesh(), coeffA_);
    }
}

template<typename ValueType>
void GaussGreenDivLaplacian<ValueType>::read(const Input& input)
{
    const UnstructuredMesh& mesh = this->field_.mesh();
    TokenList laplTokens;
    TokenList divTokens;
    if (std::holds_alternative<Dictionary>(input))
    {
        auto dict = std::get<Dictionary>(input);
        std::string lapSchemeName = "laplacian(" + gamma_.name + "," + this->field_.name + ")";
        std::string divSchemeName = "div(" + flux_.name + "," + this->getVector().name + ")";
        laplTokens = dict.subDict("laplacianSchemes").get<NeoN::TokenList>(lapSchemeName);
        divTokens = dict.subDict("divSchemes").get<NeoN::TokenList>(divSchemeName);
    }
    else
    {
        NF_ERROR_EXIT("only dictionary input supported");
    }
    laplTokens.remove(0);

    // An OpenFOAM div scheme may carry a leading `bounded` prefix (the boundedness term
    // Sp(fvc::div(phi), psi)); strip it before the mandatory `Gauss` keyword and record it so
    // implicitOperation emits the same bounding diagonal term as the un-fused BoundedDiv wrapper.
    // Always re-derived (not just set on match) so a second read() with a different scheme can't
    // leave a stale true from an earlier call.
    bounded_ = divTokens.size() > 0 && divTokens.get<std::string>(0) == "bounded";
    if (bounded_)
    {
        divTokens.remove(0);
    }
    divTokens.remove(0);

    // Accept the cell-limited vector variant `linearUpwindV` alongside `linearUpwind`; both map to
    // the linearUpwind interpolation (SurfaceInterpolation registers "linearUpwindV" for the
    // CellLimited case).
    const auto divScheme = divTokens.get<std::string>(0);
    if (divScheme != "upwind" && divScheme != "linearUpwind" && divScheme != "linearUpwindV")
    {
        NF_ERROR_EXIT(
            "GaussGreenDivLaplacian only supports 'Gauss upwind', 'Gauss linearUpwind' or "
            "'Gauss linearUpwindV' for divSchemes, got: Gauss "
            << divScheme
        );
    }
    divSurfaceInterpolation_ =
        std::make_shared<SurfaceInterpolation<ValueType>>(this->field_.exec(), mesh, divTokens);
    laplTokens.remove(0);
    faceNormalGradient_ =
        std::make_shared<FaceNormalGradient<ValueType>>(this->field_.exec(), mesh, laplTokens);
}

template<typename ValueType>
std::string GaussGreenDivLaplacian<ValueType>::getName() const
{
    return "FusedDivLapOperator";
}

template<typename ValueType>
Dictionary GaussGreenDivLaplacian<ValueType>::getConfig() const
{
    return {};
}

template class GaussGreenDivLaplacian<scalar>;
template class GaussGreenDivLaplacian<Vec3>;

} // namespace NeoN::finiteVolume::cellCentred
