// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDivLaplacian.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeDivLaplacianIntImpl(
    la::LinearSystem<ValueType>& ls,
    const VolumeField<ValueType>& /*U*/,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<ValueType>& /*divSurfInterp*/,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
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
            auto valueA = (-weight * coeffNeiA * fluxDiv + coeffNeiB * fluxLap) * one<ValueType>();
            values[ma.lowerIdx(nei, facei)] += valueA;
            Kokkos::atomic_sub(&values[ma.diagIdx(own)], valueA);

            // upper triangular (own row, nei col)
            auto valueB =
                ((1.0 - weight) * coeffOwnA * fluxDiv + coeffOwnB * fluxLap) * one<ValueType>();
            values[ma.upperIdx(own, facei)] += valueB;
            Kokkos::atomic_sub(&values[ma.diagIdx(nei)], valueB);
        },
        "computeLocalGaussGreenDivLaplacianCoefficients"
    );
}

template<typename ValueType>
void computeDivLaplacianBoundImpl(
    la::LinearSystem<ValueType>& ls,
    const VolumeField<ValueType>& U,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<ValueType>& /*divSurfInterp*/,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
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
        U.boundaryData().refGrad(),
        U.boundaryData().valueFraction(),
        U.boundaryData().refValue(),
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
            auto valueA = (valueDiv + valueLap) * one<ValueType>();

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

template<typename ValueType>
void computeDivLaplacianProcBoundImpl(
    la::LinearSystem<ValueType>& ls,
    const VolumeField<ValueType>& /*U*/,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<ValueType>& /*divSurfInterp*/,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
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
            auto lapValue = lapFlux * ownCoeffB * one<ValueType>();

            // Div upwind contribution
            auto isOwnerFace = isOwner[bcfacei] > 0.0;
            auto sign = isOwnerFace ? scalar(-1) : scalar(1);
            auto bFlux = bPhiV[bcfacei];
            auto weight = isOwnerFace ? (bFlux >= 0 ? scalar(1) : scalar(0))
                                      : (bFlux >= 0 ? scalar(0) : scalar(1));
            auto divDiag = sign * weight * bFlux * ownCoeffA * one<ValueType>();
            auto divOffDiag = -sign * (scalar(1) - weight) * bFlux * ownCoeffA * one<ValueType>();

            auto diagValue = divDiag + lapValue;
            Kokkos::atomic_sub(&values[ma.diagIdx(cell)], diagValue);
            bndDiagValues[bcfacei] += diagValue;
            bOffValues[procFacei] += divOffDiag + lapValue;
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
    if (ls.getMeshIterator() != nullptr && ls.getMeshIterator()->name() == "CellBased")
    {
        NF_ERROR_EXIT("GaussGreenDivLaplacian does not support cell-based assembly.");
    }

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
    divTokens.remove(0);

    if (divTokens.get<std::string>(0) != "upwind")
    {
        NF_ERROR_EXIT(
            "GaussGreenDivLaplacian only supports 'Gauss upwind' for divSchemes, got: Gauss "
            << divTokens.get<std::string>(0)
        );
    }
    divSurfaceInterpolation_ =
        std::make_shared<SurfaceInterpolation<ValueType>>(this->field_.exec(), mesh, divTokens);
    laplTokens.remove(0);
    faceNormalGradient_ =
        std::make_shared<FaceNormalGradient<ValueType>>(this->field_.exec(), mesh, laplTokens);
    if (faceNormalGradient_->hasImplicitCorrection())
    {
        NF_ERROR_EXIT("GaussGreenDivLaplacian does not support non-orthogonal correction. "
                      "Use 'Gauss linear uncorrected' for laplacianSchemes.");
    }
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

#define NN_DECLARE_DIVLAP_IMPL(TYPENAME)                                                           \
    template void computeDivLaplacianIntImpl(                                                      \
        la::LinearSystem<TYPENAME>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceInterpolation<TYPENAME>&,                                                     \
        const FaceNormalGradient<TYPENAME>&,                                                       \
        const dsl::Coeff,                                                                          \
        const dsl::Coeff                                                                           \
    );                                                                                             \
    template void computeDivLaplacianBoundImpl(                                                    \
        la::LinearSystem<TYPENAME>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceInterpolation<TYPENAME>&,                                                     \
        const FaceNormalGradient<TYPENAME>&,                                                       \
        const dsl::Coeff,                                                                          \
        const dsl::Coeff                                                                           \
    );                                                                                             \
    template void computeDivLaplacianProcBoundImpl(                                                \
        la::LinearSystem<TYPENAME>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceInterpolation<TYPENAME>&,                                                     \
        const FaceNormalGradient<TYPENAME>&,                                                       \
        const dsl::Coeff,                                                                          \
        const dsl::Coeff                                                                           \
    )

NN_DECLARE_DIVLAP_IMPL(scalar);
NN_DECLARE_DIVLAP_IMPL(Vec3);

template class GaussGreenDivLaplacian<scalar>;
template class GaussGreenDivLaplacian<Vec3>;

} // namespace NeoN::finiteVolume::cellCentred
