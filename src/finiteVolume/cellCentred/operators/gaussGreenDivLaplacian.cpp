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
void computeDivLapImplFace(
    la::LinearSystem<ValueType>& ls,
    const VolumeField<ValueType>& U,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<ValueType>& divSurfInterp,
    //    const SurfaceInterpolation<ValueType>& lapSurfInterp,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto exec = phi.exec();
    //    const auto weights = divSurfInterp.weight(phi, U);
    const auto sp = ls.faceToMatrixAddress();
    // const auto weightsI = lapSurfInterp.weight(phi, U);

    auto matrix = ls.matrix().view();
    auto rhs = ls.rhs().view();
    const auto [gammaV, deltaV] =
        views(gamma.internalVector(), faceNormalGradient.deltaCoeffs().internalVector());
    const auto [diaOffV, ownOffV, neiOffV] =
        views(sp->diagOffset(), sp->ownerOffset(), sp->neighbourOffset());
    const auto [phiV, /* weightsV, */ ownV, neiV, magFaceAreaV] = views(
        phi.internalVector(),
        // weights.internalVector(),
        mesh.faceOwners(),
        mesh.faceNeighbors(),
        mesh.faceAreas()
    );

    auto oneV = one<ValueType>();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto own = ownV[facei];
            auto nei = neiV[facei];

            auto fluxDiv = phiV[facei];
            const auto weight = (phiV[facei] >= 0) ? 1.0 : 0.0;
            auto fluxLap = deltaV[facei] * gammaV[facei] * magFaceAreaV[facei];

            // add neighbour contribution upper
            auto rowNeiStart = matrix.sparsity.rowOffs[nei];
            auto rowOwnStart = matrix.sparsity.rowOffs[own];

            // FIXME after profiling this shouldnt be hardcoded
            auto coeffNeiA = coeffA[nei];
            auto coeffOwnA = coeffA[own];
            auto coeffNeiB = coeffB[nei];
            auto coeffOwnB = coeffB[own];

            auto valueDiv = -weight * coeffNeiA * fluxDiv;
            auto valueLap = coeffNeiB * fluxLap;

            auto valueA = valueDiv + valueLap;
            matrix.values[rowNeiStart + neiOffV[facei]] += valueA * oneV;
            Kokkos::atomic_sub(&matrix.values[rowOwnStart + diaOffV[own]], valueA * oneV);

            // upper triangular part
            // add owner contribution lower
            valueDiv = (1 - weight) * coeffOwnA * fluxDiv;
            valueLap = coeffOwnB * fluxLap;
            auto valueB = valueDiv + valueLap;

            matrix.values[rowOwnStart + ownOffV[facei]] += valueB * oneV;
            Kokkos::atomic_sub(&matrix.values[rowNeiStart + diaOffV[nei]], valueB * oneV);
        },
        "computeLocalGaussGreenDivCoefficients"
    );

    // FIXME
    const auto surfFaceCells = mesh.boundaryMesh().faceOwners().view();
    auto [/*bweights,*/ refGradient, value, valueFraction, refValue, deltaCoeffsA] = views(
        // weights.boundaryData().value(),
        U.boundaryData().refGrad(),
        U.boundaryData().value(),
        U.boundaryData().valueFraction(),
        U.boundaryData().refValue(),
        mesh.boundaryMesh().deltaCoeffs()
    );

    auto bRhs = ls.boundaryRhs().view();
    auto bValues = ls.boundaryMatrix().values().view();

    parallelFor(
        exec,
        {nInternalFaces, phiV.size()},
        NEON_LAMBDA(const localIdx facei) {
            auto oneV = one<ValueType>();
            auto bcfacei = facei - nInternalFaces;

            auto fluxDiv = phiV[facei];
            auto fluxLap = gammaV[facei] * magFaceAreaV[facei];

            auto own = surfFaceCells[bcfacei];
            auto rowOwnStart = matrix.sparsity.rowOffs[own];
            auto coeffAOwn = coeffA[own];
            auto coeffBOwn = coeffB[own];

            auto valFrac1 = valueFraction[bcfacei];
            auto valFrac2 = 1.0 - valFrac1;

            auto bweights = 1.0;

            // FIXME
            auto valueDiv = -bweights /*[bcfacei]*/ * coeffAOwn * fluxDiv * valFrac2;
            auto valueLap = deltaV[facei] * coeffBOwn * fluxLap * valFrac1;

            auto valueA = (valueDiv + valueLap) * oneV;

            Kokkos::atomic_sub(&matrix.values[rowOwnStart + diaOffV[own]], valueA);
            bValues[bcfacei] = valueA * (-1.0);

            // div
            auto valueRhsA = ((fluxDiv * coeffAOwn) * (valFrac1 * refValue[bcfacei]))
                           + valFrac2 * refGradient[bcfacei] * (1 / deltaCoeffsA[bcfacei]);
            // lap
            auto valueRhsB =
                fluxLap * coeffBOwn
                * (valFrac1 * refValue[bcfacei] * deltaV[facei] + valFrac2 * refGradient[bcfacei]);

            Kokkos::atomic_sub(&rhs[own], valueRhsA);
            Kokkos::atomic_sub(&rhs[own], valueRhsB);

            bRhs[bcfacei] = (valueRhsA + valueRhsB);
        },
        "computeInterfaceGaussGreenDivCoefficients"
    );
};

template<typename ValueType>
void computeDivLapImplCell(
    la::LinearSystem<ValueType>& ls,
    const VolumeField<ValueType>& U,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<ValueType>& divSurfInterp,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB,
    std::shared_ptr<la::CellBasedIterator> /*iterator*/
)
{
    // TODO: implement cell-based iteration; fall back to face-based for now
    computeDivLapImplFace(ls, U, phi, gamma, divSurfInterp, faceNormalGradient, coeffA, coeffB);
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
{
    // FIXME some sanity checks are needed
    // are div and lap field the same
}

template<typename ValueType>
void GaussGreenDivLaplacian<ValueType>::explicitOperation(Vector<ValueType>& /*source*/) const
{}

template<typename ValueType>
void GaussGreenDivLaplacian<ValueType>::implicitOperation(la::LinearSystem<ValueType>& ls) const
{
    // FIXME I dont know how we can end up with a nullptr here double check
    if (ls.getMeshIterator() == nullptr)
    {
        computeDivLapImplFace(
            ls,
            this->getVector(),
            flux_,
            gamma_,
            *divSurfaceInterpolation_.get(),
            *faceNormalGradient_.get(),
            coeffA_,
            coeffB_
        );
        return;
    }

    if (ls.getMeshIterator()->name() == "CellBased")
    {
        computeDivLapImplCell(
            ls,
            this->getVector(),
            flux_,
            gamma_,
            *divSurfaceInterpolation_.get(),
            *faceNormalGradient_.get(),
            coeffA_,
            coeffB_,
            std::dynamic_pointer_cast<la::CellBasedIterator>(ls.getMeshIterator()->get())
        );
        return;
    }
    if (ls.getMeshIterator()->name() == "FaceBased")
    {
        computeDivLapImplFace(
            ls,
            this->getVector(),
            flux_,
            gamma_,
            *divSurfaceInterpolation_.get(),
            *faceNormalGradient_.get(),
            coeffA_,
            coeffB_
        );
        return;
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
    divTokens.remove(0);
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

#define NN_DECLARE_COMPUTE_IMP_DIV(TYPENAME)                                                       \
    template void computeDivLapImplFace(                                                           \
        la::LinearSystem<TYPENAME>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceInterpolation<TYPENAME>&,                                                     \
        const FaceNormalGradient<TYPENAME>&,                                                       \
        const dsl::Coeff,                                                                          \
        const dsl::Coeff                                                                           \
    );                                                                                             \
    template void computeDivLapImplCell(                                                           \
        la::LinearSystem<TYPENAME>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceInterpolation<TYPENAME>&,                                                     \
        const FaceNormalGradient<TYPENAME>&,                                                       \
        const dsl::Coeff,                                                                          \
        const dsl::Coeff,                                                                          \
        std::shared_ptr<la::CellBasedIterator> iterator                                            \
    )

NN_DECLARE_COMPUTE_IMP_DIV(scalar);
NN_DECLARE_COMPUTE_IMP_DIV(Vec3);

template class GaussGreenDivLaplacian<scalar>;
template class GaussGreenDivLaplacian<Vec3>;

} // namespace NeoN::finiteVolume::cellCentred
