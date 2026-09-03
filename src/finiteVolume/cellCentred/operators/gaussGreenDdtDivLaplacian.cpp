// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <variant>

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDdtDivLaplacian.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenLaplacian.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType, typename WeightKernel>
static void computeDdtDivLapImplCell(
    la::LinearSystem<ValueType>& ls,
    scalar dt,
    const VolumeField<ValueType>& u,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB,
    std::shared_ptr<la::CellBasedIterator> iterator,
    WeightKernel weightKernel
)
{
    const auto exec = ls.exec();
    const UnstructuredMesh& mesh = phi.mesh();

    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    auto cellBasedData = iterator->getCellBasedData();
    NF_ASSERT(
        cellBasedData != nullptr,
        "CellBasedData not initialized - call setComputeCellBasedData before invoking the "
        "cell-based kernel"
    );
    auto [cellFacesValues, cellFacesSegments] = cellBasedData->cellFaces.views();
    auto faceSignV = cellBasedData->faceSign.view();
    auto matrixColumnIdxV = cellBasedData->matrixColumnIdx.view();

    const auto [phiV, gammaV, deltaV, magFaceAreaV, vol] = views(
        phi.internalVector(),
        gamma.internalVector(),
        faceNormalGradient.deltaCoeffs().internalVector(),
        mesh.faceAreas(),
        mesh.cellVolumes()
    );

    const auto oldVector = oldTime(u).internalVector().view();
    auto [rhs, values] = views(ls.rhs(), ls.matrix().values());

    // BDF1: ddt contributes V/dt to the diagonal and V/dt * u_old to the rhs
    const scalar a0a1 = 1.0 / dt;

    parallelFor(
        exec,
        {0, iterator->size()},
        NEON_LAMBDA(const localIdx celli) {
            // DDT contribution: \int_V du/dt dV ~ V/dt * (u^{n+1} - u^n)
            const auto ddtCoeff = a0a1 * vol[celli];
            auto diagValue = ddtCoeff * one<ValueType>();
            rhs[celli] += ddtCoeff * oldVector[celli];

            // Div + Laplacian contributions, looping over the faces of this cell
            const auto numFaces = cellFacesSegments[celli + 1] - cellFacesSegments[celli];
            const auto startIdx = cellFacesSegments[celli];
            const auto cellCoeffA = coeffA[celli];
            const auto cellCoeffB = coeffB[celli];

            for (localIdx i = 0; i < numFaces; ++i)
            {
                const auto faceIdx = cellFacesValues[startIdx + i];
                const auto sign = faceSignV[startIdx + i];

                // Div interpolation weight supplied by the configured scheme's device-callable
                // inline kernel (upwind: 1 if flux leaves owner; linear: geometric weight),
                // replacing the previously hard-coded upwind so this operator honours fvSchemes.
                // NOTE: corrected schemes (e.g. linearUpwind) contribute only their upwind weight
                // here; their explicit gradient correction is not (yet) added as a deferred RHS
                // source, matching this operator's prior first-order div behaviour.
                const auto flux = phiV[faceIdx];
                const auto w = weightKernel.weight(faceIdx, flux);

                // Laplacian face coefficient: δ_f · γ_f · |S_f|
                const auto fluxLap = deltaV[faceIdx] * gammaV[faceIdx] * magFaceAreaV[faceIdx];

                ValueType offDiag;
                ValueType diagContrib;

                if (sign > 0) // celli is owner: off-diagonal is upper triangular entry
                {
                    offDiag =
                        (flux * (1.0 - w) * cellCoeffA + fluxLap * cellCoeffB) * one<ValueType>();
                    diagContrib = (flux * w * cellCoeffA - fluxLap * cellCoeffB) * one<ValueType>();
                }
                else // celli is neighbor: off-diagonal is lower triangular entry
                {
                    offDiag = (-flux * w * cellCoeffA + fluxLap * cellCoeffB) * one<ValueType>();
                    diagContrib =
                        (-flux * (1.0 - w) * cellCoeffA - fluxLap * cellCoeffB) * one<ValueType>();
                }

                values[matrixColumnIdxV[startIdx + i]] += offDiag;
                diagValue += diagContrib;
            }

            values[ma.diagIdx(celli)] += diagValue;
        },
        "GaussGreenDdtDivLaplacian::cellLoop"
    );
}

template<typename ValueType>
void GaussGreenDdtDivLaplacian<ValueType>::implicitOperation(
    la::LinearSystem<ValueType>& ls, scalar /*t*/, scalar dt
) const
{
    if (ls.getMeshIterator() == nullptr)
    {
        NF_ERROR_EXIT("Not implemented");
    }

    if (ls.getMeshIterator()->name() == "CellBased")
    {
        auto iterator =
            std::dynamic_pointer_cast<la::CellBasedIterator>(ls.getMeshIterator()->get());
        // Dispatch on the configured div scheme's inline weight kernel on the host; the concrete
        // kernel is captured by value into the device loop (no std::variant on the device).
        std::visit(
            [&](auto&& kernel)
            {
                computeDdtDivLapImplCell(
                    ls,
                    dt,
                    this->getVector(),
                    flux_,
                    gamma_,
                    *faceNormalGradient_,
                    coeffA_,
                    coeffB_,
                    iterator,
                    kernel
                );
            },
            divSurfaceInterpolation_->inlineWeightKernel(flux_)
        );
        computeLaplacianNonOrthCorrImpl(
            ls, gamma_, this->getVector(), coeffB_, *faceNormalGradient_
        );
        return;
    }
    if (ls.getMeshIterator()->name() == "FaceBased")
    {
        NF_ERROR_EXIT("Not implemented");
    }
}

template class GaussGreenDdtDivLaplacian<scalar>;
template class GaussGreenDdtDivLaplacian<Vec3>;

} // namespace NeoN::finiteVolume::cellCentred
