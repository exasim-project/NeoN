// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDdtDivLaplacian.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenLaplacian.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
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
    const bool bounded
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

                // upwind weight: 1 if flux leaves owner (flux > 0), 0 otherwise
                const auto flux = phiV[faceIdx];
                const auto w = (flux >= 0) ? scalar(1) : scalar(0);

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

                // Bounded-convection correction, folded into the cell loop: subtract the
                // cell's net internal face-flux outflow Σ_f φ_f (scaled by the div coeff) from
                // the diagonal — the implicit -Sp(div(phi)) term. signedOutflow = +flux for the
                // owner, -flux for the neighbour. This operator assembles internal faces only, so
                // no separate boundary-face correction is applied.
                if (bounded)
                {
                    const auto signedOutflow = (sign > 0) ? flux : -flux;
                    diagValue -= signedOutflow * cellCoeffA * one<ValueType>();
                }
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
        computeDdtDivLapImplCell(
            ls,
            dt,
            this->getVector(),
            flux_,
            gamma_,
            *faceNormalGradient_,
            coeffA_,
            coeffB_,
            std::dynamic_pointer_cast<la::CellBasedIterator>(ls.getMeshIterator()->get()),
            bounded_
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
