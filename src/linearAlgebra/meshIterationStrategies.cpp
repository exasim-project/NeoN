// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/macros.hpp"
#include "NeoN/linearAlgebra/meshIterationStrategies.hpp"

namespace NeoN::la
{

// MeshIteratorContext

void MeshIteratorContext::setStrategy(std::shared_ptr<MeshIterationStrategy> strategyIn)
{
    strategy = std::move(strategyIn);
}

std::shared_ptr<MeshIterationStrategy> MeshIteratorContext::get() const
{
    if (strategy)
    {
        return strategy;
    }
    return nullptr;
}

std::string MeshIteratorContext::name() const
{
    if (strategy)
    {
        return strategy->name();
    }
    return "";
}

// CellBasedIterator

CellBasedIterator::CellBasedIterator() : cellBasedIteratorData_(nullptr) {}

std::string CellBasedIterator::name() const { return "CellBased"; }

size_t CellBasedIterator::size() const { return static_cast<size_t>(cellBasedIteratorData_->size); }

std::shared_ptr<CellBasedIterator::CellBasedData> CellBasedIterator::getCellBasedData()
{
    return cellBasedIteratorData_;
}

template<typename SparsityType>
void CellBasedIterator::setComputeCellBasedData(
    const UnstructuredMesh& mesh,
    std::shared_ptr<const SparsityType> sparsityPattern,
    std::shared_ptr<const la::FaceToMatrixAddress> faceToMatrixAddress
)
{
    cellBasedIteratorData_ = computeCellBasedData(mesh, sparsityPattern, faceToMatrixAddress);
}

template<typename SparsityType>
std::shared_ptr<CellBasedIterator::CellBasedData> CellBasedIterator::computeCellBasedData(
    const UnstructuredMesh& mesh,
    std::shared_ptr<const SparsityType> sparsityPattern,
    const std::shared_ptr<const la::FaceToMatrixAddress> faceToMatrixAddress
)
{
    // Count faces per cell (owner and neighbour contributions)
    auto exec = mesh.exec();
    auto cellToFaceStencil = finiteVolume::cellCentred::CellToFaceStencil(mesh);
    auto cellInternalFaces = cellToFaceStencil.computeInternalStencil();
    auto owner = mesh.faceOwners();
    auto neighbour = mesh.faceNeighbors();

    // Pre-compute face neighbour, sign, and matrix column indices
    const auto ninternalFaceConnections = cellInternalFaces.size();
    Vector<localIdx> iFaceNeighbours(exec, ninternalFaceConnections);
    Vector<scalar> iFaceSigns(exec, ninternalFaceConnections);
    Vector<localIdx> matrixColumnIdx(exec, ninternalFaceConnections);

    const auto [ownOffs, neiOffs] =
        views(faceToMatrixAddress->ownerOffset(), faceToMatrixAddress->neighbourOffset());

    const auto [ownerV, neighbourV] = views(owner, neighbour);
    const auto [matrixRowOffs] = views(sparsityPattern->rowOffs());

    auto [iFaceNeighboursV, iFaceSignsV, matrixColumnIdxV] =
        views(iFaceNeighbours, iFaceSigns, matrixColumnIdx);

    auto [cellFacesValues, cellFacesSegments] = cellInternalFaces.views();

    parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx celli) {
            const auto faces = cellFacesSegments[celli + 1] - cellFacesSegments[celli];
            const auto startIdx = cellFacesSegments[celli];

            for (localIdx i = 0; i < faces; ++i)
            {
                const auto faceIdx = cellFacesValues[startIdx + i];
                const auto own = ownerV[faceIdx];
                const auto nei = neighbourV[faceIdx];

                const auto isOwner = (own == celli);
                const auto neiCell = isOwner ? nei : own;

                iFaceNeighboursV[startIdx + i] = neiCell;
                iFaceSignsV[startIdx + i] = isOwner ? 1.0 : -1.0;

                // Compute matrix column index
                const auto rowStart = matrixRowOffs[celli];
                if (isOwner)
                {
                    matrixColumnIdxV[startIdx + i] = rowStart + ownOffs[faceIdx];
                }
                else
                {
                    matrixColumnIdxV[startIdx + i] = rowStart + neiOffs[faceIdx];
                }
            }
        },
        "computeFaceData"
    );

    return std::make_shared<CellBasedData>(
        mesh.nCells(), cellInternalFaces, iFaceNeighbours, iFaceSigns, matrixColumnIdx
    );
}

template void CellBasedIterator::setComputeCellBasedData<CsrSparsityPattern<
    localIdx>>(const UnstructuredMesh&, std::shared_ptr<const CsrSparsityPattern<localIdx>>, std::shared_ptr<const FaceToMatrixAddress>);

template void CellBasedIterator::setComputeCellBasedData<CooSparsityPattern<
    localIdx>>(const UnstructuredMesh&, std::shared_ptr<const CooSparsityPattern<localIdx>>, std::shared_ptr<const FaceToMatrixAddress>);

template std::shared_ptr<CellBasedIterator::CellBasedData>
CellBasedIterator::computeCellBasedData<CsrSparsityPattern<
    localIdx>>(const UnstructuredMesh&, std::shared_ptr<const CsrSparsityPattern<localIdx>>, const std::shared_ptr<const FaceToMatrixAddress>);

template std::shared_ptr<CellBasedIterator::CellBasedData>
CellBasedIterator::computeCellBasedData<CooSparsityPattern<
    localIdx>>(const UnstructuredMesh&, std::shared_ptr<const CooSparsityPattern<localIdx>>, const std::shared_ptr<const FaceToMatrixAddress>);

}
