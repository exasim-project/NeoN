// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/macros.hpp"
#include "NeoN/linearAlgebra/meshIterationStrategies.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <numeric>
#include <vector>

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

sizeType CellBasedIterator::size() const
{
    return static_cast<sizeType>(cellBasedIteratorData_->size);
}

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

BatchedFaceIterator::BatchedFaceIterator() : batchedFaceData_(nullptr) {}

std::string BatchedFaceIterator::name() const { return "BatchedFace"; }

sizeType BatchedFaceIterator::size() const
{
    return static_cast<sizeType>(batchedFaceData_->nBatches);
}

std::shared_ptr<BatchedFaceIterator::BatchedFaceData> BatchedFaceIterator::getBatchedFaceData()
{
    return batchedFaceData_;
}

void BatchedFaceIterator::setComputeBatchedFaceData(const UnstructuredMesh& mesh)
{
    batchedFaceData_ = computeBatchedFaceData(mesh);
}

namespace
{

// Lowest colour that neither cell has used yet; records it in both masks.
localIdx assignColour(uint64_t& ownerMask, uint64_t& neighbourMask)
{
    const uint64_t free = ~(ownerMask | neighbourMask);
    NF_ASSERT(free != 0, "face colouring needs more than 64 colours");
    const auto colour = static_cast<localIdx>(std::countr_zero(free));
    const uint64_t bit = uint64_t(1) << colour;
    ownerMask |= bit;
    neighbourMask |= bit;
    return colour;
}

// Groups the faces [0, colour.size()) by colour: segments[c]..segments[c+1] hold the faces of
// colour c in ascending face order (a stable counting sort).
std::pair<std::vector<localIdx>, std::vector<localIdx>>
groupByColour(const std::vector<localIdx>& colour, localIdx nBatches)
{
    std::vector<localIdx> segments(static_cast<sizeType>(nBatches) + 1, 0);
    for (const auto c : colour)
    {
        ++segments[static_cast<sizeType>(c) + 1];
    }
    std::partial_sum(segments.begin(), segments.end(), segments.begin());
    std::vector<localIdx> faces(colour.size());
    auto next = segments;
    for (sizeType f = 0; f < colour.size(); ++f)
    {
        faces[static_cast<sizeType>(next[static_cast<sizeType>(colour[f])]++)] =
            static_cast<localIdx>(f);
    }
    return {faces, segments};
}

} // namespace

std::shared_ptr<BatchedFaceIterator::BatchedFaceData>
BatchedFaceIterator::computeBatchedFaceData(const UnstructuredMesh& mesh)
{
    const auto owner = mesh.faceOwners().copyToHost();
    const auto neighbour = mesh.faceNeighbors().copyToHost();
    const auto [ownerV, neighbourV] = views(owner, neighbour);
    const auto nCells = static_cast<sizeType>(mesh.nCells());
    const auto nInternalFaces = static_cast<localIdx>(mesh.nInternalFaces());

    std::vector<uint64_t> used(nCells, 0);
    std::vector<localIdx> colour(static_cast<sizeType>(nInternalFaces));
    localIdx nBatches = 0;
    for (localIdx f = 0; f < nInternalFaces; ++f)
    {
        auto& c = colour[static_cast<sizeType>(f)];
        c = assignColour(
            used[static_cast<sizeType>(ownerV[f])], used[static_cast<sizeType>(neighbourV[f])]
        );
        nBatches = std::max(nBatches, c + 1);
    }
    auto [faces, segments] = groupByColour(colour, nBatches);

    const auto exec = mesh.exec();
    return std::make_shared<BatchedFaceData>(
        nBatches,
        SegmentedVector<localIdx, localIdx>(
            Vector<localIdx>(exec, faces), Vector<localIdx>(exec, segments)
        )
    );
}

std::shared_ptr<const BatchedFaceIterator::BatchedFaceData> batchedFaceData(
    const std::shared_ptr<MeshIterationStrategy>& strategy, const UnstructuredMesh& mesh
)
{
    auto batched = std::dynamic_pointer_cast<BatchedFaceIterator>(strategy);
    if (!batched)
    {
        return nullptr;
    }
    if (!batched->getBatchedFaceData())
    {
        batched->setComputeBatchedFaceData(mesh);
    }
    return batched->getBatchedFaceData();
}

}
