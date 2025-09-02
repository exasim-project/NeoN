// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/segmentedVector.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"

namespace NeoN::la
{

const SparsityPattern& SparsityPattern::readOrCreate(const UnstructuredMesh& mesh)
{
    StencilDataBase& stencilDb = mesh.stencilDB();
    if (!stencilDb.contains("SparsityPattern"))
    {
        stencilDb.insert(std::string("SparsityPattern"), SparsityPattern(mesh));
    }
    return stencilDb.get<SparsityPattern>("SparsityPattern");
}

void updateSparsityPattern(const UnstructuredMesh& mesh, SparsityPattern& sp)
{
    // const auto faceOwnV = mesh.faceOwner().view();
    // const auto faceNeiV = mesh.faceNeighbour().view();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto exec = mesh.exec();
    auto nCells = mesh.nCells();

    // start with one to include the diagonal
    // TODO: currently the whole algorithm is performed in serial on the host
    auto nFacesPerCellH = Vector<localIdx>(SerialExecutor {}, nCells, 1);
    auto [neiOffsetH, ownOffsetH, diagOffsetH, faceOwnH, faceNeiH] = copyToHosts(
        sp.neighbourOffset(),
        sp.ownerOffset(),
        sp.diagOffset(),
        mesh.faceOwner(),
        mesh.faceNeighbour()
    );

    auto [nFacesPerCellHV, neiOffsetHV, ownOffsetHV, diagOffsetHV, faceOwnHV, faceNeiHV] =
        views(nFacesPerCellH, neiOffsetH, ownOffsetH, diagOffsetH, faceOwnH, faceNeiH);


    // accumulate number non-zeros per row
    // only the internalfaces define the sparsity pattern
    // get the number of faces per cell to allocate the correct size
    parallelFor(
        SerialExecutor {},
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            // hit on performance on serial
            auto own = faceOwnHV[facei];
            auto nei = faceNeiHV[facei];

            Kokkos::atomic_increment(&nFacesPerCellHV[own]);
            Kokkos::atomic_increment(&nFacesPerCellHV[nei]);
        }
    );

    // get number of total non-zeros
    auto rowOffsH = sp.rowOffs().copyToHost();
    auto rowOffsHV = rowOffsH.view();
    segmentsFromIntervals(nFacesPerCellH, rowOffsH);
    auto colIdxH = sp.colIdxs().copyToHost();
    auto colIdxHV = colIdxH.view();
    fill(nFacesPerCellH, 0); // reset nFacesPerCell

    // compute the lower triangular part of the matrix
    parallelFor(
        SerialExecutor {},
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            auto nei = faceNeiHV[facei];
            auto own = faceOwnHV[facei];

            // TODO this is probably inherently serial
            // return the oldValues
            // hit on performance on serial
            auto segIdxNei = Kokkos::atomic_fetch_add(&nFacesPerCellHV[nei], 1);
            neiOffsetHV[facei] = static_cast<uint8_t>(segIdxNei);

            auto startSegNei = rowOffsHV[nei];
            // neighbour --> current cell
            // colIdx --> needs to be store the owner
            Kokkos::atomic_assign(&colIdxHV[startSegNei + segIdxNei], own);
        }
    );

    map(
        nFacesPerCellH,
        KOKKOS_LAMBDA(const localIdx celli) {
            auto nFaces = nFacesPerCellHV[celli];
            diagOffsetHV[celli] = static_cast<uint8_t>(nFaces);
            colIdxHV[rowOffsHV[celli] + nFaces] = celli;
            return nFaces + 1;
        }
    );

    // compute the upper triangular part of the matrix
    parallelFor(
        SerialExecutor {},
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            auto nei = faceNeiHV[facei];
            auto own = faceOwnHV[facei];

            // return the oldValues
            // hit on performance on serial
            auto segIdxOwn =
                static_cast<uint8_t>(Kokkos::atomic_fetch_add(&nFacesPerCellHV[own], 1));
            ownOffsetHV[facei] = segIdxOwn;

            auto startSegOwn = rowOffsHV[own];
            // owner --> current cell
            // colIdx --> needs to be store the neighbour
            Kokkos::atomic_assign(&colIdxHV[startSegOwn + segIdxOwn], nei);
        }
    );
    // NOTE copy back to device
    sp.ownerOffset() = ownOffsetH.copyToExecutor(exec);
    sp.neighbourOffset() = neiOffsetH.copyToExecutor(exec);
    sp.diagOffset() = diagOffsetH.copyToExecutor(exec);
    sp.colIdxs() = colIdxH.copyToExecutor(exec);
    sp.rowOffs() = rowOffsH.copyToExecutor(exec);


    // sort colIdx
    // TODO: this implementation could be improved
    // by actually implementing the parallel sort on device
    // auto rowOffsH = sp.rowOffs().copyToHost();
    // auto colIdxH = sp.colIdxs().copyToHost();
    // auto [rowOffsHV, colIdxHV] = views(rowOffsH, colIdxH);
    // for (auto celli = 0; celli < rowOffsHV.size() - 1; celli++)
    // {
    //     auto start = rowOffsHV[celli];
    //     auto end = rowOffsHV[celli + 1];
    //     // detail::parallelSort(exec, {start, end}, &stencilValues[0]);
    //     auto sub = colIdxHV.subview(start, end - start);
    //     std::sort(sub.begin(), sub.end());
    // }
    // auto sortedValues = Vector(exec, &colIdxHV[0], colIdxHV.size(), SerialExecutor {});
    // sp.colIdxs() = sortedValues;
}

SparsityPattern createSparsity(const UnstructuredMesh& mesh)
{
    const auto exec = mesh.exec();
    const auto nCells = mesh.nCells();
    const auto nnzs = 2 * mesh.nInternalFaces() + nCells;
    auto ret = SparsityPattern(exec, nCells, nnzs);
    updateSparsityPattern(mesh, ret);
    return ret;
}

SparsityPattern::SparsityPattern(const UnstructuredMesh& mesh)
    : rowOffs_(mesh.exec(), mesh.nCells() + 1, 0),
      colIdxs_(mesh.exec(), mesh.nCells() + 2 * mesh.nInternalFaces(), 0),
      ownerOffset_(mesh.exec(), mesh.nInternalFaces(), 0),
      neighbourOffset_(mesh.exec(), mesh.nInternalFaces(), 0),
      diagOffset_(mesh.exec(), mesh.nCells(), 0)
{
    updateSparsityPattern(mesh, *this);
}


SparsityPattern::SparsityPattern(Executor exec, localIdx nRows, localIdx nnzs)
    : rowOffs_(exec, nRows + 1, 0), colIdxs_(exec, nnzs, 0),
      ownerOffset_(exec, (nnzs - nRows) / 2, 0), neighbourOffset_(exec, (nnzs - nRows) / 2, 0),
      diagOffset_(exec, nRows, 0)
{}

const NeoN::Array<uint8_t>& SparsityPattern::ownerOffset() const { return ownerOffset_; }

const NeoN::Array<uint8_t>& SparsityPattern::neighbourOffset() const { return neighbourOffset_; }

const NeoN::Array<uint8_t>& SparsityPattern::diagOffset() const { return diagOffset_; }

NeoN::Array<uint8_t>& SparsityPattern::ownerOffset() { return ownerOffset_; }

NeoN::Array<uint8_t>& SparsityPattern::neighbourOffset() { return neighbourOffset_; }

NeoN::Array<uint8_t>& SparsityPattern::diagOffset() { return diagOffset_; }

} // namespace NeoN::la
