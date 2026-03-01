// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/partition/dualGraph.hpp"

#include <cstdint>
#include <vector>

namespace NeoN::partition
{

DualGraph buildDualGraph(const UnstructuredMesh& mesh)
{
    const localIdx nCells = mesh.nCells();
    const localIdx nInternal = mesh.nInternalFaces();

    auto hostOwner = mesh.faceOwner().copyToHost();
    auto hostNeighbour = mesh.faceNeighbour().copyToHost();
    auto ownerView = hostOwner.view();
    auto neighbourView = hostNeighbour.view();

    // Build per-cell adjacency lists
    std::vector<std::vector<std::int32_t>> adj(static_cast<std::size_t>(nCells));
    for (localIdx f = 0; f < nInternal; ++f)
    {
        auto own = static_cast<std::int32_t>(ownerView[f]);
        auto nb = static_cast<std::int32_t>(neighbourView[f]);
        adj[static_cast<std::size_t>(own)].push_back(nb);
        adj[static_cast<std::size_t>(nb)].push_back(own);
    }

    // Convert to CSR
    DualGraph dg;
    dg.nCells = static_cast<std::int32_t>(nCells);
    dg.xadj.resize(static_cast<std::size_t>(nCells + 1));
    std::int32_t pos = 0;
    for (localIdx c = 0; c < nCells; ++c)
    {
        dg.xadj[static_cast<std::size_t>(c)] = pos;
        pos += static_cast<std::int32_t>(adj[static_cast<std::size_t>(c)].size());
    }
    dg.xadj[static_cast<std::size_t>(nCells)] = pos;

    dg.adjncy.reserve(static_cast<std::size_t>(pos));
    for (localIdx c = 0; c < nCells; ++c)
        for (auto nb : adj[static_cast<std::size_t>(c)])
            dg.adjncy.push_back(nb);

    return dg;
}

} // namespace NeoN::partition
