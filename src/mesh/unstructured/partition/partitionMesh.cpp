// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/partition/partitionMesh.hpp"
#include "NeoN/mesh/unstructured/partition/dualGraph.hpp"

#include <stdexcept>
#include <vector>

#ifdef NeoN_WITH_METIS
#include <metis.h>
#endif

namespace NeoN::partition
{

std::vector<int> partitionMesh(const UnstructuredMesh& mesh, int nParts)
{
    // Trivial case: no METIS call needed
    if (nParts == 1)
    {
        return std::vector<int>(static_cast<std::size_t>(mesh.nCells()), 0);
    }

#ifdef NeoN_WITH_METIS
    auto dg = buildDualGraph(mesh);

    // METIS API uses idx_t; DualGraph stores int32_t which is compatible
    static_assert(sizeof(std::int32_t) == sizeof(idx_t), "METIS idx_t must be 32-bit");

    idx_t ncon = 1;
    idx_t objval = 0;
    idx_t nParts_t = static_cast<idx_t>(nParts);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    idx_t* xadj = reinterpret_cast<idx_t*>(dg.xadj.data());
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    idx_t* adjncy = reinterpret_cast<idx_t*>(dg.adjncy.data());
    std::vector<idx_t> part(static_cast<std::size_t>(dg.nCells));

    int ret = METIS_PartGraphKway(
        &dg.nCells,
        &ncon,
        xadj,
        adjncy,
        nullptr, // vwgt
        nullptr, // vsize
        nullptr, // adjwgt
        &nParts_t,
        nullptr, // tpwgts
        nullptr, // ubvec
        nullptr, // options
        &objval,
        part.data()
    );

    if (ret != METIS_OK)
    {
        throw std::runtime_error("METIS_PartGraphKway failed");
    }

    return {part.begin(), part.end()};
#else
    (void)nParts;
    throw std::runtime_error("partitionMesh: NeoN was built without METIS support");
#endif
}

} // namespace NeoN::partition
