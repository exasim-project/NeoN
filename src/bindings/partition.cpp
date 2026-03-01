// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#ifdef NeoN_WITH_METIS
#include "NeoN/mesh/unstructured/partition/partitionMesh.hpp"
#include "NeoN/mesh/unstructured/partition/extractSubMesh.hpp"
#endif

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerPartition(nb::module_& m)
{
#ifdef NeoN_WITH_METIS
    m.def(
        "partition_mesh",
        &NeoN::partition::partitionMesh,
        "mesh"_a,
        "n_parts"_a,
        "Partition mesh cells into n_parts using METIS Kway. "
        "Returns list of part IDs (one per cell, 0-based)."
    );
    m.def(
        "extract_sub_mesh",
        &NeoN::partition::extractSubMesh,
        "mesh"_a,
        "cell_part"_a,
        "part_id"_a,
        "Extract a standalone sub-mesh for the given partition ID. "
        "Inter-partition faces become a procBoundary patch."
    );
#endif
}

} // namespace NeoN::bindings
