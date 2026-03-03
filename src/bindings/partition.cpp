// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

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

    m.def(
        "get_patch_names",
        [](const NeoN::UnstructuredMesh& mesh) -> std::vector<std::string>
        {
            if (mesh.stencilDB().contains("io::patchNames"))
            {
                return *mesh.stencilDB().get<std::shared_ptr<std::vector<std::string>>>(
                    "io::patchNames"
                );
            }
            return {};
        },
        "mesh"_a,
        "Get boundary patch names from the mesh stencilDB."
    );

    m.def(
        "get_global_cell_ids",
        [](const NeoN::UnstructuredMesh& mesh) -> std::vector<NeoN::localIdx>
        {
            if (mesh.stencilDB().contains("partition::globalCellIds"))
            {
                return *mesh.stencilDB().get<std::shared_ptr<std::vector<NeoN::localIdx>>>(
                    "partition::globalCellIds"
                );
            }
            return {};
        },
        "mesh"_a,
        "Get the global cell IDs for a partitioned sub-mesh."
    );

    m.def(
        "get_ghost_cell_ids",
        [](const NeoN::UnstructuredMesh& mesh) -> std::vector<NeoN::localIdx>
        {
            if (mesh.stencilDB().contains("partition::ghostCellGlobalIds"))
            {
                return *mesh.stencilDB().get<std::shared_ptr<std::vector<NeoN::localIdx>>>(
                    "partition::ghostCellGlobalIds"
                );
            }
            return {};
        },
        "mesh"_a,
        "Get global cell IDs of ghost cells for a partitioned sub-mesh."
    );
}

} // namespace NeoN::bindings
