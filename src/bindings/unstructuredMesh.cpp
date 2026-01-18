// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/vector/vectorTypeDefs.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/mesh/unstructured/boundaryMesh.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerUnstructuredMesh(nb::module_& m)
{
    // unstructured Mesh
    // geometric data (cell volumes, centres, face areas, etc.) and topological
    // connectivity (face owners, neighbors). It also contains a BoundaryMesh
    nb::class_<NeoN::UnstructuredMesh>(
        m, "UnstructuredMesh", "Unstructured mesh with cells, faces, and boundaries"
    )
        .def(
            nb::init<
                NeoN::vectorVector,
                NeoN::scalarVector,
                NeoN::vectorVector,
                NeoN::vectorVector,
                NeoN::vectorVector,
                NeoN::scalarVector,
                NeoN::labelVector,
                NeoN::labelVector,
                NeoN::localIdx,
                NeoN::localIdx,
                NeoN::localIdx,
                NeoN::localIdx,
                NeoN::localIdx,
                NeoN::BoundaryMesh>(),
            "points"_a,
            "cell_volumes"_a,
            "cell_centres"_a,
            "face_areas"_a,
            "face_centres"_a,
            "mag_face_areas"_a,
            "face_owner"_a,
            "face_neighbour"_a,
            "n_cells"_a,
            "n_internal_faces"_a,
            "n_boundary_faces"_a,
            "n_boundaries"_a,
            "n_faces"_a,
            "boundary_mesh"_a,
            "Create an UnstructuredMesh with all data"
        )

        .def(
            "points",
            nb::overload_cast<>(&NeoN::UnstructuredMesh::points, nb::const_),
            nb::rv_policy::reference_internal,
            "Get the vector of mesh points (vertices)"
        )
        .def(
            "cell_volumes",
            nb::overload_cast<>(&NeoN::UnstructuredMesh::cellVolumes, nb::const_),
            nb::rv_policy::reference_internal,
            "Get the vector of cell volumes"
        )
        .def(
            "cell_centres",
            nb::overload_cast<>(&NeoN::UnstructuredMesh::cellCentres, nb::const_),
            nb::rv_policy::reference_internal,
            "Get the vector of cell centres"
        )
        .def(
            "face_centres",
            nb::overload_cast<>(&NeoN::UnstructuredMesh::faceCentres, nb::const_),
            nb::rv_policy::reference_internal,
            "Get the vector of face centres"
        )
        .def(
            "face_areas",
            nb::overload_cast<>(&NeoN::UnstructuredMesh::faceAreas, nb::const_),
            nb::rv_policy::reference_internal,
            "Get the vector of face area normals"
        )
        .def(
            "mag_face_areas",
            nb::overload_cast<>(&NeoN::UnstructuredMesh::magFaceAreas, nb::const_),
            nb::rv_policy::reference_internal,
            "Get the vector of face area magnitudes"
        )

        .def(
            "face_owner",
            nb::overload_cast<>(&NeoN::UnstructuredMesh::faceOwner, nb::const_),
            nb::rv_policy::reference_internal,
            "Get the vector of face owner cell indices"
        )
        .def(
            "face_neighbour",
            nb::overload_cast<>(&NeoN::UnstructuredMesh::faceNeighbour, nb::const_),
            nb::rv_policy::reference_internal,
            "Get the vector of face neighbor cell indices"
        )

        .def("n_cells", &NeoN::UnstructuredMesh::nCells, "Get the number of cells in the mesh")
        .def(
            "n_internal_faces",
            &NeoN::UnstructuredMesh::nInternalFaces,
            "Get the number of internal faces"
        )
        .def(
            "n_boundary_faces",
            &NeoN::UnstructuredMesh::nBoundaryFaces,
            "Get the number of boundary faces"
        )
        .def(
            "n_boundaries",
            &NeoN::UnstructuredMesh::nBoundaries,
            "Get the number of boundary patches"
        )
        .def(
            "n_faces",
            &NeoN::UnstructuredMesh::nFaces,
            "Get the total number of faces (internal + boundary)"
        )

        .def(
            "boundary_mesh",
            &NeoN::UnstructuredMesh::boundaryMesh,
            nb::rv_policy::reference_internal,
            "Get the boundary mesh"
        )
        .def(
            "exec",
            &NeoN::UnstructuredMesh::exec,
            nb::rv_policy::reference_internal,
            "Get the executor associated with this mesh"
        )

        .def(
            "__repr__",
            [](const NeoN::UnstructuredMesh& mesh)
            {
                return "<UnstructuredMesh: " + std::to_string(mesh.nCells()) + " cells, "
                     + std::to_string(mesh.nFaces()) + " faces, "
                     + std::to_string(mesh.nBoundaries()) + " boundaries>";
            }
        )
        .def(
            "__str__",
            [](const NeoN::UnstructuredMesh& mesh)
            {
                return "UnstructuredMesh(cells=" + std::to_string(mesh.nCells())
                     + ", faces=" + std::to_string(mesh.nFaces())
                     + ", boundaries=" + std::to_string(mesh.nBoundaries()) + ")";
            }
        );

    // Mesh factory functions

    // These are convenience functions to create simple test meshes without
    // needing external mesh files.

    m.def(
        "create_single_cell_mesh",
        &NeoN::createSingleCellMesh,
        "exec"_a,
        "Create a mesh with a single 2D cell in 3D space.\n\n"
        "The mesh has:\n"
        "- 1 cell centred at (0.5, 0.5, 0.0)\n"
        "- 4 boundary faces: left, top, right, bottom\n"
        "- 4 boundary patches\n\n"
        "Useful for testing and simple demonstrations."
    );

    m.def(
        "create_1d_uniform_mesh",
        &NeoN::create1DUniformMesh,
        "exec"_a,
        "n_cells"_a,
        "Create a uniform 1D mesh aligned with the x-axis.\n\n"
        "Args:\n"
        "    exec: Executor for parallel operations\n"
        "    n_cells: Number of cells in the mesh\n\n"
        "Each cell has a left and right face. Useful for 1D simulations\n"
        "and testing finite volume schemes."
    );
}

} // namespace NeoN::bindings
