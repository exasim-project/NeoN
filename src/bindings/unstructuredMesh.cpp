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
    // geometric data (cell volumes, centers, face areas, etc.) and topological
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
                NeoN::BoundaryMesh>(),
            "points"_a,
            "cell_volumes"_a,
            "cell_centers"_a,
            "face_normals"_a,
            "face_centers"_a,
            "face_areas"_a,
            "face_owners"_a,
            "face_neighbors"_a,
            "boundary_mesh"_a,
            "Create an UnstructuredMesh with all data"
        )

        .def_prop_ro(
            "points",
            nb::overload_cast<>(&NeoN::UnstructuredMesh::points, nb::const_),
            nb::rv_policy::reference_internal,
            "Get the vector of mesh points (vertices)"
        )
        .def_prop_ro(
            "cell_volumes",
            nb::overload_cast<>(&NeoN::UnstructuredMesh::cellVolumes, nb::const_),
            nb::rv_policy::reference_internal,
            "Get the vector of cell volumes"
        )
        .def_prop_ro(
            "cell_centers",
            nb::overload_cast<>(&NeoN::UnstructuredMesh::cellCenters, nb::const_),
            nb::rv_policy::reference_internal,
            "Get the vector of cell centers"
        )
        .def_prop_ro(
            "face_centers",
            nb::overload_cast<>(&NeoN::UnstructuredMesh::faceCenters, nb::const_),
            nb::rv_policy::reference_internal,
            "Get the vector of face centers"
        )
        .def_prop_ro(
            "face_normals",
            nb::overload_cast<>(&NeoN::UnstructuredMesh::faceNormals, nb::const_),
            nb::rv_policy::reference_internal,
            "Get the vector of face area normals"
        )
        .def_prop_ro(
            "face_areas",
            nb::overload_cast<>(&NeoN::UnstructuredMesh::faceAreas, nb::const_),
            nb::rv_policy::reference_internal,
            "Get the vector of face area magnitudes"
        )

        .def_prop_ro(
            "face_owners",
            nb::overload_cast<>(&NeoN::UnstructuredMesh::faceOwners, nb::const_),
            nb::rv_policy::reference_internal,
            "Get the vector of face owner cell indices"
        )
        .def_prop_ro(
            "face_neighbors",
            nb::overload_cast<>(&NeoN::UnstructuredMesh::faceNeighbors, nb::const_),
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
            "n_total_faces",
            &NeoN::UnstructuredMesh::nTotalFaces,
            "Get the total number of faces (internal + boundary + processor faces)"
        )
        .def(
            "n_boundaries",
            &NeoN::UnstructuredMesh::nBoundaries,
            "Get the number of boundary patches"
        )
        .def(
            "boundary_mesh",
            nb::overload_cast<>(&NeoN::UnstructuredMesh::boundaryMesh, nb::const_),
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
                     + std::to_string(mesh.nBoundaryFaces()) + " boundary faces, "
                     + std::to_string(mesh.nBoundaries()) + " boundaries>";
            }
        )
        .def(
            "__str__",
            [](const NeoN::UnstructuredMesh& mesh)
            {
                return "UnstructuredMesh(cells=" + std::to_string(mesh.nCells())
                     + ", internal faces=" + std::to_string(mesh.nInternalFaces())
                     + ", boundary faces=" + std::to_string(mesh.nBoundaryFaces())
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
        "Lx"_a = 1.0,
        "Create a uniform 1D mesh aligned with the x-axis.\n\n"
        "Args:\n"
        "    exec: Executor for parallel operations\n"
        "    n_cells: Number of cells in the mesh\n"
        "    Lx: Length of the mesh in the x-direction (Default: 1.0)\n\n"
        "Each cell has a left and right face. Useful for 1D simulations\n"
        "and testing finite volume schemes."
    );
}

} // namespace NeoN::bindings
