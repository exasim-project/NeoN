// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h> // For std::variant support
#include <nanobind/stl/string.h>  // For std::string support
#include <nanobind/stl/vector.h>  // For std::vector support

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/vector/vectorTypeDefs.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/mesh/unstructured/boundaryMesh.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(neon, m)
{
    m.doc() = "NeoN Python bindings";


    nb::class_<NeoN::SerialExecutor>(
        m, "SerialExecutor", "Reference executor for serial CPU execution"
    )
        .def(nb::init<>(), "Create a serial executor")
        .def("name", &NeoN::SerialExecutor::name, "Get the executor name")
        .def("__repr__", [](const NeoN::SerialExecutor&) { return "<SerialExecutor>"; })
        .def("__str__", [](const NeoN::SerialExecutor&) { return "SerialExecutor"; })
        .def(
            "__hash__",
            [](const NeoN::SerialExecutor&) { return std::hash<std::string> {}("SerialExecutor"); }
        )
        .def(
            "__eq__",
            [](const NeoN::SerialExecutor&, const NeoN::SerialExecutor&)
            {
                return true; // All SerialExecutors are equal
            }
        );

    nb::class_<NeoN::CPUExecutor>(m, "CPUExecutor", "Executor for multicore CPU parallelization")
        .def(nb::init<>(), "Create a CPU executor")
        .def("name", &NeoN::CPUExecutor::name, "Get the executor name")
        .def("__repr__", [](const NeoN::CPUExecutor&) { return "<CPUExecutor>"; })
        .def("__str__", [](const NeoN::CPUExecutor&) { return "CPUExecutor"; })
        .def(
            "__hash__",
            [](const NeoN::CPUExecutor&) { return std::hash<std::string> {}("CPUExecutor"); }
        )
        .def(
            "__eq__",
            [](const NeoN::CPUExecutor&, const NeoN::CPUExecutor&)
            {
                return true; // All CPUExecutors are equal
            }
        );

    nb::class_<NeoN::GPUExecutor>(m, "GPUExecutor", "Executor for GPU offloading")
        .def(nb::init<>(), "Create a GPU executor")
        .def("name", &NeoN::GPUExecutor::name, "Get the executor name")
        .def("__repr__", [](const NeoN::GPUExecutor&) { return "<GPUExecutor>"; })
        .def("__str__", [](const NeoN::GPUExecutor&) { return "GPUExecutor"; })
        .def(
            "__hash__",
            [](const NeoN::GPUExecutor&) { return std::hash<std::string> {}("GPUExecutor"); }
        )
        .def(
            "__eq__",
            [](const NeoN::GPUExecutor&, const NeoN::GPUExecutor&)
            {
                return true; // All GPUExecutors are equal
            }
        );

    m.def(
        "executor_name",
        [](const NeoN::Executor& exec)
        { return std::visit([](auto&& e) { return e.name(); }, exec); },
        "exec"_a,
        "Get the name of an executor (works with SerialExecutor, CPUExecutor, or GPUExecutor)"
    );

    m.def(
        "executor_repr",
        [](const NeoN::Executor& exec)
        { return "<Executor: " + std::visit([](auto&& e) { return e.name(); }, exec) + ">"; },
        "exec"_a,
        "Get string representation of an executor"
    );

    m.def(
        "is_serial",
        [](const NeoN::Executor& exec)
        { return std::holds_alternative<NeoN::SerialExecutor>(exec); },
        "exec"_a,
        "Check if an executor is a SerialExecutor"
    );

    m.def(
        "is_cpu",
        [](const NeoN::Executor& exec) { return std::holds_alternative<NeoN::CPUExecutor>(exec); },
        "exec"_a,
        "Check if an executor is a CPUExecutor"
    );

    m.def(
        "is_gpu",
        [](const NeoN::Executor& exec) { return std::holds_alternative<NeoN::GPUExecutor>(exec); },
        "exec"_a,
        "Check if an executor is a GPUExecutor"
    );

    // Vec3
    // positions, velocities, normals, and other 3-component quantities.
    nb::class_<NeoN::Vec3>(m, "Vec3", "A 3D vector primitive")
        .def(nb::init<>(), "Create a zero-initialized Vec3")
        .def(
            nb::init<NeoN::scalar, NeoN::scalar, NeoN::scalar>(),
            "x"_a,
            "y"_a,
            "z"_a,
            "Create a Vec3 with specified x, y, z components"
        )
        .def(
            nb::init<NeoN::scalar>(),
            "value"_a,
            "Create a Vec3 with all components set to the same value"
        )

        .def(
            "__getitem__",
            [](const NeoN::Vec3& v, size_t i)
            {
                if (i >= 3) throw std::out_of_range("Vec3 index out of range");
                return v[i];
            },
            "i"_a,
            "Get component by index (0=x, 1=y, 2=z)"
        )
        .def(
            "__setitem__",
            [](NeoN::Vec3& v, size_t i, NeoN::scalar value)
            {
                if (i >= 3) throw std::out_of_range("Vec3 index out of range");
                v[i] = value;
            },
            "i"_a,
            "value"_a,
            "Set component by index"
        )

        .def_prop_rw(
            "x",
            [](const NeoN::Vec3& v) { return v[0]; },
            [](NeoN::Vec3& v, NeoN::scalar val) { v[0] = val; },
            "X component"
        )
        .def_prop_rw(
            "y",
            [](const NeoN::Vec3& v) { return v[1]; },
            [](NeoN::Vec3& v, NeoN::scalar val) { v[1] = val; },
            "Y component"
        )
        .def_prop_rw(
            "z",
            [](const NeoN::Vec3& v) { return v[2]; },
            [](NeoN::Vec3& v, NeoN::scalar val) { v[2] = val; },
            "Z component"
        )

        .def(
            "__add__",
            [](const NeoN::Vec3& a, const NeoN::Vec3& b) { return a + b; },
            "Add two Vec3s"
        )
        .def(
            "__sub__",
            [](const NeoN::Vec3& a, const NeoN::Vec3& b) { return a - b; },
            "Subtract two Vec3s"
        )
        .def(
            "__mul__",
            [](const NeoN::Vec3& v, NeoN::scalar s) { return v * s; },
            "Multiply Vec3 by scalar"
        )
        .def(
            "__rmul__",
            [](const NeoN::Vec3& v, NeoN::scalar s) { return s * v; },
            "Multiply scalar by Vec3"
        )

        .def(
            "__iadd__",
            [](NeoN::Vec3& a, const NeoN::Vec3& b) -> NeoN::Vec3&
            {
                a += b;
                return a;
            }
        )
        .def(
            "__isub__",
            [](NeoN::Vec3& a, const NeoN::Vec3& b) -> NeoN::Vec3&
            {
                a -= b;
                return a;
            }
        )
        .def(
            "__imul__",
            [](NeoN::Vec3& v, NeoN::scalar s) -> NeoN::Vec3&
            {
                v *= s;
                return v;
            }
        )

        .def("__eq__", [](const NeoN::Vec3& a, const NeoN::Vec3& b) { return a == b; })

        .def(
            "dot",
            [](const NeoN::Vec3& a, const NeoN::Vec3& b) { return a & b; },
            "other"_a,
            "Compute dot product with another Vec3"
        )

        .def(
            "mag",
            [](const NeoN::Vec3& v) { return NeoN::mag(v); },
            "Compute magnitude (length) of the vector"
        )

        .def(
            "__repr__",
            [](const NeoN::Vec3& v)
            {
                return "Vec3(" + std::to_string(v[0]) + ", " + std::to_string(v[1]) + ", "
                     + std::to_string(v[2]) + ")";
            }
        )
        .def(
            "__str__",
            [](const NeoN::Vec3& v)
            {
                return "(" + std::to_string(v[0]) + ", " + std::to_string(v[1]) + ", "
                     + std::to_string(v[2]) + ")";
            }
        )
        .def(
            "__len__",
            [](const NeoN::Vec3&) { return 3; },
            "Return the number of components (always 3)"
        );

    m.def(
        "mag",
        static_cast<NeoN::scalar (*)(const NeoN::Vec3&)>(&NeoN::mag),
        "vec"_a,
        "Compute magnitude of a Vec3"
    );

    m.def(
        "dot",
        [](const NeoN::Vec3& a, const NeoN::Vec3& b) { return a & b; },
        "vec1"_a,
        "vec2"_a,
        "Compute dot product of two Vec3s"
    );


    // Vector<T> is a container that supports multiple execution spaces (CPU, GPU).
    // It is similar to std::vector but with support for heterogeneous computing via executors.

    // Vector<scalar> - for scalar fields
    nb::class_<NeoN::Vector<NeoN::scalar>>(
        m, "ScalarVector", "A vector of scalar values with executor support"
    )
        .def(
            nb::init<const NeoN::Executor&, NeoN::localIdx>(),
            "exec"_a,
            "size"_a,
            "Create an uninitialized ScalarVector of given size on an executor"
        )
        .def(
            nb::init<const NeoN::Executor&, NeoN::localIdx, NeoN::scalar>(),
            "exec"_a,
            "size"_a,
            "value"_a,
            "Create a ScalarVector with uniform value"
        )
        .def(
            nb::init<const NeoN::Executor&, std::vector<NeoN::scalar>>(),
            "exec"_a,
            "values"_a,
            "Create a ScalarVector from a Python list"
        )

        .def("size", &NeoN::Vector<NeoN::scalar>::size, "Get the size of the vector")
        .def("empty", &NeoN::Vector<NeoN::scalar>::empty, "Check if the vector is empty")
        .def("__len__", &NeoN::Vector<NeoN::scalar>::size, "Get the size (for Python len())")

        .def(
            "exec",
            &NeoN::Vector<NeoN::scalar>::exec,
            "Get the executor associated with this vector"
        )

        .def(
            "copy_to_host",
            static_cast<NeoN::Vector<NeoN::scalar> (NeoN::Vector<NeoN::scalar>::*)() const>(
                &NeoN::Vector<NeoN::scalar>::copyToHost
            ),
            "Copy the vector data to host (CPU) and return as new vector"
        )

        .def(
            "resize",
            &NeoN::Vector<NeoN::scalar>::resize,
            "size"_a,
            "Resize the vector to a new size"
        )

        .def(
            "__repr__",
            [](const NeoN::Vector<NeoN::scalar>& v)
            { return "<ScalarVector size=" + std::to_string(v.size()) + ">"; }
        )
        .def(
            "__str__",
            [](const NeoN::Vector<NeoN::scalar>& v)
            { return "ScalarVector(size=" + std::to_string(v.size()) + ")"; }
        );

    // Vector<Vec3> - for vector fields
    nb::class_<NeoN::Vector<NeoN::Vec3>>(
        m, "VectorVector", "A vector of Vec3 values with executor support"
    )
        .def(
            nb::init<const NeoN::Executor&, NeoN::localIdx>(),
            "exec"_a,
            "size"_a,
            "Create an uninitialized VectorVector of given size"
        )
        .def(
            nb::init<const NeoN::Executor&, NeoN::localIdx, NeoN::Vec3>(),
            "exec"_a,
            "size"_a,
            "value"_a,
            "Create a VectorVector with uniform value"
        )
        .def(
            nb::init<const NeoN::Executor&, std::vector<NeoN::Vec3>>(),
            "exec"_a,
            "values"_a,
            "Create a VectorVector from a Python list"
        )
        .def("size", &NeoN::Vector<NeoN::Vec3>::size)
        .def("empty", &NeoN::Vector<NeoN::Vec3>::empty)
        .def("__len__", &NeoN::Vector<NeoN::Vec3>::size)
        .def("exec", &NeoN::Vector<NeoN::Vec3>::exec)
        .def(
            "copy_to_host",
            static_cast<NeoN::Vector<NeoN::Vec3> (NeoN::Vector<NeoN::Vec3>::*)() const>(
                &NeoN::Vector<NeoN::Vec3>::copyToHost
            )
        )
        .def("resize", &NeoN::Vector<NeoN::Vec3>::resize, "size"_a)
        .def(
            "__repr__",
            [](const NeoN::Vector<NeoN::Vec3>& v)
            { return "<VectorVector size=" + std::to_string(v.size()) + ">"; }
        );

    // Vector<label> - for integer indices
    nb::class_<NeoN::Vector<NeoN::label>>(
        m, "LabelVector", "A vector of label (integer) values with executor support"
    )
        .def(nb::init<const NeoN::Executor&, NeoN::localIdx>(), "exec"_a, "size"_a)
        .def(
            nb::init<const NeoN::Executor&, NeoN::localIdx, NeoN::label>(),
            "exec"_a,
            "size"_a,
            "value"_a
        )
        .def(nb::init<const NeoN::Executor&, std::vector<NeoN::label>>(), "exec"_a, "values"_a)
        .def("size", &NeoN::Vector<NeoN::label>::size)
        .def("empty", &NeoN::Vector<NeoN::label>::empty)
        .def("__len__", &NeoN::Vector<NeoN::label>::size)
        .def("exec", &NeoN::Vector<NeoN::label>::exec)
        .def(
            "copy_to_host",
            static_cast<NeoN::Vector<NeoN::label> (NeoN::Vector<NeoN::label>::*)() const>(
                &NeoN::Vector<NeoN::label>::copyToHost
            )
        )
        .def("resize", &NeoN::Vector<NeoN::label>::resize, "size"_a)
        .def(
            "__repr__",
            [](const NeoN::Vector<NeoN::label>& v)
            { return "<LabelVector size=" + std::to_string(v.size()) + ">"; }
        );

    // Boundary Mesh
    //  It contains fields for face centres, normals, areas, connectivity, and
    //  interpolation weights. The offset vector allows accessing boundary-specific data.
    nb::class_<NeoN::BoundaryMesh>(
        m, "BoundaryMesh", "Boundary mesh containing information about boundary faces"
    )
        .def(
            nb::init<
                const NeoN::Executor&,
                NeoN::labelVector,
                NeoN::vectorVector,
                NeoN::vectorVector,
                NeoN::vectorVector,
                NeoN::scalarVector,
                NeoN::vectorVector,
                NeoN::vectorVector,
                NeoN::scalarVector,
                NeoN::scalarVector,
                std::vector<NeoN::localIdx>>(),
            "exec"_a,
            "face_cells"_a,
            "cf"_a,
            "cn"_a,
            "sf"_a,
            "mag_sf"_a,
            "nf"_a,
            "delta"_a,
            "weights"_a,
            "delta_coeffs"_a,
            "offset"_a,
            "Create a BoundaryMesh with all geometric data"
        )

        .def(
            "face_cells",
            static_cast<const NeoN::labelVector& (NeoN::BoundaryMesh::*)() const>(
                &NeoN::BoundaryMesh::faceCells
            ),
            nb::rv_policy::reference_internal,
            "Get the vector of face cell indices (which cell each boundary face belongs to)"
        )
        .def(
            "cf",
            static_cast<const NeoN::vectorVector& (NeoN::BoundaryMesh::*)() const>(
                &NeoN::BoundaryMesh::cf
            ),
            nb::rv_policy::reference_internal,
            "Get the vector of face centres"
        )
        .def(
            "cn",
            static_cast<const NeoN::vectorVector& (NeoN::BoundaryMesh::*)() const>(
                &NeoN::BoundaryMesh::cn
            ),
            nb::rv_policy::reference_internal,
            "Get the vector of neighbor cell centres"
        )
        .def(
            "sf",
            static_cast<const NeoN::vectorVector& (NeoN::BoundaryMesh::*)() const>(
                &NeoN::BoundaryMesh::sf
            ),
            nb::rv_policy::reference_internal,
            "Get the vector of face area normals (area * unit normal)"
        )
        .def(
            "mag_sf",
            static_cast<const NeoN::scalarVector& (NeoN::BoundaryMesh::*)() const>(
                &NeoN::BoundaryMesh::magSf
            ),
            nb::rv_policy::reference_internal,
            "Get the vector of face area magnitudes"
        )
        .def(
            "nf",
            static_cast<const NeoN::vectorVector& (NeoN::BoundaryMesh::*)() const>(
                &NeoN::BoundaryMesh::nf
            ),
            nb::rv_policy::reference_internal,
            "Get the vector of face unit normals"
        )
        .def(
            "delta",
            static_cast<const NeoN::vectorVector& (NeoN::BoundaryMesh::*)() const>(
                &NeoN::BoundaryMesh::delta
            ),
            nb::rv_policy::reference_internal,
            "Get the vector of delta vectors (face centre to cell centre)"
        )
        .def(
            "weights",
            static_cast<const NeoN::scalarVector& (NeoN::BoundaryMesh::*)() const>(
                &NeoN::BoundaryMesh::weights
            ),
            nb::rv_policy::reference_internal,
            "Get the vector of interpolation weights"
        )
        .def(
            "delta_coeffs",
            static_cast<const NeoN::scalarVector& (NeoN::BoundaryMesh::*)() const>(
                &NeoN::BoundaryMesh::deltaCoeffs
            ),
            nb::rv_policy::reference_internal,
            "Get the vector of delta coefficients (cell-to-face distances)"
        )
        .def(
            "offset",
            &NeoN::BoundaryMesh::offset,
            nb::rv_policy::reference_internal,
            "Get the offset vector for accessing boundary-specific data"
        )

        .def(
            "__repr__",
            [](const NeoN::BoundaryMesh& bm)
            { return "<BoundaryMesh with " + std::to_string(bm.faceCells().size()) + " faces>"; }
        );

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
            &NeoN::UnstructuredMesh::points,
            nb::rv_policy::reference_internal,
            "Get the vector of mesh points (vertices)"
        )
        .def(
            "cell_volumes",
            &NeoN::UnstructuredMesh::cellVolumes,
            nb::rv_policy::reference_internal,
            "Get the vector of cell volumes"
        )
        .def(
            "cell_centres",
            &NeoN::UnstructuredMesh::cellCentres,
            nb::rv_policy::reference_internal,
            "Get the vector of cell centres"
        )
        .def(
            "face_centres",
            &NeoN::UnstructuredMesh::faceCentres,
            nb::rv_policy::reference_internal,
            "Get the vector of face centres"
        )
        .def(
            "face_areas",
            &NeoN::UnstructuredMesh::faceAreas,
            nb::rv_policy::reference_internal,
            "Get the vector of face area normals"
        )
        .def(
            "mag_face_areas",
            &NeoN::UnstructuredMesh::magFaceAreas,
            nb::rv_policy::reference_internal,
            "Get the vector of face area magnitudes"
        )

        .def(
            "face_owner",
            &NeoN::UnstructuredMesh::faceOwner,
            nb::rv_policy::reference_internal,
            "Get the vector of face owner cell indices"
        )
        .def(
            "face_neighbour",
            &NeoN::UnstructuredMesh::faceNeighbour,
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
