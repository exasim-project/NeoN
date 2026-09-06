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
#include "NeoN/mesh/unstructured/boundaryMesh.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerBoundaryMesh(nb::module_& m)
{
    // Boundary Mesh
    //  It contains fields for face centers, normals, areas, connectivity, and
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
                std::vector<NeoN::localIdx>,
                NeoN::localIdx,
                std::vector<NeoN::localIdx>>(),
            "exec"_a,
            "face_owners"_a,
            "face_centers"_a,
            "owner_cell_centers"_a,
            "face_normals"_a,
            "face_areas"_a,
            "face_unit_normals"_a,
            "delta"_a,
            "weights"_a,
            "delta_coeffs"_a,
            "offset"_a,
            "procBoundaryPatches"_a,
            "neighbourRank"_a,
            "Create a BoundaryMesh with all geometric data"
        )

        .def(
            "face_owners",
            static_cast<const NeoN::labelVector& (NeoN::BoundaryMesh::*)() const>(
                &NeoN::BoundaryMesh::faceOwners
            ),
            nb::rv_policy::reference_internal,
            "Get the vector of face cell indices (which cell each boundary face belongs to)"
        )
        .def(
            "face_centers",
            static_cast<const NeoN::vectorVector& (NeoN::BoundaryMesh::*)() const>(
                &NeoN::BoundaryMesh::faceCenters
            ),
            nb::rv_policy::reference_internal,
            "Get the vector of face centers"
        )
        .def(
            "owner_cell_centers",
            static_cast<const NeoN::vectorVector& (NeoN::BoundaryMesh::*)() const>(
                &NeoN::BoundaryMesh::ownerCellCenters
            ),
            nb::rv_policy::reference_internal,
            "Get the vector of owner cell centers"
        )
        .def(
            "face_normals",
            static_cast<const NeoN::vectorVector& (NeoN::BoundaryMesh::*)() const>(
                &NeoN::BoundaryMesh::faceNormals
            ),
            nb::rv_policy::reference_internal,
            "Get the vector of face normal vectors"
        )
        .def(
            "face_areas",
            static_cast<const NeoN::scalarVector& (NeoN::BoundaryMesh::*)() const>(
                &NeoN::BoundaryMesh::faceAreas
            ),
            nb::rv_policy::reference_internal,
            "Get the vector of face areas "
        )
        .def(
            "face_unit_normals",
            static_cast<const NeoN::vectorVector& (NeoN::BoundaryMesh::*)() const>(
                &NeoN::BoundaryMesh::faceUnitNormals
            ),
            nb::rv_policy::reference_internal,
            "Get the vector of face unit normal vectors"
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
            nb::overload_cast<>(&NeoN::BoundaryMesh::offset, nb::const_),
            nb::rv_policy::reference_internal,
            "Get the offset vector for accessing boundary-specific data"
        )
        .def(
            "__repr__",
            [](const NeoN::BoundaryMesh& bm)
            { return "<BoundaryMesh with " + std::to_string(bm.faceOwners().size()) + " faces>"; }
        );
}

} // namespace NeoN::bindings
