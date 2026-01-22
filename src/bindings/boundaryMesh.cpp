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
}

} // namespace NeoN::bindings
