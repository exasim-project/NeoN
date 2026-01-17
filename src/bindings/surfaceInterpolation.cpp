// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "bindings.hpp"

#include "NeoN/finiteVolume/cellCentred/interpolation/linear.hpp" // these are required for registration
#include "NeoN/finiteVolume/cellCentred/interpolation/upwind.hpp" // these are required for registration

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

template<typename T>
void declare_surface_interpolation(nb::module_& m, const char* name)
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;

    nb::class_<fvcc::SurfaceInterpolation<T>>(m, name)
        .def(
            "__init__",
            [](fvcc::SurfaceInterpolation<T>* self,
               const NeoN::Executor& exec,
               const NeoN::UnstructuredMesh& mesh,
               const NeoN::Input& input)
            { new (self) fvcc::SurfaceInterpolation<T>(exec, mesh, input); },
            "exec"_a,
            "mesh"_a,
            "input"_a
        )
        .def(
            "interpolate",
            [](const fvcc::SurfaceInterpolation<T>& self, const fvcc::VolumeField<T>& src)
            { return self.interpolate(src); },
            "src"_a,
            "Interpolate a volume field to a surface field"
        )
        .def(
            "interpolate_into",
            [](const fvcc::SurfaceInterpolation<T>& self,
               const fvcc::VolumeField<T>& src,
               fvcc::SurfaceField<T>& dst) { self.interpolate(src, dst); },
            "src"_a,
            "dst"_a,
            "Interpolate into an existing surface field"
        )
        .def(
            "weight",
            [](const fvcc::SurfaceInterpolation<T>& interp, const fvcc::VolumeField<T>& src)
            { return interp.weight(src); },
            "src"_a,
            "Compute interpolation weights for a volume field"
        );
}

void registerSurfaceInterpolation(nb::module_& m)
{
    declare_surface_interpolation<NeoN::scalar>(m, "SurfaceInterpolationScalar");
    declare_surface_interpolation<NeoN::Vec3>(m, "SurfaceInterpolationVec3");
}

} // namespace NeoN::bindings
