// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/tokenList.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerSurfaceInterpolation(nb::module_& m)
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;

    nb::class_<fvcc::SurfaceInterpolation<NeoN::scalar>>(
        m, "SurfaceInterpolationScalar", "Surface interpolation for scalar volume fields"
    )
        .def(
            "__init__",
            [](fvcc::SurfaceInterpolation<NeoN::scalar>& interp,
               const NeoN::SerialExecutor& exec,
               const NeoN::UnstructuredMesh& mesh,
               const std::string& scheme)
            {
                NeoN::TokenList tokens({scheme});
                new (&interp)
                    fvcc::SurfaceInterpolation<NeoN::scalar>(NeoN::Executor {exec}, mesh, tokens);
            },
            "exec"_a,
            "mesh"_a,
            "scheme"_a = "linear",
            "Create a scalar surface interpolation using a named scheme"
        )
        .def(
            "__init__",
            [](fvcc::SurfaceInterpolation<NeoN::scalar>& interp,
               const NeoN::CPUExecutor& exec,
               const NeoN::UnstructuredMesh& mesh,
               const std::string& scheme)
            {
                NeoN::TokenList tokens({scheme});
                new (&interp)
                    fvcc::SurfaceInterpolation<NeoN::scalar>(NeoN::Executor {exec}, mesh, tokens);
            },
            "exec"_a,
            "mesh"_a,
            "scheme"_a = "linear",
            "Create a scalar surface interpolation using a named scheme"
        )
        .def(
            "__init__",
            [](fvcc::SurfaceInterpolation<NeoN::scalar>& interp,
               const NeoN::GPUExecutor& exec,
               const NeoN::UnstructuredMesh& mesh,
               const std::string& scheme)
            {
                NeoN::TokenList tokens({scheme});
                new (&interp)
                    fvcc::SurfaceInterpolation<NeoN::scalar>(NeoN::Executor {exec}, mesh, tokens);
            },
            "exec"_a,
            "mesh"_a,
            "scheme"_a = "linear",
            "Create a scalar surface interpolation using a named scheme"
        )
        .def(
            "interpolate",
            [](const fvcc::SurfaceInterpolation<NeoN::scalar>& interp,
               const fvcc::VolumeField<NeoN::scalar>& src) { return interp.interpolate(src); },
            "src"_a,
            "Interpolate a scalar volume field to a surface field"
        )
        .def(
            "interpolate_into",
            [](const fvcc::SurfaceInterpolation<NeoN::scalar>& interp,
               const fvcc::VolumeField<NeoN::scalar>& src,
               fvcc::SurfaceField<NeoN::scalar>& dst) { interp.interpolate(src, dst); },
            "src"_a,
            "dst"_a,
            "Interpolate into an existing scalar surface field"
        )
        .def(
            "weight",
            [](const fvcc::SurfaceInterpolation<NeoN::scalar>& interp,
               const fvcc::VolumeField<NeoN::scalar>& src) { return interp.weight(src); },
            "src"_a,
            "Compute interpolation weights for a scalar volume field"
        );

    nb::class_<fvcc::SurfaceInterpolation<NeoN::Vec3>>(
        m, "SurfaceInterpolationVec3", "Surface interpolation for Vec3 volume fields"
    )
        .def(
            "__init__",
            [](fvcc::SurfaceInterpolation<NeoN::Vec3>& interp,
               const NeoN::SerialExecutor& exec,
               const NeoN::UnstructuredMesh& mesh,
               const std::string& scheme)
            {
                NeoN::TokenList tokens({scheme});
                new (&interp)
                    fvcc::SurfaceInterpolation<NeoN::Vec3>(NeoN::Executor {exec}, mesh, tokens);
            },
            "exec"_a,
            "mesh"_a,
            "scheme"_a = "linear"
        )
        .def(
            "__init__",
            [](fvcc::SurfaceInterpolation<NeoN::Vec3>& interp,
               const NeoN::CPUExecutor& exec,
               const NeoN::UnstructuredMesh& mesh,
               const std::string& scheme)
            {
                NeoN::TokenList tokens({scheme});
                new (&interp)
                    fvcc::SurfaceInterpolation<NeoN::Vec3>(NeoN::Executor {exec}, mesh, tokens);
            },
            "exec"_a,
            "mesh"_a,
            "scheme"_a = "linear"
        )
        .def(
            "__init__",
            [](fvcc::SurfaceInterpolation<NeoN::Vec3>& interp,
               const NeoN::GPUExecutor& exec,
               const NeoN::UnstructuredMesh& mesh,
               const std::string& scheme)
            {
                NeoN::TokenList tokens({scheme});
                new (&interp)
                    fvcc::SurfaceInterpolation<NeoN::Vec3>(NeoN::Executor {exec}, mesh, tokens);
            },
            "exec"_a,
            "mesh"_a,
            "scheme"_a = "linear"
        )
        .def(
            "interpolate",
            [](const fvcc::SurfaceInterpolation<NeoN::Vec3>& interp,
               const fvcc::VolumeField<NeoN::Vec3>& src) { return interp.interpolate(src); },
            "src"_a
        )
        .def(
            "interpolate_into",
            [](const fvcc::SurfaceInterpolation<NeoN::Vec3>& interp,
               const fvcc::VolumeField<NeoN::Vec3>& src,
               fvcc::SurfaceField<NeoN::Vec3>& dst) { interp.interpolate(src, dst); },
            "src"_a,
            "dst"_a
        )
        .def(
            "weight",
            [](const fvcc::SurfaceInterpolation<NeoN::Vec3>& interp,
               const fvcc::VolumeField<NeoN::Vec3>& src) { return interp.weight(src); },
            "src"_a
        );
}

} // namespace NeoN::bindings
