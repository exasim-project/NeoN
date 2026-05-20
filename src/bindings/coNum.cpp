// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <stdexcept>
#include <string>

#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/auxiliary/coNum.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerCoNum(nb::module_& m)
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;

    m.def(
        "compute_co_num",
        [](const NeoN::UnstructuredMesh& mesh,
           const NeoN::Vector<NeoN::scalar>& faceFlux,
           NeoN::scalar dt) -> std::pair<NeoN::scalar, NeoN::scalar>
        {
            const auto nInt = mesh.nInternalFaces();
            const auto nBnd = mesh.nBoundaryFaces();
            const auto expected = nInt + nBnd;
            if (faceFlux.size() != expected)
            {
                throw std::runtime_error(
                    "face_flux size does not match mesh face count (" + std::to_string(expected)
                    + ")"
                );
            }

            auto bcs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);
            fvcc::SurfaceField<NeoN::scalar> flux(faceFlux.exec(), "faceFlux", mesh, bcs);

            // Split flat vector into internalVector (first nInt) and boundaryData (last nBnd)
            auto flatV = faceFlux.view();
            auto intV = flux.internalVector().view();
            auto bndV = flux.boundaryData().value().view();
            NeoN::parallelFor(
                faceFlux.exec(),
                {NeoN::localIdx(0), nInt},
                NEON_LAMBDA(const NeoN::localIdx i) { intV[i] = flatV[i]; },
                "coNum::splitInternal"
            );
            NeoN::parallelFor(
                faceFlux.exec(),
                {NeoN::localIdx(0), nBnd},
                NEON_LAMBDA(const NeoN::localIdx bfi) { bndV[bfi] = flatV[nInt + bfi]; },
                "coNum::splitBoundary"
            );

            return fvcc::computeCoNum(flux, dt);
        },
        "mesh"_a,
        "face_flux"_a,
        "dt"_a,
        "Compute the maximum Courant number from a face-flux vector."
    );
}

} // namespace NeoN::bindings
