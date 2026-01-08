// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/turbulenceModels/ReynoldsStress.hpp"

#include "NeoN/core/error.hpp"
#include "NeoN/dsl/coeff.hpp"

namespace NeoN::turbulenceModels
{

ReynoldsStress::ReynoldsStress(const Executor& exec, const UnstructuredMesh& mesh, Input input)
    : exec_(exec), mesh_(mesh), input_(std::move(input)), surfaceInterpolation_(exec, mesh, input_),
      laplacian_(exec, mesh, input_)
{}

VolVectorField
ReynoldsStress::divDevReff(const VolVectorField& velocity, const VolScalarField& nuEff) const
{
    NF_DEBUG_ASSERT(
        velocity.mesh().nCells() == nuEff.mesh().nCells(),
        "divDevReff requires matching mesh sizes."
    );

    auto nuEffFace = surfaceInterpolation_.interpolate(nuEff);
    return laplacian_.laplacian(nuEffFace, velocity, dsl::Coeff(-1.0));
}

} // namespace NeoN::turbulenceModels
