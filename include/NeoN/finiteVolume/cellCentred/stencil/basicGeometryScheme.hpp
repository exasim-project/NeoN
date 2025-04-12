// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/stencil/geometryScheme.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/surfaceField.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

class BasicGeometryScheme : public GeometrySchemeFactory
{

public:

    BasicGeometryScheme(const UnstructuredMesh& mesh);

    void updateWeights(const Executor& exec, SurfaceField<scalar>& weights) override;

    void updateDeltaCoeffs(const Executor& exec, SurfaceField<scalar>& deltaCoeffs) override;

    void updateNonOrthDeltaCoeffs(const Executor& exec, SurfaceField<scalar>& nonOrthDeltaCoeffs)
        override;

    void updateNonOrthDeltaCoeffs(const Executor& exec, SurfaceField<Vector>& nonOrthDeltaCoeffs)
        override;


private:

    const UnstructuredMesh& mesh_;
};

} // namespace NeoFOAM
