// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::turbulenceModels::DES
{

using VolScalarField = NeoN::finiteVolume::cellCentred::VolumeField<scalar>;

class maxDeltaxyz
{
public:

    explicit maxDeltaxyz(const UnstructuredMesh& mesh);

    void update();

    const VolScalarField& delta() const;

private:

    const UnstructuredMesh& mesh_;
    VolScalarField delta_;
};

} // namespace NeoN::turbulenceModels::DES
