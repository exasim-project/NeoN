// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoFOAM/fields/segmentedField.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


class CellToFaceStencil
{
public:

    CellToFaceStencil(const UnstructuredMesh& mesh);

    SegmentedField<localIdx, localIdx> computeStencil() const;

private:

    const UnstructuredMesh& mesh_;
};

} // namespace NeoFOAM::finiteVolume::cellCentred
