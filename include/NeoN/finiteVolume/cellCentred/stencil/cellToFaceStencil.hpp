// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors

#pragma once

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/segmentedVector.hpp"

namespace NeoN::finiteVolume::cellCentred
{


class CellToFaceStencil
{
public:

    CellToFaceStencil(const UnstructuredMesh& mesh);

    SegmentedVector<localIdx, localIdx> computeStencil() const;

private:

    const UnstructuredMesh& mesh_;
};

} // namespace NeoN::finiteVolume::cellCentred
