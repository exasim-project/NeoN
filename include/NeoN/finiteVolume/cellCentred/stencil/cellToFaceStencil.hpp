// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/segmentedVector.hpp"

namespace NeoN::finiteVolume::cellCentred
{


class CellToFaceStencil
{
public:

    CellToFaceStencil(const UnstructuredMesh& mesh);

    /** @brief computes the internal mesh stencil
     *
     * Ie for each cell the faceIdxs of the internal faceIdxs are stored in order
     */
    SegmentedVector<localIdx, localIdx> computeInternalStencil() const;

private:

    const UnstructuredMesh& mesh_;
};

} // namespace NeoN::finiteVolume::cellCentred
