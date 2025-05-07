// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors
#pragma once

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/**
 * @brief A base class for implementing derived boundary conditions
 *
 * This class holds the information where a given boundary starts
 * and ends in the consecutive boundaryVectors class
 *
 *
 * @tparam ValueType The data type of the field.
 * @note the class template parameters is currently needed since the
 * correctBoundaryConditions functions which takes templated arguments
 * is virtual.
 */
class BoundaryPatchMixin
{

public:

    BoundaryPatchMixin() {}

    virtual ~BoundaryPatchMixin() = default;

    BoundaryPatchMixin(const UnstructuredMesh& mesh, localIdx patchID)
        : patchID_(patchID), start_(mesh.boundaryMesh().offset()[static_cast<size_t>(patchID_)]),
          end_(mesh.boundaryMesh().offset()[static_cast<size_t>(patchID_) + 1])
    {}

    BoundaryPatchMixin(localIdx start, localIdx end, localIdx patchID)
        : patchID_(patchID), start_(start), end_(end)
    {}

    localIdx patchStart() const { return start_; };

    localIdx patchEnd() const { return start_; };

    localIdx patchSize() const { return end_ - start_; }

    localIdx patchID() const { return patchID_; }

    std::pair<localIdx, localIdx> range() { return {start_, end_}; }

protected:

    localIdx patchID_; ///< The id of this patch
    localIdx start_;   ///< The start index of the patch in the boundaryVector
    localIdx end_;     ///< The end  index of the patch in the boundaryVector
};
}
