// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/error.hpp"
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
        : patchID_(patchID), start_(mesh.boundaryMesh().offset()[static_cast<sizeType>(patchID_)]),
          end_(mesh.boundaryMesh().offset()[static_cast<sizeType>(patchID_) + 1])
    {}

    BoundaryPatchMixin(localIdx start, localIdx end, localIdx patchID)
        : patchID_(patchID), start_(start), end_(end)
    {}

    localIdx patchStart() const { return start_; };

    localIdx patchEnd() const { return start_; };

    localIdx patchSize() const { return end_ - start_; }

    localIdx patchID() const { return patchID_; }

    std::pair<localIdx, localIdx> range() const { return {start_, end_}; }

protected:

    /* @brief returns patchID, throws if the mesh has no such patch */
    static localIdx checkedPatchID(const UnstructuredMesh& mesh, localIdx patchID)
    {
        // a negative patchID becomes a large value and is rejected as well
        if (static_cast<sizeType>(patchID) >= mesh.nBoundaries())
        {
            NF_THROW(
                "invalid patch id " << patchID << ", the mesh has " << mesh.nBoundaries()
                                    << " patches"
            );
        }
        return patchID;
    }

    /* @brief returns dict["type"], throws if it is missing or not registered in Factory */
    template<typename Factory>
    static std::string boundaryType(const Dictionary& dict, localIdx patchID)
    {
        if (!dict.contains("type"))
        {
            NF_THROW("boundary condition on patch " << patchID << " has no 'type' entry");
        }
        const auto& type = dict.get<std::string>("type");
        if (!Factory::table().contains(type))
        {
            std::string valid;
            for (const auto& entry : Factory::table())
            {
                valid += " " + entry.first;
            }
            NF_THROW(
                "unknown boundary condition type '" << type << "' on patch " << patchID
                                                    << ", valid types are:" << valid
            );
        }
        return type;
    }

    localIdx patchID_; ///< The id of this patch
    localIdx start_;   ///< The start index of the patch in the boundaryVector
    localIdx end_;     ///< The end  index of the patch in the boundaryVector
};
}
