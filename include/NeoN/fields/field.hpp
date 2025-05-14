// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors
#pragma once

#include <Kokkos_Core.hpp>

#include <iostream>

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/vector/vector.hpp"

#include "NeoN/fields/boundaryData.hpp"

#include <vector>

namespace NeoN
{

/**
 * @class Field
 * @brief Represents the domain fields for a computational domain.
 *
 * The Field class stores the internal fields and boundary information for
 * a computational domain. It provides access to the computed values, reference
 * values, value fractions, reference gradients, boundary types, offsets, and
 * the number of boundaries and boundary faces.
 *
 * @tparam ValueType The type of the underlying field values
 */
template<typename ValueType>
class Field
{
public:

    Field(const Executor& exec) : exec_(exec), internalVector_(exec, 0), boundaryData_(exec, 0, 0)
    {}

    Field(const Executor& exec, localIdx nCells, const std::vector<localIdx>& offsets)
        : exec_(exec), internalVector_(exec, nCells), boundaryData_(exec, offsets)
    {}

    Field(
        const Executor& exec,
        const Vector<ValueType>& internalVector,
        const BoundaryData<ValueType>& boundaryData
    )
        : exec_(exec), internalVector_(exec, internalVector), boundaryData_(exec, boundaryData)
    {}

    Field(
        const Executor& exec, const Vector<ValueType>& internalVector, std::vector<localIdx> offsets
    )
        : exec_(exec), internalVector_(exec, internalVector), boundaryData_(exec, offsets)
    {}

    Field(const Executor& exec, localIdx internalSize, localIdx boundarySize)
        : exec_(exec), internalVector_(exec, internalSize), boundaryData_(exec, boundarySize)
    {}


    Field(const Field<ValueType>& rhs)
        : exec_(rhs.exec_), internalVector_(rhs.internalVector_), boundaryData_(rhs.boundaryData_)
    {}


    Field(Field<ValueType>&& rhs)
        : exec_(std::move(rhs.exec_)), internalVector_(std::move(rhs.internalVector_)),
          boundaryData_(std::move(rhs.boundaryData_))
    {}


    Field<ValueType>& operator=(const Field<ValueType>& rhs)
    {
        internalVector_ = rhs.internalVector_;
        boundaryData_ = rhs.boundaryData_;
        return *this;
    }


    Field<ValueType>& operator=(Field<ValueType>&& rhs)
    {
        internalVector_ = std::move(rhs.internalVector_);
        boundaryData_ = std::move(rhs.boundaryData_);
        return *this;
    }


    const Vector<ValueType>& internalVector() const { return internalVector_; }


    Vector<ValueType>& internalVector() { return internalVector_; }


    const BoundaryData<ValueType>& boundaryData() const { return boundaryData_; }


    BoundaryData<ValueType>& boundaryData() { return boundaryData_; }

    const Executor& exec() const { return exec_; }

private:

    Executor exec_; ///< The executor on which the field is stored
    Vector<ValueType> internalVector_;
    BoundaryData<ValueType> boundaryData_;
};


} // namespace NeoN
