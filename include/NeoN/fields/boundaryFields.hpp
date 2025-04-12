// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors
#pragma once

#include <Kokkos_Core.hpp>

#include <iostream>

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/fields/field.hpp"

namespace NeoN
{


/**
 * @class BoundaryFields
 * @brief Represents the boundary fields for a computational domain.
 *
 * The BoundaryFields class stores the boundary conditions and related
 * information for a computational domain. It provides access to the computed
 * values, reference values, value fractions, reference gradients, boundary
 * types, offsets, and the number of boundaries and boundary faces.
 *
 * @tparam ValueType The type of the underlying field values
 */
template<typename T>
class BoundaryFields
{

public:

    /**
     * @brief Copy constructor.
     * @param rhs The boundaryFields object to be copied.
     */
    BoundaryFields(const BoundaryFields<T>& rhs)
        : exec_(rhs.exec_), value_(rhs.value_), refValue_(rhs.refValue_),
          valueFraction_(rhs.valueFraction_), refGrad_(rhs.refGrad_),
          boundaryTypes_(rhs.boundaryTypes_), offset_(rhs.offset_), nBoundaries_(rhs.nBoundaries_),
          nBoundaryFaces_(rhs.nBoundaryFaces_)
    {}


    /**
     * @brief Copy constructor.
     * @param rhs The boundaryFields object to be copied.
     */
    BoundaryFields(const Executor& exec, const BoundaryFields<T>& rhs)
        : exec_(rhs.exec_), value_(exec, rhs.value_), refValue_(exec, rhs.refValue_),
          valueFraction_(exec, rhs.valueFraction_), refGrad_(exec, rhs.refGrad_),
          boundaryTypes_(exec, rhs.boundaryTypes_), offset_(exec, rhs.offset_),
          nBoundaries_(rhs.nBoundaries_), nBoundaryFaces_(rhs.nBoundaryFaces_)
    {}


    BoundaryFields(const Executor& exec, const std::vector<localIdx>& offsets)
        : exec_(exec), value_(exec, offsets.back()), refValue_(exec, offsets.back()),
          valueFraction_(exec, offsets.back()), refGrad_(exec, offsets.back()),
          boundaryTypes_(exec, offsets.size() - 1), offset_(exec, offsets),
          nBoundaries_(offsets.size() - 1), nBoundaryFaces_(offsets.back())
    {}


    /** @copydoc BoundaryFields::value()*/
    const NeoN::Field<T>& value() const { return value_; }

    /**
     * @brief Get the view storing the computed values from the boundary
     * condition.
     * @return The view storing the computed values.
     */
    NeoN::Field<T>& value() { return value_; }

    /** @copydoc BoundaryFields::refValue()*/
    const NeoN::Field<T>& refValue() const { return refValue_; }

    /**
     * @brief Get the view storing the Dirichlet boundary values.
     * @return The view storing the Dirichlet boundary values.
     */
    NeoN::Field<T>& refValue() { return refValue_; }

    /** @copydoc BoundaryFields::valueFraction()*/
    const NeoN::Field<scalar>& valueFraction() const { return valueFraction_; }

    /**
     * @brief Get the view storing the fraction of the boundary value.
     * @return The view storing the fraction of the boundary value.
     */
    NeoN::Field<scalar>& valueFraction() { return valueFraction_; }

    /** @copydoc BoundaryFields::refGrad()*/
    const NeoN::Field<T>& refGrad() const { return refGrad_; }

    /**
     * @brief Get the view storing the Neumann boundary values.
     * @return The view storing the Neumann boundary values.
     */
    NeoN::Field<T>& refGrad() { return refGrad_; }

    /**
     * @brief Get the view storing the boundary types.
     * @return The view storing the boundary types.
     */
    const NeoN::Field<int>& boundaryTypes() const { return boundaryTypes_; }

    /**
     * @brief Get the view storing the offsets of each boundary.
     * @return The view storing the offsets of each boundary.
     */
    const NeoN::Field<localIdx>& offset() const { return offset_; }

    /**
     * @brief Get the number of boundaries.
     * @return The number of boundaries.
     */
    size_t nBoundaries() const { return nBoundaries_; }

    /**
     * @brief Get the number of boundary faces.
     * @return The number of boundary faces.
     */
    size_t nBoundaryFaces() const { return nBoundaryFaces_; }

    const Executor& exec() { return exec_; }

    /**
     * @brief Get the range for a given patchId
     * @return The number of boundary faces.
     */
    std::pair<localIdx, localIdx> range(localIdx patchId) const
    {
        return {offset_.data()[patchId], offset_.data()[patchId + 1]};
    }

private:

    Executor exec_;                     ///< The executor on which the field is stored
    NeoN::Field<T> value_;              ///< The Field storing the computed values from the
                                        ///< boundary condition.
    NeoN::Field<T> refValue_;           ///< The Field storing the Dirichlet boundary values.
    NeoN::Field<scalar> valueFraction_; ///< The Field storing the fraction of
                                        ///< the boundary value.
    NeoN::Field<T> refGrad_;            ///< The Field storing the Neumann boundary values.
    NeoN::Field<int> boundaryTypes_;    ///< The Field storing the boundary types.
    NeoN::Field<localIdx> offset_;      ///< The Field storing the offsets of each boundary.
    size_t nBoundaries_;                ///< The number of boundaries.
    size_t nBoundaryFaces_;             ///< The number of boundary faces.
};

}
