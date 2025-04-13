// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors
#pragma once

#include <Kokkos_Core.hpp>


#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/fields/vector.hpp"

#include <vector>
#include <utility>

namespace NeoN
{


/**
 * @class BoundaryData
 * @brief Represents the boundary fields for a computational domain.
 *
 * The BoundaryData class stores the boundary conditions and related
 * information for a computational domain. It provides access to the computed
 * values, reference values, value fractions, reference gradients, boundary
 * types, offsets, and the number of boundaries and boundary faces.
 *
 * @tparam ValueType The type of the underlying field values
 */
template<typename T>
class BoundaryData
{

public:

    /**
     * @brief Copy constructor.
     * @param rhs The boundaryVectors object to be copied.
     */
    BoundaryData(const BoundaryData<T>& rhs)
        : exec_(rhs.exec_), value_(rhs.value_), refValue_(rhs.refValue_),
          valueFraction_(rhs.valueFraction_), refGrad_(rhs.refGrad_),
          boundaryTypes_(rhs.boundaryTypes_), offset_(rhs.offset_), nBoundaries_(rhs.nBoundaries_),
          nBoundaryFaces_(rhs.nBoundaryFaces_)
    {}


    /**
     * @brief Copy constructor.
     * @param rhs The boundaryVectors object to be copied.
     */
    BoundaryData(const Executor& exec, const BoundaryData<T>& rhs)
        : exec_(rhs.exec_), value_(exec, rhs.value_), refValue_(exec, rhs.refValue_),
          valueFraction_(exec, rhs.valueFraction_), refGrad_(exec, rhs.refGrad_),
          boundaryTypes_(exec, rhs.boundaryTypes_), offset_(exec, rhs.offset_),
          nBoundaries_(rhs.nBoundaries_), nBoundaryFaces_(rhs.nBoundaryFaces_)
    {}


    /**
     * @brief constructor with default initialized Vectors from sizes.
     * @param exec - The executor
     * @param nBoundaryFaces - The total number of boundary faces
     * @param nBoundaryType - The total number of boundary patches
     */
    BoundaryData(const Executor& exec, int nBoundaryFaces, int nBoundaryTypes)
        : exec_(exec), value_(exec, nBoundaryFaces), refValue_(exec, nBoundaryFaces),
          valueFraction_(exec, nBoundaryFaces), refGrad_(exec, nBoundaryFaces),
          boundaryTypes_(exec, nBoundaryTypes), offset_(exec, nBoundaryTypes + 1),
          nBoundaries_(nBoundaryTypes), nBoundaryFaces_(nBoundaryFaces)
    {}

    /**
     * @brief constructor from a given offsets vector
     * @warn all members except offsets are default constructed
     * @param exec - The executor
     * @param offsets - The total number of boundary faces
     */
    BoundaryData(const Executor& exec, const std::vector<localIdx>& offsets)
        : BoundaryData(exec, offsets.back(), offsets.size() - 1)
    {
        offset_ = Vector(exec, offsets);
    }


    /** @copydoc BoundaryData::value()*/
    const Vector<T>& value() const { return value_; }

    /**
     * @brief Get the view storing the computed values from the boundary
     * condition.
     * @return The view storing the computed values.
     */
    Vector<T>& value() { return value_; }

    /** @copydoc BoundaryData::refValue()*/
    const Vector<T>& refValue() const { return refValue_; }

    /**
     * @brief Get the view storing the Dirichlet boundary values.
     * @return The view storing the Dirichlet boundary values.
     */
    Vector<T>& refValue() { return refValue_; }

    /** @copydoc BoundaryData::valueFraction()*/
    const Vector<scalar>& valueFraction() const { return valueFraction_; }

    /**
     * @brief Get the view storing the fraction of the boundary value.
     * @return The view storing the fraction of the boundary value.
     */
    Vector<scalar>& valueFraction() { return valueFraction_; }

    /** @copydoc BoundaryData::refGrad()*/
    const Vector<T>& refGrad() const { return refGrad_; }

    /**
     * @brief Get the view storing the Neumann boundary values.
     * @return The view storing the Neumann boundary values.
     */
    Vector<T>& refGrad() { return refGrad_; }

    /**
     * @brief Get the view storing the boundary types.
     * @return The view storing the boundary types.
     */
    const Vector<int>& boundaryTypes() const { return boundaryTypes_; }

    /**
     * @brief Get the view storing the offsets of each boundary.
     * @return The view storing the offsets of each boundary.
     */
    const Vector<localIdx>& offset() const { return offset_; }

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

    Executor exec_;                ///< The executor on which the field is stored
    Vector<T> value_;              ///< The Vector storing the computed values from the
                                   ///< boundary condition.
    Vector<T> refValue_;           ///< The Vector storing the Dirichlet boundary values.
    Vector<scalar> valueFraction_; ///< The Vector storing the fraction of
                                   ///< the boundary value.
    Vector<T> refGrad_;            ///< The Vector storing the Neumann boundary values.
    Vector<int> boundaryTypes_;    ///< The Vector storing the boundary types.
    Vector<localIdx> offset_;      ///< The Vector storing the offsets of each boundary.
    size_t nBoundaries_;           ///< The number of boundaries.
    size_t nBoundaryFaces_;        ///< The number of boundary faces.
};

}
