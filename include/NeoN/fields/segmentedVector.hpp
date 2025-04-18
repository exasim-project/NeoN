// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors
#pragma once

#include "NeoN/core/view.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/vector.hpp"

namespace NeoN
{

/**
 * @brief Compute segment offsets from an input field corresponding to lengths by computing a prefix
 * sum.
 *
 * The offsets are computed by a prefix sum of the input values. So, with given
 * input of {1, 2, 3, 4, 5} the offsets are {0, 1, 3, 6, 10, 15}.
 * Note that the length of offSpan must be  length of intervals + 1
 * and are all elements of offSpan are required to be zero
 *
 * @param[in] in The values to compute the offsets from.
 * @param[in,out] offsets The field to store the resulting offsets in.
 */
template<typename IndexType>
IndexType segmentsFromIntervals(const Vector<IndexType>& intervals, Vector<IndexType>& offsets)
{
    IndexType finalValue = 0;
    const auto inSpan = intervals.view();
    // skip the first element of the offsets
    // assumed to be zero
    auto offsSpan = offsets.view().subspan(1);
    // NOTE avoid compiler warning by static_casting to localIdx since offsSpan
    // is a std::span
    NF_ASSERT_EQUAL(inSpan.size(), offsSpan.size());
    NeoN::parallelScan(
        intervals.exec(),
        {0, offsSpan.size()},
        KOKKOS_LAMBDA(const localIdx i, IndexType& update, const bool final) {
            update += inSpan[i];
            if (final)
            {
                // offsSpan is a std::span, thus [] takes unsigned idx
                offsSpan[i] = update;
            }
        },
        finalValue
    );
    return finalValue;
}

/**
 * @brief A class representing a segment of indices.
 *
 * @tparam IndexType The type of the indices.
 */
template<typename ValueType, typename IndexType = NeoN::localIdx>
class SegmentedVectorView
{
public:

    /**
     * @brief A span with the values.
     */
    View<ValueType> values;

    /**
     * @brief A span of indices representing the segments.
     */
    View<IndexType> segments;

    /**
     * @brief Get the bounds of a segment.
     *
     * @param segI The index of the segment.
     * @return A pair of indices representing the start and end of the segment.
     */
    KOKKOS_INLINE_FUNCTION
    Kokkos::pair<IndexType, IndexType> bounds(localIdx segI) const
    {
        return Kokkos::pair<IndexType, IndexType> {segments[segI], segments[segI + 1]};
    }

    /**
     * @brief Get the range, ie. [start,end), of a segment.
     *
     * @param segI The index of the segment.
     * @return A pair of indices representing the start and length of the segment.
     */
    KOKKOS_INLINE_FUNCTION
    Kokkos::pair<IndexType, IndexType> range(localIdx segI) const
    {
        return Kokkos::pair<IndexType, IndexType> {
            segments[segI], segments[segI + 1] - segments[segI]
        };
    }

    /**
     * @brief Get a subspan of values corresponding to a segment.
     *
     * @tparam ValueType The type of the values.
     * @param segI The index of the segment.
     * @return A subspan of values corresponding to the segment.
     */
    KOKKOS_INLINE_FUNCTION View<ValueType> span(localIdx segI) const
    {
        auto [start, length] = range(segI);
        return values.subspan(start, length);
    }

    /**
     * @brief Access an element of the segments.
     *
     * @param i The index of the element.
     * @return The value of the element at the specified index.
     */
    KOKKOS_INLINE_FUNCTION
    IndexType operator[](localIdx i) const { return segments[i]; }
};

/**
 * @class SegmentedVector
 * @brief Data structure that stores a segmented fields or a vector of vectors
 *
 * @ingroup Vectors
 */
template<typename ValueType, typename IndexType>
class SegmentedVector
{
public:


    /**
     * @brief Create a segmented field with a given size and number of segments.
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param numSegments  number of segments
     */
    SegmentedVector(const Executor& exec, localIdx size, localIdx numSegments)
        : values_(exec, size), segments_(exec, numSegments + 1)
    {}

    /*
     * @brief Create a segmented field from intervals.
     * @param intervals The intervals to create the segmented field from.
     * @note The intervals are the lengths of each segment
     */
    SegmentedVector(const Vector<IndexType>& intervals)
        : values_(intervals.exec(), 0),
          segments_(intervals.exec(), intervals.size() + 1, IndexType(0))
    {
        IndexType valueSize = segmentsFromIntervals(intervals, segments_);
        values_ = Vector<ValueType>(intervals.exec(), valueSize);
    }


    /**
     * @brief Constructor to create a segmentedVector from values and the segments.
     * @param values The values of the segmented field.
     * @param segments The segments of the segmented field.
     */
    SegmentedVector(const Vector<ValueType>& values, const Vector<IndexType>& segments)
        : values_(values), segments_(segments)
    {
        NF_ASSERT(values.exec() == segments.exec(), "Executors are not the same.");
    }


    /**
     * @brief Get the executor associated with the segmented field.
     * @return Reference to the executor.
     */
    const Executor& exec() const { return values_.exec(); }

    /**
     * @brief Get the size of the segmented field.
     * @return The size of the segmented field.
     */
    localIdx size() const { return values_.size(); }

    /**
     * @brief Get the number of segments in the segmented field.
     * @return The number of segments.
     */
    localIdx numSegments() const { return segments_.size() - 1; }


    /**
     * @brief get a view of the segmented field
     * @return View of the fields
     */
    [[nodiscard]] SegmentedVectorView<ValueType, IndexType> view() &
    {
        return SegmentedVectorView<ValueType, IndexType> {values_.view(), segments_.view()};
    }

    // ensures no return a span of a temporary object --> invalid memory access
    [[nodiscard]] SegmentedVectorView<ValueType, IndexType> view() && = delete;

    /**
     * @brief get the combined value and range spans of the segmented field
     * @return Combined value and range spans of the fields
     */
    [[nodiscard]] std::pair<View<ValueType>, View<IndexType>> spans() &
    {
        return {values_.view(), segments_.view()};
    }

    // ensures not to return a span of a temporary object --> invalid memory access
    [[nodiscard]] std::pair<View<ValueType>, View<IndexType>> spans() && = delete;

    const Vector<ValueType>& values() const { return values_; }

    const Vector<IndexType>& segments() const { return segments_; }

private:

    Vector<ValueType> values_;
    Vector<IndexType> segments_; //!< stores the [start, end) of segment i at index i, i+1
};

} // namespace NeoN
