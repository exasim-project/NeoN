// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/core/primitives/label.hpp"
#include "NeoFOAM/fields/field.hpp"

namespace NeoFOAM
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
IndexType segmentsFromIntervals(const Field<IndexType>& intervals, Field<IndexType>& offsets)
{
    IndexType finalValue = 0;
    const auto inSpan = intervals.view();
    // skip the first element of the offsets
    // assumed to be zero
    auto offsSpan = offsets.view().subspan(1);
    NF_ASSERT_EQUAL(inSpan.size(), offsSpan.size());
    NeoFOAM::parallelScan(
        intervals.exec(),
        {0, offsSpan.size()},
        KOKKOS_LAMBDA(const std::size_t i, IndexType& update, const bool final) {
            update += inSpan[i];
            if (final)
            {
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
template<typename ValueType, typename IndexType = NeoFOAM::localIdx>
class SegmentedFieldView
{
public:

    /**
     * @brief A span with the values.
     */
    std::span<ValueType> values;

    /**
     * @brief A span of indices representing the segments.
     */
    std::span<IndexType> segments;

    /**
     * @brief Get the bounds of a segment.
     *
     * @param segI The index of the segment.
     * @return A pair of indices representing the start and end of the segment.
     */
    KOKKOS_INLINE_FUNCTION
    Kokkos::pair<IndexType, IndexType> bounds(std::size_t segI) const
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
    Kokkos::pair<IndexType, IndexType> range(std::size_t segI) const
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
    KOKKOS_INLINE_FUNCTION std::span<ValueType> span(std::size_t segI) const
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
    IndexType operator[](std::size_t i) const { return segments[i]; }
};

/**
 * @class SegmentedField
 * @brief Data structure that stores a segmented fields or a vector of vectors
 *
 * @ingroup Fields
 */
template<typename ValueType, typename IndexType>
class SegmentedField
{
public:


    /**
     * @brief Create a segmented field with a given size and number of segments.
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param numSegments  number of segments
     */
    SegmentedField(const Executor& exec, size_t size, size_t numSegments)
        : values_(exec, size), segments_(exec, numSegments + 1)
    {}

    /*
     * @brief Create a segmented field from intervals.
     * @param intervals The intervals to create the segmented field from.
     * @note The intervals are the lengths of each segment
     */
    SegmentedField(const Field<IndexType>& intervals)
        : values_(intervals.exec(), 0),
          segments_(intervals.exec(), intervals.size() + 1, IndexType(0))
    {
        IndexType valueSize = segmentsFromIntervals(intervals, segments_);
        values_ = Field<ValueType>(intervals.exec(), valueSize);
    }


    /**
     * @brief Constructor to create a segmentedField from values and the segments.
     * @param values The values of the segmented field.
     * @param segments The segments of the segmented field.
     */
    SegmentedField(const Field<ValueType>& values, const Field<IndexType>& segments)
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
    size_t size() const { return values_.size(); }

    /**
     * @brief Get the number of segments in the segmented field.
     * @return The number of segments.
     */
    size_t numSegments() const { return segments_.size() - 1; }


    /**
     * @brief get a view of the segmented field
     * @return View of the fields
     */
    [[nodiscard]] SegmentedFieldView<ValueType, IndexType> view() &
    {
        return SegmentedFieldView<ValueType, IndexType> {values_.view(), segments_.view()};
    }

    // ensures no return a span of a temporary object --> invalid memory access
    [[nodiscard]] SegmentedFieldView<ValueType, IndexType> view() && = delete;

    /**
     * @brief get the combined value and range spans of the segmented field
     * @return Combined value and range spans of the fields
     */
    [[nodiscard]] std::pair<std::span<ValueType>, std::span<IndexType>> spans() &
    {
        return {values_.view(), segments_.view()};
    }

    // ensures not to return a span of a temporary object --> invalid memory access
    [[nodiscard]] std::pair<std::span<ValueType>, std::span<IndexType>> spans() && = delete;

    const Field<ValueType>& values() const { return values_; }

    const Field<IndexType>& segments() const { return segments_; }

private:

    Field<ValueType> values_;
    Field<IndexType> segments_; //!< stores the [start, end) of segment i at index i, i+1
};

} // namespace NeoFOAM
