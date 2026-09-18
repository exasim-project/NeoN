// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/error.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/view.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"

#include <variant>
#include <vector>


namespace NeoN
{

/**
 * @class Array
 * @brief An executor-aware contiguous container for structural and indexing data.
 *
 * @details
 * Array is intended for data that is primarily used as a generic collection,
 * particularly for structural, indexing, or other non-arithmetic data.
 *
 * Unlike Vector, Array does not primarily model numerical quantities or
 * provide vector-style arithmetic semantics. Use Array when the primary
 * purpose of the data is storage and indexed access rather than numerical operations.
 *
 * @tparam ValueType The type of the elements stored in the array.
 *
 * @ingroup Arrays
 */
template<typename ValueType>
class Array
{

public:

    using ArrayValueType = ValueType;

    /**
     * @brief Creates an uninitialized Array with the given size on an executor.
     *
     * @param exec Executor on which the array data is allocated.
     * @param size Number of elements in the array.
     */
    Array(const Executor& exec, localIdx size) : size_(size), data_(nullptr), exec_(exec)
    {
        void* ptr = nullptr;
        std::visit(
            [&ptr, size](const auto& concreteExec)
            { ptr = concreteExec.template alloc<ValueType>(static_cast<size_t>(size)); },
            exec_
        );
        data_ = static_cast<ValueType*>(ptr);
    }

    /**
     * @brief Creates an Array from existing data on the given executor.
     *
     * @param exec Executor on which the array data is allocated.
     * @param in Pointer to the source data.
     * @param size Number of elements to copy.
     * @param hostExec Executor on which the source data is located.
     */
    Array(
        const Executor& exec,
        const ValueType* in,
        localIdx size,
        Executor hostExec = SerialExecutor()
    )
        : size_(size), data_(nullptr), exec_(exec)
    {
        void* ptr = nullptr;
        std::visit(
            [&ptr, size](const auto& concreteExec)
            { ptr = concreteExec.template alloc<ValueType>(static_cast<size_t>(size)); },
            exec_
        );
        data_ = static_cast<ValueType*>(ptr);
        std::visit(detail::deepCopyVisitor<ValueType>(size_, in, data_), hostExec, exec_);
    }


    /**
     * @brief Creates an Array with the given size and initializes all elements to a value.
     *
     * @param exec Executor on which the array data is allocated.
     * @param size Number of elements in the array.
     * @param value Value used to initialize all elements.
     */
    Array(const Executor& exec, localIdx size, ValueType value)
        : size_(size), data_(nullptr), exec_(exec)
    {
        void* ptr = nullptr;
        std::visit(
            [&ptr, size](const auto& execu)
            { ptr = execu.template alloc<ValueType>(static_cast<size_t>(size)); },
            exec_
        );
        data_ = static_cast<ValueType*>(ptr);
        NeoN::fill(*this, value);
    }

    /**
     * @brief Create an Array from a std::vector on a given executor
     * @param exec  Executor on which the array data is allocated
     * @param in std::vector containing the values to copy into the array
     */
    Array(const Executor& exec, std::vector<ValueType> in)
        : Array(exec, in.data(), static_cast<localIdx>(in.size()))
    {}


    /**
     * @brief Creates a copy of an Array on the specified executor.
     *
     * @param exec Executor on which the new array is allocated.
     * @param in Array to copy.
     */
    Array(const Executor& exec, const Array<ValueType>& in)
        : Array(exec, in.data(), in.size(), in.exec())
    {}

    /**
     * @brief Creates a copy of another Array on the same executor.
     *
     * @param rhs Array to copy from.
     */
    Array(const Array<ValueType>& rhs) : Array(rhs.exec(), rhs.data(), rhs.size(), rhs.exec()) {}


    /**
     * @brief Move constructor.
     *
     * @param rhs Array whose resources are transferred to the new Array.
     */
    Array(Array<ValueType>&& rhs) noexcept : size_(rhs.size_), data_(rhs.data_), exec_(rhs.exec_)
    {
        rhs.data_ = nullptr;
        rhs.size_ = 0;
    };

    /**
     * @brief Destroy the Array object.
     */
    ~Array()
    {
        std::visit([this](const auto& exec) { exec.free(data_); }, exec_);
        data_ = nullptr;
    }

    /**
     * @brief Applies a function to each element of the Array.
     *
     * @param f Function or functor applied to the array elements.
     *
     * @note The function should be compatible with execution by Kokkos. Ideally, it should be
     * Kokkos lambda.
     */
    template<typename func>
    void apply(func f)
    {
        map(*this, f);
    }

    /**
     * @brief Creates a copy of the Array on a specified executor.
     *
     * @param dstExec Executor on which the copied array is allocated.
     * @return A copy of this Array on @p dstExec.
     */
    [[nodiscard]] Array<ValueType> copyToExecutor(Executor dstExec) const
    {
        if (dstExec == exec_) return *this;

        Array<ValueType> result(dstExec, size_);
        std::visit(detail::deepCopyVisitor(size_, data_, result.data()), exec_, dstExec);

        return result;
    }

    /**
     * @brief Creates a host-resident copy of the Array.
     *
     * @return A copy of this Array allocated on the serial executor.
     */
    [[nodiscard]] Array<ValueType> copyToHost() const { return copyToExecutor(SerialExecutor()); }

    /**
     * @brief Copies the Array data to a host-resident Array.
     *
     * @param result Destination Array. Its size must match the source Array.
     *
     * @warning An error is raised if @p result does not have the same size as
     * this Array.
     */
    void copyToHost(Array<ValueType>& result)
    {
        NF_DEBUG_ASSERT(
            result.size() == size_, "Parsed Array size not the same as current field size"
        );
        result = copyToExecutor(SerialExecutor());
    }

    // ensures no return of device address on host --> invalid memory access
    Array& operator[](const localIdx i) = delete;

    // ensures no return of device address on host --> invalid memory access
    const Array& operator[](const localIdx i) const = delete;

    /**
     * @brief Assigns a value to all elements of the Array.
     *
     * @param rhs Value assigned to each element.
     */
    void operator=(const ValueType& rhs)
    {
        NF_ERROR_EXIT("Not implemented");
        fill(*this, rhs);
    }

    /**
     * @brief Copies the contents of another Array.
     *
     * @param rhs Array to copy from.
     *
     * @warning The source and destination Arrays must use the same executor.
     * The destination is resized if necessary.
     */
    void operator=(const Array<ValueType>& rhs)
    {
        NF_ASSERT(exec_ == rhs.exec_, "Executors are not the same");
        if (this->size() != rhs.size())
        {
            this->resize(rhs.size());
        }
        setContainer(*this, rhs.view());
    }

    /**
     * @brief Resizes the Array.
     *
     * @param size New number of elements.
     */
    void resize(const localIdx size)
    {
        void* ptr = nullptr;
        if (!empty())
        {
            std::visit(
                [this, &ptr, size](const auto& exec)
                { ptr = exec.template realloc<ValueType>(this->data_, static_cast<size_t>(size)); },
                exec_
            );
        }
        else
        {
            std::visit(
                [&ptr, size](const auto& exec)
                { ptr = exec.template alloc<ValueType>(static_cast<size_t>(size)); },
                exec_
            );
        }
        data_ = static_cast<ValueType*>(ptr);
        size_ = size;
    }

    /**
     * @brief Returns a pointer to the underlying array data.
     *
     * @return Pointer to the first element.
     */
    [[nodiscard]] inline ValueType* data() { return data_; }

    /**
     * @brief Returns a pointer to the underlying array data.
     *
     * @return Pointer to the first element.
     */
    [[nodiscard]] inline const ValueType* data() const { return data_; }

    /**
     * @brief Returns the executor associated with the Array.
     *
     * @return Reference to the Array's executor.
     */
    [[nodiscard]] inline const Executor& exec() const { return exec_; }

    /**
     * @brief Returns the number of elements in the Array.
     *
     * @return Number of elements.
     */
    [[nodiscard]] inline localIdx size() const { return size_; }

    /**
     * @brief Returns the number of elements in the Array.
     *
     * @return Number of elements.
     */
    [[nodiscard]] inline label ssize() const { return static_cast<label>(size_); }

    /**
     * @brief Checks whether the Array contains no elements.
     *
     * @return True if the Array is empty, otherwise false.
     */
    [[nodiscard]] inline bool empty() const { return size() == 0; }

    // return of a temporary --> invalid memory access
    View<ValueType> view() && = delete;

    // return of a temporary --> invalid memory access
    View<const ValueType> view() const&& = delete;

    /**
     * @brief Returns a view of the Array data.
     *
     * @return Non-owning view of the Array data.
     */
    [[nodiscard]] inline View<ValueType> view() &
    {
        return View<ValueType>(data_, static_cast<size_t>(size_));
    }

    /**
     * @brief Returns a view of the Array data.
     *
     * @return Non-owning view of the Array data.
     */
    [[nodiscard]] inline View<const ValueType> view() const&
    {
        return View<const ValueType>(data_, static_cast<size_t>(size_));
    }

    // return of a temporary --> invalid memory access
    [[nodiscard]] View<ValueType> view(std::pair<localIdx, localIdx> range) && = delete;

    // return of a temporary --> invalid memory access
    [[nodiscard]] View<const ValueType> view(std::pair<localIdx, localIdx> range) const&& = delete;

    /**
     * @brief Returns a view of a range of the Array.
     *
     * @param range Half-open range of elements to include in the view.
     * @return Non-owning view of the specified range.
     */
    [[nodiscard]] inline View<ValueType> view(std::pair<localIdx, localIdx> range) &
    {
        return View<ValueType>(
            data_ + range.first, static_cast<size_t>(range.second - range.first)
        );
    }

    /**
     * @brief Returns a view of a range of the Array.
     *
     * @param range Half-open range of elements to include in the view.
     * @return Non-owning view of the specified range.
     */
    [[nodiscard]] inline View<const ValueType> view(std::pair<localIdx, localIdx> range) const&
    {
        return View<const ValueType>(
            data_ + range.first, static_cast<size_t>(range.second - range.first)
        );
    }

    /**
     * @brief Returns the index range of the Array.
     *
     * @return The half-open index range [0, size()).
     */
    [[nodiscard]] inline std::pair<localIdx, localIdx> range() const { return {0, size()}; }

private:

    localIdx size_ {0};         //!< Number of elements in the Array.
    ValueType* data_ {nullptr}; //!< Pointer to the underlying Array data.
    const Executor exec_;       //!< Executor associated with the Array.

    /**
     * @brief Checks if two Arrays have the same size and the same executor.
     * @param rhs Array to compare with.
     */
    void validateOtherArray(const Array<ValueType>& rhs) const
    {
        NF_DEBUG_ASSERT(size() == rhs.size(), "Arrays are not the same size.");
        NF_DEBUG_ASSERT(exec() == rhs.exec(), "Executors are not the same.");
    }
};

} // namespace NeoN
