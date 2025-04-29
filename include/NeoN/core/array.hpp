
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/core/error.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/view.hpp"
#include "NeoN/core/vector/vectorFreeFunctions.hpp"

#include <variant>
#include <vector>


namespace NeoN
{

namespace detail
{

/**
 * @brief A helper function to simplify the common pattern of copying between and to executor.
 * @param size The number of elements to copy.
 * @param srcPtr Pointer to the original block of memory.
 * @param dstPtr Pointer to the target block of memory.
 * @tparam ValueType The type of the underlying elements.
 * @returns A function that takes a source and an destination executor
 */
template<typename ValueType>
auto deepCopyVisitor(localIdx ssize, const ValueType* srcPtr, ValueType* dstPtr);

}

/**
 * @class Vector
 * @brief A class to contain the data and executors for a field and define some basic operations.
 *
 * @ingroup Vectors
 */
template<typename ValueType>
class Array
{

public:

    using ArrayValueType = ValueType;

    /**
     * @brief Create an uninitialized Vector with a given size on an executor
     * @param exec  Executor associated to the field
     * @param size  size of the field
     */
    Array(const Executor& exec, localIdx size) : size_(size), data_(nullptr), exec_(exec)
    {
        void* ptr = nullptr;
        std::visit(
            [&ptr, size](const auto& concreteExec)
            { ptr = concreteExec.alloc(static_cast<size_t>(size) * sizeof(ValueType)); },
            exec_
        );
        data_ = static_cast<ValueType*>(ptr);
    }

    /**
     * @brief Create a Vector with a given size from existing memory on an executor
     * @param exec  Executor associated to the field
     * @param in    Pointer to existing data
     * @param size  size of the field
     * @param hostExec Executor where the original data is located
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
            { ptr = concreteExec.alloc(static_cast<size_t>(size) * sizeof(ValueType)); },
            exec_
        );
        data_ = static_cast<ValueType*>(ptr);
        std::visit(detail::deepCopyVisitor<ValueType>(size_, in, data_), hostExec, exec_);
    }


    /**
     * @brief Create a Vector with a given size on an executor and uniform value
     * @param exec  Executor associated to the field
     * @param size  size of the field
     * @param value  the  default value
     */
    Array(const Executor& exec, localIdx size, ValueType value)
        : size_(size), data_(nullptr), exec_(exec)
    {
        // void* ptr = nullptr;
        // std::visit(
        //     [&ptr, size](const auto& execu)
        //     { ptr = execu.alloc(static_cast<size_t>(size) * sizeof(ValueType)); },
        //     exec_
        // );
        // data_ = static_cast<ValueType*>(ptr);
        // NeoN::fill(*this, value);
    }

    /**
     * @brief Create a Vector from a given vector of values on an executor
     * @param exec  Executor associated to the field
     * @param in a vector of elements to copy over
     */
    Array(const Executor& exec, std::vector<ValueType> in)
        : Array(exec, in.data(), static_cast<localIdx>(in.size()))
    {}


    /**
     * @brief Create a Vector as a copy of a Vector on a specified executor
     * @param exec  Executor associated to the field
     * @param in a Vector of elements to copy over
     */
    Array(const Executor& exec, const Array<ValueType>& in)
        : Array(exec, in.data(), in.size(), in.exec())
    {}

    /**
     * @brief Copy constructor, creates a new field with the same size and data as the parsed field.
     * @param rhs The field to copy from.
     */
    Array(const Array<ValueType>& rhs) : Array(rhs.exec(), rhs.data(), rhs.size(), rhs.exec()) {}


    /**
     * @brief Move constructor, moves the data from the parsed field to the new field.
     * @param rhs The field to move from.
     */
    Array(Array<ValueType>&& rhs) noexcept : size_(rhs.size_), data_(rhs.data_), exec_(rhs.exec_)
    {
        rhs.data_ = nullptr;
        rhs.size_ = 0;
    };

    /**
     * @brief Destroy the Vector object.
     */
    ~Array()
    {
        std::visit([this](const auto& exec) { exec.free(data_); }, exec_);
        data_ = nullptr;
    }

    /**
     * @brief applies a functor, transformation, to the field
     * @param f The functor to map over the field.
     * @note Ideally the f should be a KOKKOS_LAMBA
     */
    template<typename func>
    void apply(func f)
    {
        // FIXME:
        // auto [start, end] = range;
        // if (end == 0)
        // {
        //     end = this->size();
        // }
        // auto viewA = this->view();
        // parallelFor(
        //     this->exec(), {start, end}, KOKKOS_LAMBDA(const localIdx i) { viewA[i] = f(i); }
        // );
    }

    /**
     * @brief Copies the data to a new field on a specific executor.
     * @param dstExec The executor on which the data should be copied.
     * @returns A copy of the field on the host.
     */
    [[nodiscard]] Array<ValueType> copyToExecutor(Executor dstExec) const
    {
        if (dstExec == exec_) return *this;

        Array<ValueType> result(dstExec, size_);
        std::visit(detail::deepCopyVisitor(size_, data_, result.data()), exec_, dstExec);

        return result;
    }

    /**
     * @brief Returns a copy of the field back to the host.
     * @returns A copy of the field on the host.
     */
    [[nodiscard]] Array<ValueType> copyToHost() const { return copyToExecutor(SerialExecutor()); }

    /**
     * @brief Copies the data (from anywhere) to a parsed host field.
     * @param result The field into which the data must be copied. Must be
     * sized.
     *
     * @warning exits if the size of the result field is not the same as the
     * source field.
     */
    void copyToHost(Array<ValueType>& result)
    {
        NF_DEBUG_ASSERT(
            result.size() == size_, "Parsed Vector size not the same as current field size"
        );
        result = copyToExecutor(SerialExecutor());
    }

    // ensures no return of device address on host --> invalid memory access
    Array& operator[](const localIdx i) = delete;

    // ensures no return of device address on host --> invalid memory access
    const Array& operator[](const localIdx i) const = delete;

    /**
     * @brief Assignment operator, Sets the field values to that of the passed value.
     * @param rhs The value to set the field to.
     */
    void operator=(const ValueType& rhs)
    {
        // FIXME:
        NF_ERROR_EXIT("Not implemented");
        // (*this, rhs);
    }

    /**
     * @brief Assignment operator, Sets the field values to that of the parsed field.
     * @param rhs The field to copy from.
     *
     * @warning This field will be sized to the size of the parsed field.
     */
    void operator=(const Array<ValueType>& rhs)
    {
        // FIXME:
        NF_ASSERT(exec_ == rhs.exec_, "Executors are not the same");
        // if (this->size() != rhs.size())
        // {
        //     this->resize(rhs.size());
        // }
        // setVector(*this, rhs.view());
    }

    /**
     * @brief Resizes the field to a new size.
     * @param size The new size to set the field to.
     */
    void resize(const localIdx size)
    {
        void* ptr = nullptr;
        if (!empty())
        {
            std::visit(
                [this, &ptr, size](const auto& exec)
                { ptr = exec.realloc(this->data_, static_cast<size_t>(size) * sizeof(ValueType)); },
                exec_
            );
        }
        else
        {
            std::visit(
                [&ptr, size](const auto& exec)
                { ptr = exec.alloc(static_cast<size_t>(size) * sizeof(ValueType)); },
                exec_
            );
        }
        data_ = static_cast<ValueType*>(ptr);
        size_ = size;
    }

    /**
     * @brief Direct access to the underlying field data
     * @return Pointer to the first cell data in the field.
     */
    [[nodiscard]] inline ValueType* data() { return data_; }

    /**
     * @brief Direct access to the underlying field data
     * @return Pointer to the first cell data in the field.
     */
    [[nodiscard]] inline const ValueType* data() const { return data_; }

    /**
     * @brief Gets the executor associated with the field.
     * @return Reference to the executor.
     */
    [[nodiscard]] inline const Executor& exec() const { return exec_; }

    /**
     * @brief Gets the size of the field.
     * @return The size of the field.
     */
    [[nodiscard]] inline localIdx size() const { return size_; }

    /**
     * @brief Gets the size of the field.
     * @return The size of the field.
     */
    [[nodiscard]] inline label ssize() const { return static_cast<label>(size_); }

    /**
     * @brief Checks if the field is empty.
     * @return True if the field is empty, false otherwise.
     */
    [[nodiscard]] inline bool empty() const { return size() == 0; }

    // return of a temporary --> invalid memory access
    View<ValueType> view() && = delete;

    // return of a temporary --> invalid memory access
    View<const ValueType> view() const&& = delete;

    /**
     * @brief Gets the field as a view.
     * @return View of the field.
     */
    [[nodiscard]] inline View<ValueType> view() &
    {
        return View<ValueType>(data_, static_cast<size_t>(size_));
    }

    /**
     * @brief Gets the field as a view.
     * @return View of the field.
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
     * @brief Gets a sub view of the field as a view.
     * @return View of the field.
     */
    [[nodiscard]] inline View<ValueType> view(std::pair<localIdx, localIdx> range) &
    {
        return View<ValueType>(
            data_ + range.first, static_cast<size_t>(range.second - range.first)
        );
    }

    /**
     * @brief Gets a sub view of the field as a view.
     * @return View of the field.
     */
    [[nodiscard]] inline View<const ValueType> view(std::pair<localIdx, localIdx> range) const&
    {
        return View<const ValueType>(
            data_ + range.first, static_cast<size_t>(range.second - range.first)
        );
    }

    /**
     * @brief Gets the range of the field.
     * @return The range of the field {0, size()}.
     */
    [[nodiscard]] inline std::pair<localIdx, localIdx> range() const { return {0, size()}; }

private:

    localIdx size_ {0};         //!< Size of the field.
    ValueType* data_ {nullptr}; //!< Pointer to the field data.
    const Executor exec_;       //!< Executor associated with the field. (CPU, GPU, openMP, etc.)

    /**
     * @brief Checks if two fields are the same size and have the same executor.
     * @param rhs The field to compare with.
     */
    void validateOtherArray(const Array<ValueType>& rhs) const
    {
        NF_DEBUG_ASSERT(size() == rhs.size(), "Arrays are not the same size.");
        NF_DEBUG_ASSERT(exec() == rhs.exec(), "Executors are not the same.");
    }
};

} // namespace NeoN
