// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors
#pragma once

#include <Kokkos_Core.hpp>

#include <iostream>

#include "NeoN/core/error.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/view.hpp"
#include "NeoN/fields/fieldFreeFunctions.hpp"
#include "NeoN/fields/fieldTypeDefs.hpp"

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
auto deepCopyVisitor(localIdx size, const ValueType* srcPtr, ValueType* dstPtr)
{
    return [size, srcPtr, dstPtr](const auto& srcExec, const auto& dstExec)
    {
        Kokkos::deep_copy(
            dstExec.createKokkosView(dstPtr, size), srcExec.createKokkosView(srcPtr, size)
        );
    };
}
}

/**
 * @class Field
 * @brief A class to contain the data and executors for a field and define some basic operations.
 *
 * @ingroup Fields
 */
template<typename ValueType>
class Field
{

public:

    using FieldValueType = ValueType;

    /**
     * @brief Create an uninitialized Field with a given size on an executor
     * @param exec  Executor associated to the field
     * @param size  size of the field
     */
    Field(const Executor& exec, localIdx size) : size_(size), data_(nullptr), exec_(exec)
    {
        void* ptr = nullptr;
        std::visit(
            [&ptr, size](const auto& concreteExec)
            { ptr = concreteExec.alloc(size * sizeof(ValueType)); },
            exec_
        );
        data_ = static_cast<ValueType*>(ptr);
    }

    /**
     * @brief Create a Field with a given size from existing memory on an executor
     * @param exec  Executor associated to the field
     * @param in    Pointer to existing data
     * @param size  size of the field
     * @param hostExec Executor where the original data is located
     */
    Field(
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
            { ptr = concreteExec.alloc(size * sizeof(ValueType)); },
            exec_
        );
        data_ = static_cast<ValueType*>(ptr);
        std::visit(detail::deepCopyVisitor<ValueType>(size_, in, data_), hostExec, exec_);
    }

    /**
     * @brief Create a Field with a given size on an executor and uniform value
     * @param exec  Executor associated to the field
     * @param size  size of the field
     * @param value  the  default value
     */
    Field(const Executor& exec, localIdx size, ValueType value)
        : size_(size), data_(nullptr), exec_(exec)
    {
        void* ptr = nullptr;
        std::visit(
            [&ptr, size](const auto& execu) { ptr = execu.alloc(size * sizeof(ValueType)); }, exec_
        );
        data_ = static_cast<ValueType*>(ptr);
        NeoN::fill(*this, value);
    }

    /**
     * @brief Create a Field from a given vector of values on an executor
     * @param exec  Executor associated to the field
     * @param in a vector of elements to copy over
     */
    Field(const Executor& exec, std::vector<ValueType> in) : Field(exec, in.data(), in.size()) {}

    /**
     * @brief Create a Field as a copy of a Field on a specified executor
     * @param exec  Executor associated to the field
     * @param in a Field of elements to copy over
     */
    Field(const Executor& exec, const Field<ValueType>& in)
        : Field(exec, in.data(), in.size(), in.exec())
    {}

    /**
     * @brief Copy constructor, creates a new field with the same size and data as the parsed field.
     * @param rhs The field to copy from.
     */
    Field(const Field<ValueType>& rhs) : Field(rhs.exec(), rhs.data(), rhs.size(), rhs.exec()) {}

    /**
     * @brief Move constructor, moves the data from the parsed field to the new field.
     * @param rhs The field to move from.
     */
    Field(Field<ValueType>&& rhs) noexcept : size_(rhs.size_), data_(rhs.data_), exec_(rhs.exec_)
    {
        rhs.data_ = nullptr;
        rhs.size_ = 0;
    }

    /**
     * @brief Destroy the Field object.
     */
    ~Field()
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
        map(*this, f);
    }

    /**
     * @brief Copies the data to a new field on a specific executor.
     * @param dstExec The executor on which the data should be copied.
     * @returns A copy of the field on the host.
     */
    [[nodiscard]] Field<ValueType> copyToExecutor(Executor dstExec) const
    {
        if (dstExec == exec_) return *this;

        Field<ValueType> result(dstExec, size_);
        std::visit(detail::deepCopyVisitor(size_, data_, result.data()), exec_, dstExec);

        return result;
    }

    /**
     * @brief Returns a copy of the field back to the host.
     * @returns A copy of the field on the host.
     */
    [[nodiscard]] Field<ValueType> copyToHost() const { return copyToExecutor(SerialExecutor()); }

    /**
     * @brief Copies the data (from anywhere) to a parsed host field.
     * @param result The field into which the data must be copied. Must be
     * sized.
     *
     * @warning exits if the size of the result field is not the same as the
     * source field.
     */
    void copyToHost(Field<ValueType>& result)
    {
        NF_DEBUG_ASSERT(
            result.size() == size_, "Parsed Field size not the same as current field size"
        );
        result = copyToExecutor(SerialExecutor());
    }

    // ensures no return of device address on host --> invalid memory access
    ValueType& operator[](const localIdx i) = delete;

    // ensures no return of device address on host --> invalid memory access
    const ValueType& operator[](const localIdx i) const = delete;

    /**
     * @brief Assignment operator, Sets the field values to that of the passed value.
     * @param rhs The value to set the field to.
     */
    void operator=(const ValueType& rhs) { fill(*this, rhs); }

    /**
     * @brief Assignment operator, Sets the field values to that of the parsed field.
     * @param rhs The field to copy from.
     *
     * @warning This field will be sized to the size of the parsed field.
     */
    void operator=(const Field<ValueType>& rhs)
    {
        NF_ASSERT(exec_ == rhs.exec_, "Executors are not the same");
        if (this->size() != rhs.size())
        {
            this->resize(rhs.size());
        }
        setField(*this, rhs.view());
    }

    /**
     * @brief Arithmetic add operator, addition of a second field.
     * @param rhs The field to add with this field.
     * @returns The result of the addition.
     */
    Field<ValueType>& operator+=(const Field<ValueType>& rhs)
    {
        validateOtherField(rhs);
        add(*this, rhs);
        return *this;
    }

    /**
     * @brief Arithmetic subtraction operator, subtraction by a second field.
     * @param rhs The field to subtract from this field.
     * @returns The result of the subtraction.
     */
    Field<ValueType>& operator-=(const Field<ValueType>& rhs)
    {
        validateOtherField(rhs);
        sub(*this, rhs);
        return *this;
    }

    /**
     * @brief Arithmetic multiply operator, multiply by a second field.
     * @param rhs The field to subtract from this field.
     * @returns The result of the multiply.
     */
    [[nodiscard]] Field<ValueType> operator*(const Field<scalar>& rhs)
    {
        validateOtherField(rhs);
        Field<ValueType> result(exec_, size_);
        result = *this;
        mul(result, rhs);
        return result;
    }

    /**
     * @brief Arithmetic multiply operator, multiplies every cell in the field
     * by a scalar.
     * @param rhs The scalar to multiply with the field.
     * @returns The result of the multiplication.
     */
    [[nodiscard]] Field<ValueType> operator*(const scalar rhs)
    {
        Field<ValueType> result(exec_, size_);
        result = *this;
        scalarMul(result, rhs);
        return result;
    }

    /**
     * @brief Arithmetic multiply operator, multiplies this field by another field element-wise.
     * @param rhs The field to multiply with this field.
     * @returns The result of the element-wise multiplication.
     */
    Field<ValueType>& operator*=(const Field<scalar>& rhs)
    {
        validateOtherField(rhs);
        Field<ValueType>& result = *this;
        mul(result, rhs);
        return result;
    }

    /**
     * @brief Arithmetic multiply-assignment operator, multiplies every cell in the field
     * by a scalar and updates the field in place.
     * @param rhs The scalar to multiply with the field.
     */
    Field<ValueType>& operator*=(const scalar rhs)
    {
        Field<ValueType>& result = *this;
        scalarMul(result, rhs);
        return result;
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
                { ptr = exec.realloc(this->data_, size * sizeof(ValueType)); },
                exec_
            );
        }
        else
        {
            std::visit(
                [&ptr, size](const auto& exec) { ptr = exec.alloc(size * sizeof(ValueType)); },
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
    [[nodiscard]] ValueType* data() { return data_; }

    /**
     * @brief Direct access to the underlying field data
     * @return Pointer to the first cell data in the field.
     */
    [[nodiscard]] const ValueType* data() const { return data_; }

    /**
     * @brief Gets the executor associated with the field.
     * @return Reference to the executor.
     */
    [[nodiscard]] const Executor& exec() const { return exec_; }

    /**
     * @brief Gets the size of the field.
     * @return The size of the field.
     */
    [[nodiscard]] localIdx size() const { return size_; }

    /**
     * @brief Gets the size of the field.
     * @return The size of the field.
     */
    [[nodiscard]] label ssize() const { return static_cast<label>(size_); }

    /**
     * @brief Checks if the field is empty.
     * @return True if the field is empty, false otherwise.
     */
    [[nodiscard]] bool empty() const { return size() == 0; }

    // return of a temporary --> invalid memory access
    View<ValueType> view() && = delete;

    // return of a temporary --> invalid memory access
    View<const ValueType> view() const&& = delete;

    /**
     * @brief Gets the field as a view.
     * @return View of the field.
     */
    [[nodiscard]] View<ValueType> view() & { return View<ValueType>(data_, size_); }

    /**
     * @brief Gets the field as a view.
     * @return View of the field.
     */
    [[nodiscard]] View<const ValueType> view() const&
    {
        return View<const ValueType>(data_, size_);
    }

    // return of a temporary --> invalid memory access
    [[nodiscard]] View<ValueType> view(std::pair<localIdx, localIdx> range) && = delete;

    // return of a temporary --> invalid memory access
    [[nodiscard]] View<const ValueType> view(std::pair<localIdx, localIdx> range) const&& = delete;

    /**
     * @brief Gets a sub view of the field as a view.
     * @return View of the field.
     */
    [[nodiscard]] View<ValueType> view(std::pair<localIdx, localIdx> range) &
    {
        return View<ValueType>(data_ + range.first, range.second - range.first);
    }

    /**
     * @brief Gets a sub view of the field as a view.
     * @return View of the field.
     */
    [[nodiscard]] View<const ValueType> view(std::pair<localIdx, localIdx> range) const&
    {
        return View<const ValueType>(data_ + range.first, range.second - range.first);
    }

    /**
     * @brief Gets the range of the field.
     * @return The range of the field {0, size()}.
     */
    [[nodiscard]] std::pair<localIdx, localIdx> range() const { return {0, size()}; }

private:

    localIdx size_ {0};         //!< Size of the field.
    ValueType* data_ {nullptr}; //!< Pointer to the field data.
    const Executor exec_;       //!< Executor associated with the field. (CPU, GPU, openMP, etc.)

    /**
     * @brief Checks if two fields are the same size and have the same executor.
     * @param rhs The field to compare with.
     */
    void validateOtherField(const Field<ValueType>& rhs) const
    {
        NF_DEBUG_ASSERT(size() == rhs.size(), "Fields are not the same size.");
        NF_DEBUG_ASSERT(exec() == rhs.exec(), "Executors are not the same.");
    }
};

/**
 * @brief Arithmetic add operator, addition of two fields.
 * @param lhs The field to add with this field.
 * @param rhs The field to add with this field.
 * @returns The result of the addition.
 */
template<typename T>
[[nodiscard]] Field<T> operator+(Field<T> lhs, const Field<T>& rhs)
{
    lhs += rhs;
    return lhs;
}

/**
 * @brief Arithmetic subtraction operator, subtraction one field from another.
 * @param lhs The field to subtract from.
 * @param rhs The field to subtract by.
 * @returns The result of the subtraction.
 */
template<typename T>
[[nodiscard]] Field<T> operator-(Field<T> lhs, const Field<T>& rhs)
{
    lhs -= rhs;
    return lhs;
}

} // namespace NeoN
