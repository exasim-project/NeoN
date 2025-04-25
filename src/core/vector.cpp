// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors
#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/core/Vector.hpp"

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

template<typename ValueType>
Vector<ValueType>::Vector(const Executor& exec, localIdx size)
    : size_(size), data_(nullptr), exec_(exec)
{
    void* ptr = nullptr;
    std::visit(
        [&ptr, size](const auto& concreteExec)
        { ptr = concreteExec.alloc(static_cast<size_t>(size) * sizeof(ValueType)); },
        exec_
    );
    data_ = static_cast<ValueType*>(ptr);
}

template<typename ValueType>
Vector<ValueType>::Vector(
    const Executor& exec, const ValueType* in, localIdx size, Executor hostExec
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

template<typename ValueType>
Vector<ValueType>::Vector(const Executor& exec, localIdx size, ValueType value)
    : size_(size), data_(nullptr), exec_(exec)
{
    void* ptr = nullptr;
    std::visit(
        [&ptr, size](const auto& execu)
        { ptr = execu.alloc(static_cast<size_t>(size) * sizeof(ValueType)); },
        exec_
    );
    data_ = static_cast<ValueType*>(ptr);
    NeoN::fill(*this, value);
}

template<typename ValueType>
Vector<ValueType>::Vector(const Executor& exec, std::vector<ValueType> in)
    : Vector(exec, in.data(), static_cast<localIdx>(in.size()))
{}

template<typename ValueType>
Vector<ValueType>::Vector(const Executor& exec, const Vector<ValueType>& in)
    : Vector(exec, in.data(), in.size(), in.exec())
{}

template<typename ValueType>
Vector<ValueType>::Vector(const Vector<ValueType>& rhs)
    : Vector(rhs.exec(), rhs.data(), rhs.size(), rhs.exec())
{}

template<typename ValueType>
Vector<ValueType>::Vector(Vector<ValueType>&& rhs) noexcept
    : size_(rhs.size_), data_(rhs.data_), exec_(rhs.exec_)
{
    rhs.data_ = nullptr;
    rhs.size_ = 0;
}

template<typename ValueType>
Vector<ValueType>::~Vector()
{
    std::visit([this](const auto& exec) { exec.free(data_); }, exec_);
    data_ = nullptr;
}

template<typename ValueType>
[[nodiscard]] Vector<ValueType> Vector<ValueType>::copyToExecutor(Executor dstExec) const
{
    if (dstExec == exec_) return *this;

    Vector<ValueType> result(dstExec, size_);
    std::visit(detail::deepCopyVisitor(size_, data_, result.data()), exec_, dstExec);

    return result;
}

template<typename ValueType>
[[nodiscard]] Vector<ValueType>::Vector<ValueType> copyToHost() const
{
    return copyToExecutor(SerialExecutor());
}

template<typename ValueType>
void Vector<ValueType>::copyToHost(Vector<ValueType>& result)
{
    NF_DEBUG_ASSERT(
        result.size() == size_, "Parsed Vector size not the same as current field size"
    );
    result = copyToExecutor(SerialExecutor());
}

template<typename ValueType>
void Vector<ValueType>::operator=(const ValueType& rhs)
{
    fill(*this, rhs);
}

template<typename ValueType>
void Vector<ValueType>::operator=(const Vector<ValueType>& rhs)
{
    NF_ASSERT(exec_ == rhs.exec_, "Executors are not the same");
    if (this->size() != rhs.size())
    {
        this->resize(rhs.size());
    }
    setVector(*this, rhs.view());
}

template<typename ValueType>
Vector<ValueType>& Vector<ValueType>::operator+=(const Vector<ValueType>& rhs)
{
    validateOtherVector(rhs);
    add(*this, rhs);
    return *this;
}

template<typename ValueType>
Vector<ValueType>& Vector<ValueType>::operator-=(const Vector<ValueType>& rhs)
{
    validateOtherVector(rhs);
    sub(*this, rhs);
    return *this;
}

template<typename ValueType>
[[nodiscard]] Vector<ValueType> Vector<ValueType>::operator*(const Vector<scalar>& rhs)
{
    validateOtherVector(rhs);
    Vector<ValueType> result(exec_, size_);
    result = *this;
    mul(result, rhs);
    return result;
}

template<typename ValueType>
[[nodiscard]] Vector<ValueType> Vector<ValueType>::operator*(const scalar rhs)
{
    Vector<ValueType> result(exec_, size_);
    result = *this;
    scalarMul(result, rhs);
    return result;
}

template<typename ValueType>
Vector<ValueType>& Vector<ValueType>::operator*=(const Vector<scalar>& rhs)
{
    validateOtherVector(rhs);
    Vector<ValueType>& result = *this;
    mul(result, rhs);
    return result;
}

template<typename ValueType>
Vector<ValueType>& Vector<ValueType>::operator*=(const scalar rhs)
{
    Vector<ValueType>& result = *this;
    scalarMul(result, rhs);
    return result;
}

template<typename ValueType>
void Vector<ValueType>::resize(const localIdx size)
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

template<typename ValueType>
void Vector<ValueType>::validateOtherVector(const Vector<ValueType>& rhs) const
{
    NF_DEBUG_ASSERT(size() == rhs.size(), "Vectors are not the same size.");
    NF_DEBUG_ASSERT(exec() == rhs.exec(), "Executors are not the same.");
}


/**
 * @brief Arithmetic add operator, addition of two fields.
 * @param lhs The field to add with this field.
 * @param rhs The field to add with this field.
 * @returns The result of the addition.
 */
template<typename T>
[[nodiscard]] Vector<T> operator+(Vector<T> lhs, const Vector<T>& rhs)
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
[[nodiscard]] Vector<T> operator-(Vector<T> lhs, const Vector<T>& rhs)
{
    lhs -= rhs;
    return lhs;
}

} // namespace NeoN
